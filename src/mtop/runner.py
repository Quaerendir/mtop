"""
mtop.runner — the model runner's argv, the server's environment, and the
joins between runners, /api/ps models and GPUs.

Everything Ollama negotiated (flash attention, KV dtype, batch, layer split)
exists only on the runner's command line; /api/ps reports the context length
and stops. This module turns that argv into a dict and links it to what the
API and the GPU drivers know.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .util import to_float

ENV_PREFIXES = ("OLLAMA_", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "HIP_", "HSA_", "ROCR_", "GPU_DEVICE_ORDINAL", "GGML_", "LLAMA_")


ENV_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS")


def inference_env(pairs: Iterable[str]) -> dict[str, str]:
    """Filter KEY=VALUE strings down to the inference-relevant ones, masked."""
    out: dict[str, str] = {}
    for kv in pairs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if not k.startswith(ENV_PREFIXES):
            continue
        if any(m in k.upper() for m in ENV_SECRET_MARKERS) and v:
            v = "••••"
        out[k] = v
    return dict(sorted(out.items()))


def link_runners_to_gpus(runners: list[dict] | None, gpus: list[dict] | None) -> None:
    """Join runner processes to cards by PID, both ways, in place.

    Each runner gains ``gpu`` (["nvidia:0", ...]) and ``gpu_mem_mib``; each
    GPU's ``procs`` entries gain ``model`` when the PID is a known runner.
    PIDs are host-namespace on both sides (NVML reports host PIDs; runner PIDs
    come from the host /proc walk), so a runner discovered through the
    in-container exec fallback — container-namespace PIDs — does not link.
    """
    if not runners or not gpus:
        return
    by_pid = {r["pid"]: r for r in runners if r.get("pid")}
    for r in by_pid.values():
        r.pop("gpu", None)
        r.pop("gpu_mem_mib", None)
    for g in gpus:
        for proc in g.get("procs") or []:
            r = by_pid.get(proc.get("pid"))
            if r is None:
                proc.pop("model", None)
                continue
            proc["model"] = r.get("model_name") or r.get("digest", "")[:12]
            tag = f"{g.get('vendor', '?')}:{g.get('index', '?')}"
            r.setdefault("gpu", []).append(tag)
            if proc.get("mem_mib"):
                r["gpu_mem_mib"] = r.get("gpu_mem_mib", 0) + proc["mem_mib"]


# Every name this process has shipped under. Kept as a hint for the *display*
# only — tree membership is decided by ppid, never by these (see process_tree).
RUNNER_BASENAMES = {"llama-server", "ollama_llama_server", "ollama-runner"}


# argv flag -> (key, takes_value). Long and short spellings both appear
# depending on the llama.cpp vintage Ollama vendored.
_RUNNER_FLAGS: dict[str, tuple[str, bool]] = {
    "--model": ("model", True), "-m": ("model", True),
    "--ctx-size": ("ctx", True), "-c": ("ctx", True),
    "--batch-size": ("batch", True), "-b": ("batch", True),
    "--ubatch-size": ("ubatch", True), "-ub": ("ubatch", True),
    "--parallel": ("parallel", True), "-np": ("parallel", True),
    "--n-gpu-layers": ("ngl", True), "--gpu-layers": ("ngl", True),
    "-ngl": ("ngl", True),
    "--cache-type-k": ("kv_k", True), "-ctk": ("kv_k", True),
    "--cache-type-v": ("kv_v", True), "-ctv": ("kv_v", True),
    "--kv-cache-type": ("kv", True),      # new engine: one flag for K and V
    "--threads": ("threads", True), "-t": ("threads", True),
    "--ollama-engine": ("ollama_engine", False),
    "--multiuser-cache": ("multiuser_cache", False),
    "--mmproj": ("mmproj", True),
    "--port": ("port", True),
    "--tensor-split": ("tensor_split", True), "-ts": ("tensor_split", True),
    "--main-gpu": ("main_gpu", True), "-mg": ("main_gpu", True),
    "--direct-io": ("direct_io", False),   # pre-0.33 spelling
    "--load-mode": ("load_mode", True),    # 0.33+: mmap | dio | ...
    "--no-mmap": ("no_mmap", False),
    "--context-shift": ("context_shift", False),
}


def parse_runner_argv(args: list[str]) -> dict | None:
    """Extract the effective inference config from a runner's argv.

    This is the only place the *negotiated* settings are visible. `/api/ps`
    reports the context length and nothing else; everything Ollama worked out
    between the Modelfile, the environment and its own heuristics — flash
    attention, KV cache dtype, batch sizes, layer split — exists only here.
    Notably `OLLAMA_FLASH_ATTENTION=1` shows up as `--flash-attn on` while an
    unset environment yields `--flash-attn auto`, which is how you tell whether
    the variable actually reached the server.
    """
    if not args:
        return None
    base = os.path.basename(args[0])
    if base not in RUNNER_BASENAMES and not (
            base == "ollama" and "runner" in args[1:]):
        return None

    out: dict[str, Any] = {}
    i = 1
    while i < len(args):
        spec = _RUNNER_FLAGS.get(args[i])
        if spec is None:
            # --flash-attn takes an optional value: 'on'/'off'/'auto' in recent
            # builds, bare (implying on) in older ones.
            if args[i] in ("--flash-attn", "-fa"):
                nxt = args[i + 1] if i + 1 < len(args) else None
                if nxt and not nxt.startswith("-"):
                    out["flash_attn"] = nxt
                    i += 2
                    continue
                out["flash_attn"] = "on"
            i += 1
            continue
        key, takes_value = spec
        if not takes_value:
            out[key] = True
            i += 1
            continue
        if i + 1 < len(args):
            out[key] = args[i + 1]
        i += 2

    # The Ollama engine spells the KV cache dtype as one flag; fold it into
    # the K/V pair the renderer already understands.
    kv = out.pop("kv", None)
    if kv is not None:
        out.setdefault("kv_k", kv)
        out.setdefault("kv_v", kv)
    out["engine"] = "ollama" if out.pop("ollama_engine", False) else "llama"

    model = out.get("model", "")
    digest = ""
    if model:
        stem = os.path.basename(str(model))
        if stem.startswith("sha256-"):
            digest = stem[len("sha256-"):]
    out["digest"] = digest
    return out


def _claim(runner: dict, model: dict) -> None:
    """Copy the API's view of a matched model onto its runner.

    `size_vram` matters most: on accelerator-backed hosts the weights are device
    allocations that never appear in the runner's VmRSS. On a GB10 Spark an 82 GB
    model shows a 7.5 GiB RSS — the bytes come out of the same unified pool
    (/proc/meminfo sees them) but are not charged to the process.
    """
    runner["model_name"] = model.get("name", "")
    runner["vram"] = model.get("size_vram")
    runner["model_size"] = model.get("size")


def match_runners_to_models(runners: list[dict], models: list[dict]) -> None:
    """Best-effort blob -> tag mapping, mutating `runners` in place.

    The `--model` path is a blob digest; `/api/tags` and `/api/ps` expose the
    *manifest* digest, so there is no direct join. Two honest heuristics, in
    order, and a shortened digest when neither is conclusive — a wrong tag on a
    monitoring screen is worse than no tag:

    1. context length, when it uniquely identifies one loaded model;
    2. one unmatched runner left facing one unmatched model.
    """
    unclaimed = list(models)
    for r in runners:
        ctx = to_float(r.get("ctx"))
        if ctx is None:
            continue
        hits = [m for m in unclaimed if to_float(m.get("context_length")) == ctx]
        if len(hits) == 1:
            _claim(r, hits[0])
            unclaimed.remove(hits[0])
    rest = [r for r in runners if not r.get("model_name")]
    if len(rest) == 1 and len(unclaimed) == 1:
        _claim(rest[0], unclaimed[0])


def processor_label(size: int | float | None, size_vram: int | float | None) -> str:
    """The PROCESSOR column exactly as `ollama ps` computes it (cmd/cmd.go)."""
    size = size or 0
    size_vram = size_vram or 0
    if size_vram == 0:
        return "100% CPU"
    if size_vram == size:
        return "100% GPU"
    if size_vram > size or size == 0:
        return "Unknown"
    cpu = round((size - size_vram) / size * 100)
    return f"{cpu}%/{100 - cpu}% CPU/GPU"
