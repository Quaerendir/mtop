"""
mtop.export — machine-readable output: Prometheus exposition text and the
helpers `--json` / `--watch` share with it.

Everything here is a pure function of a snapshot dict, so the same data the
screen shows is what a scraper gets. Metric names carry the `mtop_` prefix
(this is what the exporter emits, not what Ollama itself would), grouped by
a second token: api, model, server, runner, gpu.

Stdlib only, no imports from the rest of the package (the bundler embeds
this file as a module, and __init__ imports from it — not the other way).
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Any

FOREVER_AFTER_SEC = 10 * 365 * 86400   # expires_at this far out == keep_alive -1

# ── parsing helpers ───────────────────────────────────────────────────────────

def parse_iso(iso_str: str) -> datetime | None:
    """Ollama/Go timestamps: nanosecond fractions, trailing Z, or an offset."""
    if not iso_str:
        return None
    s = iso_str.strip().replace("Z", "+00:00")
    if "." in s:
        # fromisoformat takes at most 6 fractional digits (pre-3.11 strictly).
        dot = s.index(".")
        end = dot + 1
        while end < len(s) and s[end].isdigit():
            end += 1
        s = s[:dot + 1] + s[dot + 1:end][:6].ljust(6, "0") + s[end:]
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?i?B?)\s*$", re.IGNORECASE)
_SIZE_UNITS = {"": 1, "B": 1,
               "KIB": 1 << 10, "MIB": 1 << 20, "GIB": 1 << 30, "TIB": 1 << 40, "PIB": 1 << 50,
               "KB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3, "TB": 1000 ** 4, "PB": 1000 ** 5,
               "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40, "P": 1 << 50}


def parse_size(text: str) -> int | None:
    """'1.536GiB' / '427MiB' / '14 GB' -> bytes (docker CLI and go-units spellings)."""
    m = _SIZE_RE.match(str(text or ""))
    if not m:
        return None
    unit = _SIZE_UNITS.get(m.group(2).upper())
    if unit is None:
        return None
    return int(float(m.group(1)) * unit)


def pct(text: Any) -> float | None:
    try:
        return float(str(text).strip().rstrip("%"))
    except (ValueError, TypeError):
        return None


def num(v: Any) -> float | None:
    try:
        f = float(str(v).strip())
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


# ── prometheus exposition ─────────────────────────────────────────────────────

def _esc(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(v: float) -> str:
    if v == math.inf:
        return "+Inf"
    if v == -math.inf:
        return "-Inf"
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


class _Writer:
    def __init__(self):
        self.lines: list[str] = []
        self._declared: set[str] = set()

    def metric(self, name: str, help_: str, kind: str = "gauge") -> None:
        if name not in self._declared:
            self._declared.add(name)
            self.lines.append(f"# HELP {name} {help_}")
            self.lines.append(f"# TYPE {name} {kind}")

    def sample(self, name: str, labels: dict[str, Any], value: float | None) -> None:
        if value is None:
            return
        lbl = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items() if v is not None)
        self.lines.append(f"{name}{{{lbl}}} {_fmt(value)}" if lbl else f"{name} {_fmt(value)}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def prometheus_text(snap: dict, version: str, now: float | None = None) -> str:
    """Render a snapshot as Prometheus text exposition format (0.0.4)."""
    now = time.time() if now is None else now
    w = _Writer()
    mode = snap.get("mode")
    base = {"mode": mode, "container": snap.get("container") if mode == "docker" else None}

    # ── meta / health ──
    healthy = snap.get("status") in ("running", "api-only") and bool(snap.get("models_ok"))
    w.metric("mtop_up", "1 when the Ollama source is up and its API answered")
    w.sample("mtop_up", {}, 1 if healthy else 0)
    w.metric("mtop_info", "mtop build and data source")
    w.sample("mtop_info", {**base, "version": version, "runtime": snap.get("runtime"),
                           "api_url": snap.get("api_url")}, 1)
    w.metric("mtop_snapshot_timestamp_seconds", "Unix time the snapshot was taken")
    w.sample("mtop_snapshot_timestamp_seconds", {}, now)

    # ── endpoints / models ──
    endpoints = snap.get("endpoints") or [{
        "label": snap.get("api_url"), "url": snap.get("api_url"),
        "models_ok": snap.get("models_ok"), "models": snap.get("models") or [],
        "version": (snap.get("server") or {}).get("version"),
    }]
    w.metric("mtop_api_up", "1 when /api/ps answered on this endpoint")
    w.metric("mtop_ollama_info", "Ollama server version per endpoint")
    w.metric("mtop_models_loaded", "Models currently loaded on the endpoint")
    w.metric("mtop_model_size_bytes", "Total size of a loaded model")
    w.metric("mtop_model_vram_bytes", "Portion of a loaded model in accelerator memory")
    w.metric("mtop_model_context_length", "Context window of a loaded model")
    w.metric("mtop_model_expires_seconds",
             "Seconds until the model is unloaded; +Inf for keep_alive -1")
    for ep in endpoints:
        e = {"endpoint": ep.get("label"), "url": ep.get("url")}
        w.sample("mtop_api_up", e, 1 if ep.get("models_ok") else 0)
        if ep.get("version"):
            w.sample("mtop_ollama_info", {**e, "version": ep["version"]}, 1)
        models = ep.get("models") or []
        if ep.get("models_ok"):
            w.sample("mtop_models_loaded", e, len(models))
        for m in models:
            lbl = {**e, "model": m.get("name"), "family": (m.get("details") or {}).get("family"),
                   "quantization": (m.get("details") or {}).get("quantization_level")}
            w.sample("mtop_model_size_bytes", lbl, num(m.get("size")))
            w.sample("mtop_model_vram_bytes", lbl, num(m.get("size_vram")))
            w.sample("mtop_model_context_length", lbl, num(m.get("context_length")))
            t = parse_iso(m.get("expires_at") or "")
            if t is not None and t.year > 1:
                left = (t - datetime.now(timezone.utc)).total_seconds()
                w.sample("mtop_model_expires_seconds", lbl,
                         math.inf if left >= FOREVER_AFTER_SEC else round(left, 3))

    # ── server (container / process) ──
    stats = snap.get("res_stats") or {}
    if stats:
        w.metric("mtop_server_cpu_percent",
                 "CPU of the server process tree or container, summed across cores")
        cpu = stats.get("cpu_pct")
        w.sample("mtop_server_cpu_percent", base,
                 cpu if cpu is not None else pct(stats.get("cpu")))
        w.metric("mtop_server_cpu_limit_cores", "Cores the server may use (cgroup/quota/affinity)")
        w.sample("mtop_server_cpu_limit_cores", base, num(snap.get("cpu_limit")))
        used = stats.get("mem_used_bytes")
        limit = stats.get("mem_limit_bytes")
        if used is None and stats.get("mem_usage"):
            parts = str(stats["mem_usage"]).split("/")
            used = parse_size(parts[0]) if parts else None
            limit = parse_size(parts[1]) if len(parts) > 1 else None
        w.metric("mtop_server_memory_bytes", "Memory of the server process tree or container "
                                             "(kind label: pss, rss, or cgroup)")
        w.sample("mtop_server_memory_bytes",
                 {**base, "kind": stats.get("mem_kind") or "cgroup"}, num(used))
        w.metric("mtop_server_memory_limit_bytes", "Memory available to the server")
        w.sample("mtop_server_memory_limit_bytes", base, num(limit))
        if stats.get("procs") is not None:
            w.metric("mtop_server_processes", "Processes in the server tree (server + runners)")
            w.sample("mtop_server_processes", base, num(stats.get("procs")))
    if snap.get("uptime_sec") is not None:
        w.metric("mtop_server_uptime_seconds", "Seconds since the server (container) started")
        w.sample("mtop_server_uptime_seconds", base, num(snap.get("uptime_sec")))

    # ── runners ──
    runners = snap.get("runners") or []
    if runners:
        w.metric("mtop_runner_info", "Effective inference config of a model runner process")
        w.metric("mtop_runner_rss_bytes", "Resident set of the runner process (host RAM)")
        w.metric("mtop_runner_vram_bytes", "Ollama's size_vram for the model this runner serves")
        w.metric("mtop_runner_gpu_memory_bytes", "Device memory charged to the runner (NVML)")
        for r in runners:
            pid = r.get("pid")
            model = r.get("model_name") or (r.get("digest") or "")[:12] or None
            kv_k, kv_v = r.get("kv_k"), r.get("kv_v")
            kv = kv_k if kv_k == kv_v else "/".join(x for x in (kv_k, kv_v) if x) or None
            w.sample("mtop_runner_info", {
                "pid": pid, "model": model, "engine": r.get("engine"),
                "ctx": r.get("ctx"), "batch": r.get("batch"), "ubatch": r.get("ubatch"),
                "parallel": r.get("parallel"), "flash_attn": r.get("flash_attn"),
                "kv_cache": kv, "ngl": r.get("ngl"), "threads": r.get("threads"),
                "load_mode": ("dio" if r.get("direct_io") else r.get("load_mode")),
                "gpu": ",".join(r.get("gpu") or []) or None,
            }, 1)
            lbl = {"pid": pid, "model": model}
            w.sample("mtop_runner_rss_bytes", lbl, num(r.get("rss")))
            w.sample("mtop_runner_vram_bytes", lbl, num(r.get("vram")))
            if r.get("gpu_mem_mib"):
                w.sample("mtop_runner_gpu_memory_bytes", lbl, r["gpu_mem_mib"] * (1 << 20))

    # ── gpus ──
    gpus = snap.get("gpus") or []
    if gpus:
        w.metric("mtop_gpu_info", "GPU identity")
        w.metric("mtop_gpu_utilization_percent", "GPU compute utilization")
        w.metric("mtop_gpu_memory_used_bytes", "Device memory in use (system RAM on unified parts)")
        w.metric("mtop_gpu_memory_total_bytes", "Device memory total")
        w.metric("mtop_gpu_temperature_celsius", "GPU temperature")
        w.metric("mtop_gpu_power_watts", "GPU power draw")
        w.metric("mtop_gpu_gtt_used_bytes", "GTT (host memory) used by an AMD dGPU")
        w.metric("mtop_gpu_gtt_total_bytes", "GTT pool size of an AMD dGPU")
        w.metric("mtop_gpu_process_memory_bytes", "Device memory per compute process (NVML)")
        for g in gpus:
            key = {"vendor": g.get("vendor"), "index": g.get("index")}
            w.sample("mtop_gpu_info", {**key, "name": g.get("name"),
                                       "unified": "1" if g.get("unified") else "0"}, 1)
            w.sample("mtop_gpu_utilization_percent", key, num(g.get("util")))
            mu, mt = num(g.get("mem_used")), num(g.get("mem_total"))
            w.sample("mtop_gpu_memory_used_bytes", key, mu * (1 << 20) if mu is not None else None)
            w.sample("mtop_gpu_memory_total_bytes", key, mt * (1 << 20) if mt is not None else None)
            w.sample("mtop_gpu_temperature_celsius", key, num(g.get("temp")))
            w.sample("mtop_gpu_power_watts", key, num(g.get("power")))
            gu, gt = num(g.get("gtt_used")), num(g.get("gtt_total"))
            w.sample("mtop_gpu_gtt_used_bytes", key, gu * (1 << 20) if gu is not None else None)
            w.sample("mtop_gpu_gtt_total_bytes", key, gt * (1 << 20) if gt is not None else None)
            for pr in g.get("procs") or []:
                if pr.get("mem_mib"):
                    w.sample("mtop_gpu_process_memory_bytes",
                             {**key, "pid": pr.get("pid"), "model": pr.get("model")},
                             pr["mem_mib"] * (1 << 20))
    return w.text()
