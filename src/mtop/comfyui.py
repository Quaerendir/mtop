"""
mtop.comfyui — the ComfyUI side of `--backend comfyui`: which models a
workflow loads, and the model catalog.

ComfyUI has no `/api/ps`: nothing in its API says which models sit in memory.
What it does say is which loader nodes the running job and the recent jobs
contain (`/queue`, `/history`). ComfyUI keeps a loaded model cached after the
job that loaded it, until memory runs short, a `/free` call or a restart
(which also empties the history), so the models of the recent jobs are the
best available guess at what is resident — shown as *recent*, never as loaded.
Only the job running right now gives a certain answer (*in use*).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Model weights, by extension. A loader input whose value ends in one of these
# names a model file; LoadImage's `image` (a .png) does not.
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")

# Loader input name prefix -> kind shown in the KIND column. Covers the stock
# loaders (CheckpointLoaderSimple, UNETLoader, CLIPLoader, DualCLIPLoader,
# VAELoader, LoraLoader, ControlNetLoader, UpscaleModelLoader, ...) and the
# custom ones that follow the same naming (UnetLoaderGGUF: unet_name).
KIND_BY_INPUT = (
    ("ckpt", "checkpoint"),
    ("unet", "unet"),
    ("clip_vision", "clip_vision"),
    ("clip", "clip"),
    ("vae", "vae"),
    ("lora", "lora"),
    ("control_net", "controlnet"),
    ("style_model", "style"),
    ("upscale_model", "upscale"),
)

# /models folders that hold no model files.
NON_MODEL_FOLDERS = frozenset({"configs", "custom_nodes", "datasets"})

HISTORY_ITEMS = 20       # jobs read from /history per cycle


def input_kind(key: str, class_type: str) -> str:
    for prefix, kind in KIND_BY_INPUT:
        if key.startswith(prefix):
            return kind
    return class_type


def graph_models(graph: Any) -> list[tuple[str, str]]:
    """(file, kind) for every model a workflow graph loads, in node order.

    A model is a string input of a node whose class name contains "load" and
    whose value is a model file name. One file loaded by two nodes is listed
    once.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    if not isinstance(graph, dict):
        return out
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct = str(node.get("class_type", ""))
        if "load" not in ct.lower():
            continue
        for k, v in (node.get("inputs") or {}).items():
            if (isinstance(v, str) and v.lower().endswith(MODEL_EXTS)
                    and v not in seen):
                seen.add(v)
                out.append((v, input_kind(k, ct)))
    return out


def _graph(item: Any) -> Any:
    """The workflow graph of a /queue or /history prompt tuple
    [number, prompt_id, graph, extra_data, outputs_to_execute]."""
    return item[2] if isinstance(item, (list, tuple)) and len(item) > 2 else None


def _ended_ms(entry: dict) -> int | None:
    """When a history job finished (ms since the epoch), from its status messages."""
    ts = None
    for msg in (entry.get("status") or {}).get("messages") or []:
        if isinstance(msg, (list, tuple)) and len(msg) == 2 and isinstance(msg[1], dict):
            t = msg[1].get("timestamp")
            if isinstance(t, (int, float)):
                ts = max(ts or 0, int(t))
    return ts


def queue_summary(queue: Any) -> dict:
    """{running, pending, prompt_id} from /queue."""
    q = queue if isinstance(queue, dict) else {}
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    pid = running[0][1] if running and isinstance(running[0], (list, tuple)) \
        and len(running[0]) > 1 else None
    return {"running": len(running), "pending": len(pending), "prompt_id": pid}


def models_in_use(queue: Any, history: Any) -> list[dict]:
    """Models of the running job (state `in use`), then of recent jobs
    (`recent`), newest first.

    `running` is True only for the running job's models — the one case where
    the model is certainly in memory. `last_used` is when the newest job that
    loaded the model finished (ISO 8601, UTC); `jobs` how many of the
    recent jobs loaded it.
    """
    models: dict[str, dict] = {}
    q = queue if isinstance(queue, dict) else {}
    for item in q.get("queue_running") or []:
        for name, kind in graph_models(_graph(item)):
            models.setdefault(name, {"name": name, "kind": kind, "state": "in use",
                                     "running": True, "last_used": None, "jobs": 0})
    entries = list(history.values()) if isinstance(history, dict) else []
    # /history lists jobs oldest first.
    for entry in reversed(entries[-HISTORY_ITEMS:]):
        if not isinstance(entry, dict):
            continue
        ended = _ended_ms(entry)
        for name, kind in graph_models(_graph(entry.get("prompt"))):
            m = models.setdefault(name, {"name": name, "kind": kind, "state": "recent",
                                         "running": False, "last_used": None, "jobs": 0})
            m["jobs"] += 1
            if m["last_used"] is None and ended is not None:
                m["last_used"] = datetime.fromtimestamp(
                    ended / 1000, timezone.utc).isoformat(timespec="seconds")
    return list(models.values())


def is_hf_cache_entry(name: str) -> bool:
    """A file inside a Hugging Face hub cache (`models--org--name/...`), which
    ComfyUI lists in every folder an extra_model_paths entry points at it."""
    return any(part.startswith("models--") for part in name.replace("\\", "/").split("/"))


def catalog_counts(lists: dict[str, list]) -> dict:
    """{folder: n} of the non-empty model folders, HF hub cache files left out
    and counted once (unique paths) under `hf_cache`."""
    counts: dict[str, int] = {}
    hf: set[str] = set()
    for folder, files in lists.items():
        n = 0
        for f in files or []:
            if not isinstance(f, str):
                continue
            if is_hf_cache_entry(f):
                hf.add(f)
            else:
                n += 1
        if n:
            counts[folder] = n
    return {"folders": counts, "hf_cache": len(hf)}


def _running_item(queue: Any, prompt_id: str | None) -> Any:
    q = queue if isinstance(queue, dict) else {}
    for item in q.get("queue_running") or []:
        if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id:
            return item
    return None


def node_class(queue: Any, prompt_id: str | None, node: str | None) -> str | None:
    """class_type of a node of the running job's graph (`KSampler`)."""
    graph = _graph(_running_item(queue, prompt_id))
    n = graph.get(str(node)) if isinstance(graph, dict) and node is not None else None
    return str(n.get("class_type")) if isinstance(n, dict) and n.get("class_type") else None


def queued_by_client(queue: Any, prompt_id: str | None) -> bool:
    """True when the running job was queued with a client_id — its progress
    then goes to that client only (see mtop.comfywatch)."""
    item = _running_item(queue, prompt_id)
    extra = item[3] if item is not None and len(item) > 3 else None
    return bool(isinstance(extra, dict) and extra.get("client_id"))


def last_job(history: Any) -> dict | None:
    """The newest finished job: {prompt_id, outcome, took (s), ended (ISO)}.
    outcome is success, error or interrupted."""
    if not isinstance(history, dict) or not history:
        return None
    pid, entry = list(history.items())[-1]
    if not isinstance(entry, dict):
        return None
    status = entry.get("status") or {}
    times: dict[str, int] = {}
    for msg in status.get("messages") or []:
        if isinstance(msg, (list, tuple)) and len(msg) == 2 and isinstance(msg[1], dict):
            t = msg[1].get("timestamp")
            if isinstance(t, (int, float)):
                times[str(msg[0])] = int(t)
    outcome = ("interrupted" if "execution_interrupted" in times
               else str(status.get("status_str") or "unknown"))
    start, end = times.get("execution_start"), _ended_ms(entry)
    return {"prompt_id": pid, "outcome": outcome,
            "took": round((end - start) / 1000, 1) if start and end else None,
            "ended": (datetime.fromtimestamp(end / 1000, timezone.utc)
                      .isoformat(timespec="seconds") if end else None)}
