"""--backend comfyui: models from the workflow graphs of the running and recent
jobs, the catalog, the result shape, CLI defaults, the screen and Prometheus."""

import os
import sys

import pytest

import mtop
from conftest import FakeWin

from mtop import comfyui, export, procfs


def _graph(*loaders):
    """A workflow graph: one node per (class_type, inputs), plus a sampler."""
    g = {str(i): {"class_type": ct, "inputs": inputs}
         for i, (ct, inputs) in enumerate(loaders, 1)}
    g["99"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1}}
    return g


QWEN = _graph(("UNETLoader", {"unet_name": "qwen_image.safetensors", "weight_dtype": "default"}),
              ("CLIPLoader", {"clip_name": "qwen3vl_8b.safetensors", "type": "qwen_image"}),
              ("VAELoader", {"vae_name": "qwen_vae.safetensors"}),
              ("LoadImage", {"image": "ref.png"}))
FLUX = _graph(("CheckpointLoaderSimple", {"ckpt_name": "flux1-schnell.safetensors"}),
              ("LoraLoader", {"lora_name": "style.safetensors", "strength_model": 1.0}),
              ("VAELoader", {"vae_name": "qwen_vae.safetensors"}))


def _hist_entry(graph, ended_ms):
    return {"prompt": [1, "id", graph, {}, ["9"]],
            "status": {"status_str": "success", "completed": True, "messages": [
                ["execution_start", {"prompt_id": "id", "timestamp": ended_ms - 5000}],
                ["execution_success", {"prompt_id": "id", "timestamp": ended_ms}]]}}


# ── workflow graphs ─────────────────────────────────────────────────────────

def test_graph_models_finds_loader_files_and_skips_images_and_non_loaders():
    assert comfyui.graph_models(QWEN) == [("qwen_image.safetensors", "unet"),
                                         ("qwen3vl_8b.safetensors", "clip"),
                                         ("qwen_vae.safetensors", "vae")]


def test_graph_models_kinds_and_custom_loaders():
    g = _graph(("DualCLIPLoader", {"clip_name1": "t5.safetensors", "clip_name2": "l.safetensors"}),
               ("UnetLoaderGGUF", {"unet_name": "flux-Q4.gguf"}),
               ("CLIPVisionLoader", {"clip_name": "vision.safetensors"}),
               ("SomeModelLoader", {"model_name": "x.pth"}))
    assert comfyui.graph_models(g) == [("t5.safetensors", "clip"), ("l.safetensors", "clip"),
                                       ("flux-Q4.gguf", "unet"),
                                       ("vision.safetensors", "clip"),
                                       ("x.pth", "SomeModelLoader")]


def test_graph_models_tolerates_junk():
    assert comfyui.graph_models(None) == []
    assert comfyui.graph_models({"1": "x", "2": {"class_type": "VAELoader"}}) == []


# ── running / recent ────────────────────────────────────────────────────────

def test_running_job_models_are_in_use_and_history_ones_recent():
    queue = {"queue_running": [[5, "p5", FLUX, {}, ["9"]]], "queue_pending": []}
    history = {"a": _hist_entry(QWEN, 1_700_000_000_000),
               "b": _hist_entry(QWEN, 1_700_000_100_000)}
    models = comfyui.models_in_use(queue, history)
    by = {m["name"]: m for m in models}
    assert [m["name"] for m in models[:3]] == ["flux1-schnell.safetensors",
                                               "style.safetensors", "qwen_vae.safetensors"]
    assert by["flux1-schnell.safetensors"]["state"] == "in use"
    assert by["flux1-schnell.safetensors"]["running"] is True
    # shared with the running job: stays in use, but the history still counts
    assert by["qwen_vae.safetensors"]["state"] == "in use"
    assert by["qwen_vae.safetensors"]["jobs"] == 2
    q = by["qwen_image.safetensors"]
    assert q["state"] == "recent" and q["running"] is False and q["jobs"] == 2
    assert q["last_used"] == "2023-11-14T22:15:00+00:00"       # the newer job


def test_history_is_capped_to_the_newest_jobs():
    history = {f"j{i}": _hist_entry(FLUX if i == 0 else QWEN, 1_700_000_000_000 + i)
               for i in range(comfyui.HISTORY_ITEMS + 1)}
    names = {m["name"] for m in comfyui.models_in_use({}, history)}
    assert "flux1-schnell.safetensors" not in names


def test_queue_summary():
    q = {"queue_running": [[1, "abc", {}, {}, []]], "queue_pending": [[2, "d"], [3, "e"]]}
    assert comfyui.queue_summary(q) == {"running": 1, "pending": 2, "prompt_id": "abc"}
    assert comfyui.queue_summary(None) == {"running": 0, "pending": 0, "prompt_id": None}


# ── catalog ─────────────────────────────────────────────────────────────────

def test_catalog_counts_drops_empty_folders_and_counts_hf_cache_once():
    hf = ["models--Qwen--Q/snapshots/abc/model-00001.safetensors",
          "models--Qwen--Q/snapshots/abc/model-00002.safetensors"]
    lists = {"checkpoints": ["a.safetensors"] + hf, "vae": ["v.safetensors"] + hf,
             "loras": [], "controlnet": hf}
    assert comfyui.catalog_counts(lists) == {"folders": {"checkpoints": 1, "vae": 1},
                                             "hf_cache": 2}


# ── collector ───────────────────────────────────────────────────────────────

def _collector(monkeypatch, routes, calls=None):
    def get(url, *a, **k):
        path = url.split("8188", 1)[1]
        if calls is not None:
            calls.append(path)
        return routes.get(path.split("?")[0], (False, "HTTP 404"))
    monkeypatch.setattr(mtop.util, "http_get_json", get)
    return mtop.Collector(container="comfyui", api_url="http://localhost:8188", interval=1.0,
                          show_gpu=False, mode="api", backend="comfyui")


ROUTES = {
    "/queue": (True, {"queue_running": [], "queue_pending": [[1, "x"]]}),
    "/history": (True, {"a": _hist_entry(QWEN, 1_700_000_000_000)}),
    "/models": (True, ["checkpoints", "vae", "custom_nodes"]),
    "/models/checkpoints": (True, ["a.safetensors", "b.safetensors"]),
    "/models/vae": (True, ["qwen_vae.safetensors"]),
    "/system_stats": (True, {"system": {"comfyui_version": "0.37.0"}}),
}


def test_result_shape_and_catalog_is_cached(monkeypatch):
    calls = []
    c = _collector(monkeypatch, ROUTES, calls)
    try:
        res = c._comfyui_result(c.primary)
        assert res["models_ok"] and res["queue"]["pending"] == 1
        assert res["catalog"] == {"folders": {"checkpoints": 2, "vae": 1}, "hf_cache": 0}
        assert [m["name"] for m in res["models"]] == ["qwen_image.safetensors",
                                                     "qwen3vl_8b.safetensors",
                                                     "qwen_vae.safetensors"]
        assert "/models/custom_nodes" not in calls
        n = len(calls)
        c._comfyui_result(c.primary)
        assert calls[n:] == ["/queue", f"/history?max_items={comfyui.HISTORY_ITEMS}"]
    finally:
        c.close()


def test_queue_failure_is_an_api_error_and_history_failure_is_not(monkeypatch):
    c = _collector(monkeypatch, {**ROUTES, "/queue": (False, "connection refused")})
    try:
        res = c._comfyui_result(c.primary)
        assert not res["models_ok"] and res["models_err"] == "connection refused"
    finally:
        c.close()
    c = _collector(monkeypatch, {**ROUTES, "/history": (False, "HTTP 500")})
    try:
        res = c._comfyui_result(c.primary)
        assert res["models_ok"] and res["models"] == []
    finally:
        c.close()


def test_version_comes_from_system_stats(monkeypatch):
    c = _collector(monkeypatch, ROUTES)
    try:
        snap = c.collect(100.0)                 # past SLOW_FLOOR: slow path runs
        assert snap["server"]["version"] == "0.37.0"
        assert snap["backend"] == "comfyui" and snap["models_ok"]
    finally:
        c.close()


def test_no_runner_parsing_for_comfyui(monkeypatch):
    c = _collector(monkeypatch, ROUTES)
    try:
        assert c._parse_argv(["/usr/bin/ollama", "runner", "--model", "x"]) is None
    finally:
        c.close()


# ── process lookup ──────────────────────────────────────────────────────────

def test_find_comfyui_pids_needs_comfy_in_script_path_or_cwd(monkeypatch):
    argv = {"10": ["python3", "/opt/ComfyUI/main.py", "--port", "8188"],
            "11": ["python3", "main.py", "--listen"],
            "12": ["python3", "main.py"],
            "13": ["/usr/bin/bash", "/opt/ComfyUI/main.py"],
            "14": ["python3", "-u", "/srv/other/app.py"]}
    cwd = {"11": "/home/u/comfyui", "12": "/home/u/someproject"}
    monkeypatch.setattr(procfs.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(procfs.os, "listdir", lambda p: list(argv) + ["self"])
    monkeypatch.setattr(procfs, "read_proc_cmdline", lambda pid: argv.get(str(pid)))
    monkeypatch.setattr(procfs.os, "readlink", lambda p: cwd[p.split("/")[2]])
    assert sorted(procfs.find_comfyui_pids()) == [10, 11]


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse(monkeypatch, *argv):
    seen = {}
    monkeypatch.setattr(sys, "argv", ["mtop", *argv])
    monkeypatch.setattr(mtop.cli, "headless_main", lambda args: seen.setdefault("a", args) and 0)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    with pytest.raises(SystemExit):
        mtop.cli.main()
    return seen.get("a")


def test_cli_defaults_follow_the_backend(monkeypatch):
    args = _parse(monkeypatch, "--backend", "comfyui", "--json")
    assert args.api_url == "http://localhost:8188" and args.container == "comfyui"
    args = _parse(monkeypatch, "--backend", "comfyui", "--json", "-c", "comfy", "-u",
                  "http://gpu:8189")
    assert args.api_url == "http://gpu:8189" and args.container == "comfy"


def test_cli_rejects_control_with_comfyui(monkeypatch, capsys):
    assert _parse(monkeypatch, "--backend", "comfyui", "--control") is None
    assert "--control is not available" in capsys.readouterr().err


# ── screen ──────────────────────────────────────────────────────────────────

def _snap(**ep):
    e = {"label": "l", "url": "u", "models_ok": True,
         "queue": {"running": 1, "pending": 2},
         "catalog": {"folders": {"checkpoints": 2, "vae": 1}, "hf_cache": 61},
         "models": [{"name": "flux1-schnell.safetensors", "kind": "checkpoint",
                     "state": "in use", "running": True, "last_used": None, "jobs": 0},
                    {"name": "qwen_vae.safetensors", "kind": "vae", "state": "recent",
                     "running": False, "last_used": "2020-01-01T00:00:00+00:00", "jobs": 3}]}
    e.update(ep)
    return {"backend": "comfyui", "endpoints": [e]}


def test_models_table():
    w = FakeWin(rows=10, cols=160)
    mtop.render_models(w, 0, _snap())
    text = w.text()
    assert " MODELS " in text and "LOADED MODELS" not in text
    assert "queue: 1 running · 2 pending" in text
    assert "catalog: 2 checkpoints · 1 vae · 61 in HF cache dirs" in text
    flux = next(ln for ln in text.splitlines() if "flux1-schnell" in ln)
    assert "checkpoint" in flux and "in use" in flux and "now" in flux
    vae = next(ln for ln in text.splitlines() if "qwen_vae" in ln)
    assert "recent" in vae and "ago" in vae


def test_models_table_empty_and_error():
    w = FakeWin(rows=6, cols=160)
    mtop.render_models(w, 0, _snap(models=[]))
    assert "No jobs in the history yet" in w.text()
    w = FakeWin(rows=6, cols=160)
    mtop.render_models(w, 0, _snap(models_ok=False, models_err="HTTP 502"))
    assert "API error: HTTP 502" in w.text()


def test_header_names_comfyui():
    w = FakeWin(rows=4, cols=160)
    mtop.render_header(w, 0, {"status": "running", "mode": "local", "backend": "comfyui",
                              "pid": 42, "uptime": "", "server": {"version": "0.37.0"}},
                       stale=False)
    text = w.text()
    assert "ComfyUI Model Monitor" in text
    assert "comfyui: ● pid 42" in text and "comfyui 0.37.0" in text


# ── Prometheus ──────────────────────────────────────────────────────────────

def test_prometheus_queue_loaded_and_last_used():
    snap = {**_snap(), "status": "api-only", "models_ok": True}
    snap["endpoints"][0]["version"] = "0.37.0"
    text = export.prometheus_text(snap, "9.9.9", now=0.0)
    assert 'mtop_comfyui_info{endpoint="l",url="u",version="0.37.0"} 1' in text
    assert 'mtop_comfyui_queue_jobs{endpoint="l",url="u",state="running"} 1' in text
    assert 'mtop_comfyui_queue_jobs{endpoint="l",url="u",state="pending"} 2' in text
    # only the running job's model counts as loaded
    assert 'mtop_models_loaded{endpoint="l",url="u"} 1' in text
    assert 'mtop_comfyui_model_last_used_seconds{endpoint="l",url="u",' \
           'model="qwen_vae.safetensors",kind="vae"}' in text


def test_prometheus_other_backends_carry_no_comfyui_metrics():
    text = export.prometheus_text({"backend": "ollama", "endpoints": []}, "9.9.9", now=0.0)
    assert "mtop_comfyui_queue_jobs" not in text


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs /proc")
def test_find_comfyui_pid_none_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(procfs, "find_comfyui_pids", lambda: [])
    assert procfs.find_comfyui_pid(8188) is None


# ── GPU memory of the server, colored logs ──────────────────────────────────

def test_server_gpu_mib_sums_the_server_tree(monkeypatch):
    monkeypatch.setattr(mtop.collector, "process_tree", lambda pid: [pid, pid + 1])
    gpus = [{"vendor": "nvidia", "index": 0,
             "procs": [{"pid": 3732, "mem_mib": 30000}, {"pid": 3733, "mem_mib": 200},
                       {"pid": 9999, "mem_mib": 5000}]}]
    assert mtop.Collector._server_gpu_mib(3732, gpus) == 30200
    assert mtop.Collector._server_gpu_mib(None, gpus) is None
    assert mtop.Collector._server_gpu_mib(3732, [{"vendor": "amd", "index": 0}]) is None


def test_gpu_memory_on_screen_and_in_prometheus():
    snap = _snap(gpu_mem_mib=30208)
    w = FakeWin(rows=10, cols=200)
    mtop.render_models(w, 0, snap)
    assert "GPU memory held: 29.5 GiB" in w.text()
    text = export.prometheus_text({**snap, "status": "api-only"}, "9.9.9", now=0.0)
    assert f'mtop_comfyui_gpu_memory_bytes{{endpoint="l",url="u"}} {30208 * 1024**2}' in text


def test_strip_ansi():
    assert mtop.logs.strip_ansi("\x1b[32m[INFO]\x1b[0m Prompt executed") == "[INFO] Prompt executed"
    assert mtop.logs.strip_ansi("plain") == "plain"


def test_catalog_gets_its_own_line():
    w = FakeWin(rows=10, cols=200)
    mtop.render_models(w, 0, _snap(gpu_mem_mib=1024))
    assert w.line(1).strip() == "queue: 1 running · 2 pending   GPU memory held: 1.0 GiB"
    assert w.line(2).strip().startswith("catalog: 2 checkpoints")


class _FakeLogs:
    name = "docker-api logs"

    def tail(self, n):
        return True, [(1.0, "\x1b[32m[INFO]\x1b[0m Prompt executed in 3.1 seconds")]


def test_logs_carry_no_request_stats(monkeypatch):
    c = _collector(monkeypatch, ROUTES)
    try:
        c._log_source, c._log_mode = _FakeLogs(), "docker"
        logs = c._read_logs("docker")
        assert logs["ok"] and logs["requests"] is None
        assert logs["lines"][0]["text"] == "[INFO] Prompt executed in 3.1 seconds"
    finally:
        c.close()
    w = FakeWin(rows=4, cols=160)
    mtop.render_logs(w, 0, {"logs": logs})
    assert "requests" not in w.line(0) and "docker-api logs" in w.line(0)
