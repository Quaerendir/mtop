"""--backend llama-swap: GPU memory of vLLM's EngineCore, what counts as
loaded, and backend-specific wording on screen."""

import mtop
from conftest import FakeWin

from mtop import export


# ── GPU join through a runner's descendants ─────────────────────────────────

def _vllm_setup():
    runners = [{"pid": 100, "engine": "vllm", "model_name": "qwen3.8-27b"}]
    gpus = [{"vendor": "nvidia", "index": 0,
             "procs": [{"pid": 3753, "mem_mib": 200}, {"pid": 101, "mem_mib": 72294}]}]
    parents = {101: 100, 100: 50, 50: 1, 3753: 1}
    return runners, gpus, parents.get


def test_engine_core_memory_is_credited_to_the_vllm_runner():
    runners, gpus, parent_of = _vllm_setup()
    mtop.link_runners_to_gpus(runners, gpus, parent_of)
    assert runners[0]["gpu"] == ["nvidia:0"] and runners[0]["gpu_mem_mib"] == 72294
    procs = gpus[0]["procs"]
    assert procs[1]["model"] == "qwen3.8-27b" and "model" not in procs[0]


def test_without_ppid_lookup_only_exact_pids_link():
    runners, gpus, _ = _vllm_setup()
    mtop.link_runners_to_gpus(runners, gpus)
    assert "gpu" not in runners[0]


def test_several_workers_on_one_card_sum_once_per_card():
    runners = [{"pid": 100, "model_name": "m"}]
    gpus = [{"vendor": "nvidia", "index": 0,
             "procs": [{"pid": 101, "mem_mib": 10}, {"pid": 102, "mem_mib": 5}]}]
    mtop.link_runners_to_gpus(runners, gpus, {101: 100, 102: 101}.get)
    assert runners[0]["gpu"] == ["nvidia:0"] and runners[0]["gpu_mem_mib"] == 15


def test_ancestry_walk_is_bounded():
    chain = {p: p - 1 for p in range(101, 120)}         # 119 -> 118 -> ... -> 100
    runners = [{"pid": 100, "model_name": "m"}]
    gpus = [{"vendor": "nvidia", "index": 0, "procs": [{"pid": 119, "mem_mib": 1}]}]
    mtop.link_runners_to_gpus(runners, gpus, chain.get)
    assert "gpu" not in runners[0]


# ── what counts as loaded ───────────────────────────────────────────────────

V1_MODELS = {"data": [
    {"id": "gpt-oss-120b", "description": "big", "status": {"value": "unloaded"}},
    {"id": "qwen3.8-27b", "description": "default", "status": {"value": "ready"}},
]}
RUNNING = {"running": [{"model": "qwen3.8-27b", "state": "ready", "ttl": 86400,
                        "proxy": "http://127.0.0.1:8105"}]}


def _result(monkeypatch, running):
    def get(url, *a, **k):
        if url.endswith("/v1/models"):
            return True, V1_MODELS
        return running
    monkeypatch.setattr(mtop.util, "http_get_json", get)
    c = mtop.Collector(container="llama-swap", api_url="http://localhost:8001", interval=1.0,
                       show_gpu=False, mode="api", backend="llama-swap")
    try:
        return c._llama_swap_result(c.primary)
    finally:
        c.close()


def test_running_flag_comes_from_the_running_endpoint(monkeypatch):
    res = _result(monkeypatch, (True, RUNNING))
    by = {m["name"]: m for m in res["models"]}
    assert by["qwen3.8-27b"]["running"] is True and by["qwen3.8-27b"]["port"] == "8105"
    assert by["gpt-oss-120b"]["running"] is False


def test_running_falls_back_to_catalog_status(monkeypatch):
    res = _result(monkeypatch, (False, "HTTP 404 Not Found"))
    assert {m["name"]: m["running"] for m in res["models"]} == {
        "gpt-oss-120b": False, "qwen3.8-27b": True}


def test_prometheus_counts_only_running_models_and_labels_the_backend():
    snap = {"status": "running", "models_ok": True, "mode": "local", "backend": "llama-swap",
            "endpoints": [{"label": "localhost:8001", "url": "http://localhost:8001",
                           "models_ok": True, "models": [
                               {"name": "gpt-oss-120b", "state": "unloaded", "running": False},
                               {"name": "qwen3.8-27b", "state": "ready", "running": True}]}]}
    text = export.prometheus_text(snap, "9.9.9", now=0.0)
    assert 'mtop_models_loaded{endpoint="localhost:8001",url="http://localhost:8001"} 1' in text
    assert 'backend="llama-swap"' in text.split("mtop_info{", 1)[1].split("\n", 1)[0]


def test_prometheus_ollama_models_have_no_flag_and_all_count():
    snap = {"status": "running", "models_ok": True, "mode": "api",
            "endpoints": [{"label": "l", "url": "u", "models_ok": True,
                           "models": [{"name": "a"}, {"name": "b"}]}]}
    text = export.prometheus_text(snap, "9.9.9", now=0.0)
    assert 'mtop_models_loaded{endpoint="l",url="u"} 2' in text
    assert 'backend="ollama"' in text


# ── wording ─────────────────────────────────────────────────────────────────

def test_header_title_names_the_backend():
    for backend, name in (("llama-swap", "llama-swap Model Monitor"),
                          ("ollama", "Ollama Model Monitor")):
        w = FakeWin(rows=4, cols=140)
        mtop.render_header(w, 0, {"status": "running", "mode": "local", "backend": backend,
                                  "uptime": ""}, stale=False)
        assert name in w.line(0)


def test_server_config_message_for_llama_swap():
    w = FakeWin(rows=6, cols=160)
    mtop.render_server_config(w, 0, {"mode": "local", "backend": "llama-swap",
                                     "server": {"env": {}, "env_source": "process"}})
    text = w.text()
    assert "OLLAMA_" not in text and "llama-swap's environment (process)" in text
