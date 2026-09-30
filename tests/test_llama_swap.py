"""--backend llama-swap: GPU memory of vLLM's EngineCore, what counts as
loaded, backend-specific wording, the log file, and llama-server runners."""

import os

import pytest

import mtop
from conftest import FakeWin

from mtop import export, logs, procfs


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


# ── llama-swap's access log ──────────────────────────────────────────────────

SWAP_LINE = ('[INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200 1234 '
             '"curl/8.5.0" 1.352148926s')


def test_llama_swap_request_line_parses():
    g = logs.parse_gin(SWAP_LINE)
    assert g == {"status": 200, "latency": 1.352148926, "client": "127.0.0.1",
                 "method": "POST", "path": "/v1/chat/completions"}
    assert logs.parse_gin(SWAP_LINE.replace(" 200 ", " 502 "))["status"] == 502
    assert logs.line_level(SWAP_LINE.replace(" 200 ", " 502 ")) == "error"
    micro = '[INFO] Request ::1 "GET /running HTTP/1.1" 200 15 "curl/8.5.0" 36.4µs'
    assert logs.parse_gin(micro)["latency"] == 36.4e-6


def test_llama_swap_polling_counts_as_monitor_traffic():
    lines = [(100.0, SWAP_LINE),
             (100.0, '[INFO] Request 127.0.0.1 "GET /running HTTP/1.1" 200 15 "x" 20µs'),
             (100.0, '[INFO] Request 127.0.0.1 "GET /v1/models?x=1 HTTP/1.1" 200 9 "x" 9µs')]
    st = logs.request_stats(lines, now=110.0)
    assert st["total"] == 1 and st["monitor"] == 2
    assert st["by_path"] == {"/v1/chat/completions": 1}


# ── following a log file ─────────────────────────────────────────────────────

def test_file_logs_start_at_the_end_untimed_then_stamp_new_lines(tmp_path, monkeypatch):
    f = tmp_path / "llama-swap.log"
    f.write_text("old 1\nold 2\n")
    src = logs.FileLogs(str(f))
    ok, lines = src.tail(10)
    assert ok and lines == [(None, "old 1"), (None, "old 2")]

    monkeypatch.setattr(logs.time, "time", lambda: 1234.5)
    with open(f, "a") as fh:
        fh.write(SWAP_LINE + "\nhalf a li")
    ok, lines = src.tail(10)
    assert lines[-1] == (1234.5, SWAP_LINE)            # the partial line waits
    with open(f, "a") as fh:
        fh.write("ne\n")
    ok, lines = src.tail(10)
    assert lines[-1] == (1234.5, "half a line")
    assert src.tail(2)[1] == lines[-2:]


def test_file_logs_initial_read_is_bounded_and_drops_the_cut_line(tmp_path, monkeypatch):
    monkeypatch.setattr(logs.FileLogs, "INITIAL_BYTES", 20)
    f = tmp_path / "big.log"
    f.write_text("".join(f"line {i:04d}\n" for i in range(1000)))
    ok, lines = logs.FileLogs(str(f)).tail(10)
    assert ok and [t for _, t in lines] == ["line 0998", "line 0999"]


def test_file_logs_follow_truncation_and_report_missing_files(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("a\nb\nc\n")
    src = logs.FileLogs(str(f))
    src.tail(10)
    f.write_text("new\n")                              # truncated and rewritten
    ok, lines = src.tail(10)
    assert ok and lines == [(None, "new")]
    os.remove(f)
    ok, err = src.tail(10)
    assert not ok and str(f) in err


# ── where a unit's output goes ───────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("[Service]\nExecStart=/x\nStandardOutput=append:/home/u/llama-swap.log\n",
     "/home/u/llama-swap.log"),
    ("StandardOutput=file:/var/log/o.log\n", "/var/log/o.log"),
    ("StandardOutput=append:/a.log\n# drop-in\nStandardOutput=journal\n", None),
    ("StandardOutput=journal\n# drop-in\nStandardOutput=truncate:/b.log\n", "/b.log"),
    ("[Service]\nExecStart=/x\n", None),
])
def test_parse_unit_output_file(text, expected):
    assert procfs.parse_unit_output_file(text) == expected


def test_log_source_prefers_the_units_output_file(monkeypatch):
    col = mtop.collector
    monkeypatch.setattr(col, "IS_LINUX", True)
    monkeypatch.setattr(col, "systemd_llama_swap", lambda: ("active", 1, None))
    monkeypatch.setattr(col, "systemd_output_file",
                        lambda unit, user=False: "/home/u/llama-swap.log" if user else None)
    c = mtop.Collector(container="llama-swap", api_url="http://localhost:8001", interval=1.0,
                       show_gpu=False, mode="local", backend="llama-swap")
    try:
        src = c._build_log_source("local")
        assert isinstance(src, logs.FileLogs) and src.path == "/home/u/llama-swap.log"
        monkeypatch.setattr(col, "systemd_output_file", lambda unit, user=False: None)
        src = c._build_log_source("local")
        assert isinstance(src, logs.JournalLogs) and src.user and src.unit == "llama-swap.service"
    finally:
        c.close()


# ── llama-server runners under llama-swap ────────────────────────────────────

SWAP_LLAMA_SERVER = ["/home/u/llama.cpp/build/bin/llama-server", "--model",
                     "/home/u/models/Qwen3.8-27B-Q4_K_M.gguf", "--host", "0.0.0.0",
                     "--port", "8123", "-ngl", "999", "--ctx-size", "262144"]
SWAP_VLLM = ["/v/bin/python", "/v/bin/vllm", "serve", "Qwen/Qwen3.8-27B", "--port", "8105",
             "--served-model-name", "qwen3.8-27b"]


def test_llama_swap_backend_parses_both_engines():
    c = mtop.Collector(container="llama-swap", api_url="http://localhost:8001", interval=1.0,
                       show_gpu=False, mode="api", backend="llama-swap")
    try:
        assert c._parse_argv(SWAP_VLLM)["engine"] == "vllm"
        r = c._parse_argv(SWAP_LLAMA_SERVER)
        assert r["engine"] == "llama" and r["ctx"] == "262144" and r["port"] == "8123"
        assert c._parse_argv(["/usr/bin/python3", "-m", "http.server"]) is None
    finally:
        c.close()


def test_llama_server_joins_the_catalog_by_port():
    models = [{"name": "qwen3.8-27b", "port": "8105", "state": "ready", "ttl": 86400},
              {"name": "qwen3.8-fable-heretic-q4", "port": "8123", "state": "starting",
               "ttl": 86400, "description": "gguf"},
              {"name": "gpt-oss-120b", "port": None, "state": "unloaded"}]
    runners = [mtop.parse_vllm_argv(SWAP_VLLM), mtop.parse_runner_argv(SWAP_LLAMA_SERVER)]
    mtop.runner.match_vllm_runners_to_models(runners, models)
    assert runners[0]["model_name"] == "qwen3.8-27b"
    assert runners[1]["model_name"] == "qwen3.8-fable-heretic-q4"
    assert runners[1]["state"] == "starting"


def test_unmatched_llama_server_shows_its_gguf_and_state_shows_for_any_engine():
    r = mtop.parse_runner_argv(SWAP_LLAMA_SERVER)
    w = FakeWin(rows=6, cols=200)
    mtop.render_runners(w, 0, {"runners": [r]})
    assert "Qwen3.8-27B-Q4_K_M.gguf" in w.text()
    r.update(model_name="qwen3.8-fable-heretic-q4", state="ready")
    w = FakeWin(rows=6, cols=200)
    mtop.render_runners(w, 0, {"runners": [r]})
    assert "qwen3.8-fable-heretic-q4" in w.text() and "state:ready" in w.text()
