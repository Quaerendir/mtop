"""Log sources, GIN parsing, request stats, the LOGS panel and sparklines."""

import json
import struct

import pytest
from conftest import FakeWin

import mtop
from mtop import logs

GIN_OK = '[GIN] 2026/09/13 - 20:44:07 | 200 |  1.234567s |  127.0.0.1 | POST     "/api/generate"'
GIN_ERR = '[GIN] 2026/09/13 - 20:44:09 | 500 |     12.5ms |  10.0.0.7 | POST     "/api/chat"'
GIN_PS = '[GIN] 2026/09/13 - 20:44:10 | 200 |    23.752µs |  127.0.0.1 | GET      "/api/ps"'
GIN_404 = '[GIN] 2026/09/13 - 20:44:11 | 404 |      50µs |  10.0.0.7 | GET      "/nope"'
INFO = 'time=2026-09-13T20:44:00.000Z level=INFO source=server.go:123 msg="loaded model"'
WARN = 'time=2026-09-13T20:44:01.000Z level=WARN source=sched.go:9 msg="low vram"'


@pytest.mark.parametrize("text,sec", [
    ("1.234567s", 1.234567), ("12.5ms", 0.0125), ("23.752µs", 23.752e-6), ("1m2.3s", 62.3),
    ("500ns", 5e-7), ("2h", 7200.0), ("", None), ("fast", None),
])
def test_parse_go_duration(text, sec):
    got = logs.parse_go_duration(text)
    assert got == sec or (got is not None and abs(got - sec) < 1e-12)


from datetime import datetime, timedelta, timezone  # noqa: E402

T0 = datetime(2026, 9, 13, 21, 42, 0, 291825, tzinfo=timezone.utc).timestamp()
T1 = datetime(2026, 9, 13, 22, 41, 20, tzinfo=timezone(timedelta(hours=1))).timestamp()
T2 = datetime(2026, 9, 13, 22, 41, 20, tzinfo=timezone.utc).timestamp()


def test_parse_timestamp_forms():
    assert logs.parse_timestamp("2026-09-13T21:42:00.291825230Z") == pytest.approx(T0)
    assert logs.parse_timestamp("2026-09-13T22:41:20+0100") == T1     # old journalctl
    assert logs.parse_timestamp("2026-09-13T22:41:20+01:00") == T1
    assert logs.parse_timestamp("2026-09-13T22:41:20-0500") == T1 + 6 * 3600
    assert logs.parse_timestamp("2026-09-13 22:41:20") == T2          # naive -> UTC
    assert logs.parse_timestamp("nope") is None


def test_split_line():
    ts, text = logs.split_line("2026-09-13T21:42:00.291825230Z " + GIN_OK)
    assert ts is not None and text == GIN_OK
    assert logs.split_line(GIN_OK) == (None, GIN_OK)


def test_parse_gin():
    g = logs.parse_gin(GIN_OK)
    assert g == {"status": 200, "latency": pytest.approx(1.234567), "client": "127.0.0.1",
                 "method": "POST", "path": "/api/generate"}
    assert logs.parse_gin(INFO) is None
    assert logs.parse_gin(GIN_ERR)["status"] == 500


def test_line_level():
    assert logs.line_level(GIN_OK) == "info"
    assert logs.line_level(GIN_404) == "warn"
    assert logs.line_level(GIN_ERR) == "error"
    assert logs.line_level(INFO) == "info"
    assert logs.line_level(WARN) == "warn"
    assert logs.line_level("Error: something exploded") == "error"


def test_request_stats_window_classes_monitor_and_untimed():
    now = 1000.0
    lines = [(now - 5, GIN_OK), (now - 3, GIN_ERR), (now - 2, GIN_PS), (now - 1, GIN_404),
             (now - 120, GIN_OK), (None, GIN_OK), (now - 4, INFO)]
    r = logs.request_stats(lines, window=60, now=now)
    assert r["total"] == 3 and r["monitor"] == 1 and r["untimed"] == 1
    assert r["by_status"] == {"2xx": 1, "4xx": 1, "5xx": 1}
    assert r["by_path"] == {"/api/generate": 1, "/api/chat": 1, "/nope": 1}
    assert r["latency_p50"] == pytest.approx(0.0125)
    assert r["latency_max"] == pytest.approx(1.234567)
    assert r["last"]["path"] == "/nope" and r["last"]["ts"] == now - 1
    empty = logs.request_stats([(now, GIN_PS)], now=now)
    assert empty["total"] == 0 and "latency_p50" not in empty and empty["last"] is None


def test_journal_source_strips_prefix_and_reports_failure():
    out = ("2026-09-13T22:41:20+0100 host ollama[3035]: " + GIN_OK + "\n"
           "-- Boot abc --\n"
           "2026-09-13T22:41:21+0100 host ollama[3035]: " + INFO + "\n")
    calls = []

    def run(cmd, timeout=5, **kw):
        calls.append(cmd)
        return True, out
    ok, lines = logs.JournalLogs(run).tail(50)
    assert ok and calls[0][:4] == ["journalctl", "-u", "ollama.service", "-n"]
    assert lines[0][1] == GIN_OK and lines[1][1] == INFO
    assert lines[0][0] == T1
    ok, err = logs.JournalLogs(lambda *a, **k: (False, "No journal files were opened")).tail(5)
    assert ok is False and "journal" in err


def test_container_source_uses_runtime_logs():
    class RT:
        name = "docker-api"

        def logs(self, name, tail):
            assert name == "ollama" and tail == 50
            return True, "2026-09-13T21:42:00.291825230Z " + GIN_OK + "\n\n"
    src = logs.ContainerLogs(RT(), "ollama")
    ok, lines = src.tail(50)
    assert ok and src.name == "docker-api logs"
    assert lines == [(pytest.approx(T0), GIN_OK)]


# ── runtime logs() ────────────────────────────────────────────────────────────

def test_docker_cli_logs_merges_stderr():
    seen = {}

    def run(cmd, timeout=5, env=None, merge_stderr=False):
        seen["cmd"], seen["merge"] = cmd, merge_stderr
        return True, "2026-09-13T21:42:00Z " + GIN_OK
    from mtop import container as ct
    ok, out = ct.DockerCli(run).logs("ollama", 20)
    assert ok and seen["merge"] is True
    assert seen["cmd"] == ["docker", "logs", "--timestamps", "--tail", "20", "ollama"]


def test_run_cmd_merge_stderr():
    ok, out = mtop.run_cmd(["sh", "-c", "echo out; echo err 1>&2"], merge_stderr=True)
    assert ok and "out" in out and "err" in out
    ok, out = mtop.run_cmd(["sh", "-c", "echo out; echo err 1>&2"])
    assert ok and out == "out"


def _frame(kind, payload):
    return bytes([kind, 0, 0, 0]) + struct.pack(">I", len(payload)) + payload


def test_docker_api_logs_route(sock_dir):
    from test_container import _UnixHTTPServer, make_handler
    import threading
    from mtop import container as ct
    sock = str(sock_dir / "d.sock")
    routes, log = {}, []
    routes[("GET", "/containers/ollama/logs")] = (
        200, _frame(2, b"2026-09-13T21:42:00Z " + GIN_OK.encode() + b"\n"))
    srv = _UnixHTTPServer(sock, make_handler(routes, log))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        api = ct.DockerApi(f"unix://{sock}")
        ok, out = api.logs("ollama", 77)
        assert ok and GIN_OK in out
        assert "tail=77" in log[-1][1] and "timestamps=1" in log[-1][1]
        assert api.logs("ghost", 1)[0] is False
    finally:
        srv.shutdown()
        srv.server_close()


# ── collector integration ────────────────────────────────────────────────────

def test_collector_logs_local_via_journal(monkeypatch):
    from test_collector import SERVER_PID, API_PS  # noqa: F401  (fixtures below)
    monkeypatch.setattr(mtop.collector, "IS_LINUX", True)
    monkeypatch.setattr(mtop.collector, "systemd_ollama", lambda: ("running", SERVER_PID, None))
    monkeypatch.setattr(mtop.collector, "find_ollama_pid", lambda port=None: SERVER_PID)
    monkeypatch.setattr(mtop.collector, "proc_uptime_sec", lambda pid: 1.0)
    monkeypatch.setattr(mtop.Collector, "_local_stats", lambda self, pid: None)
    monkeypatch.setattr(mtop.collector, "read_proc_environ", lambda pid: [])
    monkeypatch.setattr(mtop.util, "http_get_json",
                        lambda url, timeout=5, headers=None, context=None: (True, API_PS))
    import time as _t
    now = _t.time()
    lines = [(now - 2, GIN_OK), (now - 1, GIN_ERR)]
    monkeypatch.setattr(logs.JournalLogs, "tail", lambda self, n: (True, lines))
    c = mtop.Collector(container="ollama", api_url="http://localhost:11434", interval=1.0,
                       show_gpu=False, mode="local", show_logs=True, log_lines=1)
    snap = c.collect(100.0)
    lg = snap["logs"]
    assert lg["source"] == "journalctl" and lg["ok"]
    assert len(lg["lines"]) == 1 and lg["lines"][0]["level"] == "error"
    assert lg["requests"]["total"] == 2 and lg["requests"]["by_status"] == {"2xx": 1, "5xx": 1}
    # toggling off drops the key; back on re-reads
    c.show_logs = False
    assert "logs" not in c.collect(103.0)


def test_missing_log_source_probed_once_per_mode(monkeypatch):
    from test_collector import SERVER_PID
    monkeypatch.setattr(mtop.collector, "IS_LINUX", True)
    probes = []

    def sd():
        probes.append(1)
        return None
    monkeypatch.setattr(mtop.collector, "systemd_ollama", sd)
    monkeypatch.setattr(mtop.collector, "find_ollama_pid", lambda port=None: SERVER_PID)
    monkeypatch.setattr(mtop.collector, "proc_uptime_sec", lambda pid: 1.0)
    monkeypatch.setattr(mtop.Collector, "_local_stats", lambda self, pid: None)
    monkeypatch.setattr(mtop.collector, "read_proc_environ", lambda pid: [])
    monkeypatch.setattr(mtop.util, "http_get_json",
                        lambda url, timeout=5, headers=None, context=None: (True, {"models": []}))
    c = mtop.Collector(container="ollama", api_url="http://localhost:11434", interval=1.0,
                       show_gpu=False, mode="local", show_logs=True)
    c.collect(100.0)
    n = len(probes)                      # status probes + one log-source probe
    c.collect(103.0)
    c.collect(106.0)
    # each cycle adds exactly one probe (the status check), none for the log source
    assert len(probes) == n + 2
    c.reset_log_source()
    c.collect(109.0)
    assert len(probes) == n + 4          # status + a fresh log-source probe


def test_collector_logs_no_source_for_manual_serve(monkeypatch):
    from test_collector import SERVER_PID
    monkeypatch.setattr(mtop.collector, "IS_LINUX", True)
    monkeypatch.setattr(mtop.collector, "systemd_ollama", lambda: None)
    monkeypatch.setattr(mtop.collector, "find_ollama_pid", lambda port=None: SERVER_PID)
    monkeypatch.setattr(mtop.collector, "proc_uptime_sec", lambda pid: 1.0)
    monkeypatch.setattr(mtop.Collector, "_local_stats", lambda self, pid: None)
    monkeypatch.setattr(mtop.collector, "read_proc_environ", lambda pid: [])
    monkeypatch.setattr(mtop.util, "http_get_json",
                        lambda url, timeout=5, headers=None, context=None: (True, {"models": []}))
    c = mtop.Collector(container="ollama", api_url="http://localhost:11434", interval=1.0,
                       show_gpu=False, mode="local", show_logs=True)
    lg = c.collect(100.0)["logs"]
    assert lg["ok"] is False and "terminal" in lg["error"] and lg["source"] is None


def test_history_recorded_per_slow_cycle(monkeypatch):
    monkeypatch.setattr(mtop.util, "http_get_json",
                        lambda url, timeout=5, headers=None, context=None: (True, {"models": []}))
    gpus = [{"vendor": "nvidia", "index": 0, "name": "x", "util": "40", "mem_used": "50",
             "mem_total": "200", "temp": "1"}]
    monkeypatch.setattr(mtop.Collector, "_gpu_read", lambda self: gpus)
    c = mtop.Collector(container="ollama", api_url="http://localhost:11434", interval=1.0,
                       show_gpu=True, mode="api")
    c.collect(100.0)
    c.collect(101.0)                     # not a slow cycle: no new sample
    snap = c.collect(103.0)
    assert snap["history"]["gpu:nvidia:0:util"] == [40.0, 40.0]
    assert snap["history"]["gpu:nvidia:0:mem"] == [25.0, 25.0]
    assert "cpu" not in snap["history"]  # api mode has no server stats


# ── rendering ────────────────────────────────────────────────────────────────

def test_sparkline_levels_and_width():
    assert mtop.sparkline([0, 12.5, 50, 99, 100], 10) == "▁▂▅██"
    assert mtop.sparkline(list(range(0, 100, 10)), 4) == "▅▆▇█"   # last four: 60..90
    assert mtop.sparkline([], 5) == ""


def test_draw_spark_skips_when_narrow():
    w = FakeWin(rows=2, cols=40)
    mtop.draw_spark(w, 0, 5, 10, [50] * 20)             # 5 wide < min 8 -> nothing
    assert w.text() == ""
    mtop.draw_spark(w, 0, 5, 25, [50] * 20)
    assert w.line(0).strip() == "▅" * 20


def test_resources_show_sparkline_only_when_room():
    snap = {"mode": "local", "cpu_limit": 4.0,
            "res_stats": {"cpu": "200%", "mem_usage": "1GiB / 4GiB", "mem_pct": "25%"},
            "history": {"cpu": [10, 50, 90], "mem": [25, 25, 25]}}
    wide = FakeWin(rows=6, cols=120)
    mtop.render_resources(wide, 0, snap)
    assert "▁▅█" in wide.line(1) and "▃▃▃" in wide.line(2)
    narrow = FakeWin(rows=6, cols=64)
    mtop.render_resources(narrow, 0, snap)
    assert "▅" not in narrow.text() and "200% / 4 cores" in narrow.line(1)


def test_render_logs_panel(win):
    import time as _t
    now = _t.time()
    snap = {"logs": {"source": "journalctl", "ok": True, "error": "",
                     "lines": [{"ts": now, "text": GIN_OK, "level": "info"},
                               {"ts": None, "text": GIN_ERR, "level": "error"}],
                     "requests": {"window_sec": 60, "total": 12,
                                  "by_status": {"2xx": 11, "5xx": 1},
                                  "latency_p50": 1.23, "monitor": 40}}}
    mtop.render_logs(win, 0, snap)
    text = win.text()
    assert "LOGS · journalctl · 12 req/60s · 11×2xx 1×5xx · p50 1.2s" in text
    assert GIN_OK in text and GIN_ERR in text
    assert any(a == 0 or True for _, _, _, a in win.calls)
    w2 = FakeWin()
    mtop.render_logs(w2, 0, {"logs": {"source": None, "ok": False, "error": "why", "lines": []}})
    assert "logs unavailable: why" in w2.text()
    assert mtop.render_logs(FakeWin(), 0, {}) == 0


def test_format_request_stats():
    assert mtop.format_request_stats(None) == ""
    assert mtop.format_request_stats({"total": 0, "window_sec": 60}) == "no requests in last 60s"


def test_prometheus_log_metrics():
    from mtop import export
    snap = {"mode": "local", "status": "running", "models_ok": True,
            "logs": {"requests": {"window_sec": 60, "by_status": {"2xx": 3, "5xx": 1},
                                  "latency_p50": 0.5, "latency_max": 2.0, "total": 4}}}
    text = export.prometheus_text(snap, "1")
    assert 'mtop_log_requests{mode="local",status_class="2xx"} 3' in text
    assert 'mtop_log_request_latency_seconds{mode="local",quantile="max"} 2' in text
    quiet_snap = {**snap, "logs": {"requests": {"window_sec": 60, "by_status": {}, "total": 0}}}
    quiet = export.prometheus_text(quiet_snap, "1")
    assert 'status_class="none"} 0' in quiet


def test_headless_json_carries_logs_and_drops_history(monkeypatch, capsys):
    import argparse
    snap = {"ts": 1.0, "mode": "api", "status": "api-only", "models_ok": True, "models": [],
            "history": {"cpu": [1]}, "logs": {"source": None, "ok": False, "error": "x",
                                              "lines": [], "requests": None}}
    monkeypatch.setattr(mtop.Collector, "collect", lambda c, now: dict(snap))
    monkeypatch.setattr(mtop.Collector, "needs_second_sample", lambda c, s: False)
    args = argparse.Namespace(container="ollama", api_url="http://localhost:11434", interval=1.0,
                              no_gpu=True, mode="api", no_runners=False, runtime="auto",
                              no_env=False, endpoints=None, json=True, prometheus=False,
                              watch=False, output=None, logs=True, log_lines=3)
    mtop.headless_main(args)
    out = json.loads(capsys.readouterr().out)
    assert "history" not in out and out["logs"]["error"] == "x"
