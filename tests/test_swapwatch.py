"""llama-swap TTL countdown: model activity from /api/events (swapwatch)."""

import http.server
import json
import threading
import time

import pytest

import mtop
from conftest import FakeWin
from mtop.swapwatch import SwapActivity


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def ev(kind, data):
    return {"type": kind, "data": json.dumps(data)}


def log(text, source="proxy"):
    return ev("logData", {"source": source, "data": text})


def status(**states):
    return ev("modelStatus", [{"id": k, "state": v} for k, v in states.items()])


UP_LINE = ('[INFO] Request 10.0.0.2 "POST /upstream/basal/v1/systemone HTTP/1.1" '
           '200 410 "Python-urllib/3.14" 882.6ms\n')


@pytest.fixture
def watch():
    clock = Clock()
    w = SwapActivity(mtop.Endpoint("localhost:8001"), clock=clock)
    w.connected()
    w.feed(log(UP_LINE * 3))                 # proxy backlog: no timestamps, skipped
    w.feed(log("model output", "upstream"))  # upstream backlog
    w.feed(status(basal="ready", qwen="stopped"))
    return w, clock


def test_backlog_is_not_activity(watch):
    w, _ = watch
    assert w.idle_since("basal") is None


def test_upstream_request_line_restarts_the_window(watch):
    w, clock = watch
    clock.t = 1100.0
    w.feed(log(UP_LINE))
    assert w.idle_since("basal") == 1100.0
    w.feed(log(UP_LINE.replace("basal", "qwen"), "upstream"))  # a model's own output
    assert w.idle_since("qwen") is None


def test_inflight_request_holds_the_window_open(watch):
    w, clock = watch
    w.feed(ev("inflight", {"operation": "upsert",
                           "request": {"id": "7", "model": "basal", "req_path": "/v1/chat"}}))
    clock.t = 1500.0
    assert w.idle_since("basal") == 1500.0
    w.feed(ev("inflight", {"operation": "remove", "id": "7"}))
    clock.t = 1600.0
    assert w.idle_since("basal") == 1500.0


def test_becoming_ready_opens_the_window_and_stopping_forgets_it(watch):
    w, clock = watch
    clock.t = 1200.0
    w.feed(status(basal="ready", qwen="starting"))
    w.feed(status(basal="ready", qwen="ready"))
    assert w.idle_since("qwen") == 1200.0
    assert w.idle_since("basal") is None     # was ready before mtop looked
    w.feed(status(basal="ready", qwen="stopped"))
    assert w.idle_since("qwen") is None


def test_upstream_model_id_with_a_slash(watch):
    w, clock = watch
    w.feed(status(**{"org/my model": "ready"}))
    w.feed(log('[INFO] Request ::1 "GET /upstream/org/my%20model/health HTTP/1.1" '
               '200 2 "curl" 1ms\n'))
    assert w.idle_since("org/my model") == clock.t


def _models(monkeypatch, since):
    def get(url, *a, **k):
        if url.endswith("/v1/models"):
            return True, {"data": [{"id": "basal", "status": {"value": "ready"}}]}
        return True, {"running": [{"model": "basal", "state": "ready", "ttl": 3600,
                                   "proxy": "http://127.0.0.1:8102"}]}
    monkeypatch.setattr(mtop.util, "http_get_json", get)
    c = mtop.Collector(container="llama-swap", api_url="http://localhost:8001", interval=1.0,
                       show_gpu=False, mode="api", backend="llama-swap")
    if since is not ...:
        class W:
            def idle_since(self, model):
                return since
        c._activity[c.primary.label] = W()
    try:
        return c._llama_swap_result(c.primary)["models"][0]
    finally:
        c._activity.clear()
        c.close()


def test_collector_counts_the_ttl_down(monkeypatch):
    m = _models(monkeypatch, time.monotonic() - 600)
    assert m["ttl"] == 3600 and m["ttl_left"] == 3000
    assert mtop.export.parse_iso(m["expires_at"]) is not None


def test_collector_without_a_known_last_use_reports_only_the_ttl(monkeypatch):
    for since in (None, ...):
        m = _models(monkeypatch, since)
        assert m["ttl"] == 3600 and "ttl_left" not in m and "expires_at" not in m


def test_ttl_column_counts_down_or_marks_an_upper_bound():
    def row(**m):
        w = FakeWin(rows=8, cols=140)
        mtop.render_models(w, 0, {"backend": "llama-swap", "endpoints": [{
            "label": "local", "models_ok": True,
            "models": [{"name": "basal", "state": "ready", "running": True, **m}]}]})
        return next(w.line(i) for i in range(8) if w.line(i).lstrip().startswith("basal"))
    assert "50m 0s" in row(ttl=3600, ttl_left=3000)
    assert "≤1h 0m" in row(ttl=3600)
    assert "—" in row(ttl=None)


def test_event_stream_end_to_end():
    body = "".join(f"event:message\ndata:{json.dumps(e)}\n\n" for e in (
        log(UP_LINE), status(basal="ready"), log(UP_LINE)))

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            assert self.path == "/api/events"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body.encode())
            self.wfile.flush()
            time.sleep(0.5)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    w = SwapActivity(mtop.Endpoint(f"http://127.0.0.1:{srv.server_port}"))
    w.start()
    try:
        deadline = time.monotonic() + 3
        while w.idle_since("basal") is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert w.idle_since("basal") is not None
    finally:
        w.stop()
        srv.shutdown()
