"""--control: the POST helper, the model cursor and the stop (unload) action."""

import curses
import http.server
import json
import threading
import time

import pytest

import mtop
from conftest import FakeWin


def _model(name):
    return {"name": name, "size": 4, "size_vram": 4, "context_length": 1}


def _snap(*endpoints):
    return {"endpoints": [{"label": label, "models_ok": True,
                           "models": [_model(n) for n in names]}
                          for label, names in endpoints]}


@pytest.fixture
def post_server():
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, dict(self.headers), json.loads(body)))
            out = json.dumps({"done": True, "done_reason": "unload"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    finally:
        srv.shutdown()
        srv.server_close()


def test_endpoint_post_json_sends_json_body_and_headers(post_server):
    base, seen = post_server
    ep = mtop.Endpoint(base, {"Authorization": "Bearer sekret"})
    ok, data = ep.post_json("/api/generate", {"model": "a:7b", "keep_alive": 0})
    assert ok and data["done_reason"] == "unload"
    path, headers, body = seen[-1]
    assert path == "/api/generate" and body == {"model": "a:7b", "keep_alive": 0}
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer sekret"


def test_cursor_follows_the_model_not_the_row():
    c = mtop.ModelControl([mtop.Endpoint("localhost:11434")])
    c.sync(_snap(("local", ["a", "b", "c"])))
    assert c.selected == (0, "a")
    c.move(1, _snap(("local", ["a", "b", "c"])))
    assert c.selected == (0, "b")
    c.sync(_snap(("local", ["c", "a", "b"])))          # /api/ps reordered
    assert c.selected == (0, "b")
    c.sync(_snap(("local", ["c", "a"])))               # b unloaded: take its position
    assert c.selected == (0, "a")
    c.move(5, _snap(("local", ["c", "a"])))
    assert c.selected == (0, "a")                      # clamped at the end
    c.sync(_snap(("local", [])))
    assert c.selected is None


def test_cursor_spans_endpoints_and_skips_failed_ones():
    eps = [mtop.Endpoint("a=localhost:1"), mtop.Endpoint("b=localhost:2")]
    c = mtop.ModelControl(eps)
    snap = _snap(("a", ["x"]), ("b", ["y"]))
    snap["endpoints"].insert(1, {"label": "dead", "models_ok": False, "models": []})
    c.sync(snap)
    c.move(1, snap)
    assert c.selected == (2, "y")


def _wait_idle(c):
    deadline = time.monotonic() + 2
    while c.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert c.busy is None


def test_stop_asks_first_and_posts_keep_alive_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(mtop.util, "http_post_json",
                        lambda url, body, *a, **k: (calls.append((url, body)), (True, {}))[1])
    eps = [mtop.Endpoint("a=localhost:1"), mtop.Endpoint("b=localhost:2")]
    c = mtop.ModelControl(eps)
    snap = _snap(("a", ["x"]), ("b", ["y"]))
    c.sync(snap)
    c.move(1, snap)

    c.request_stop()
    text, _ = c.notice(time.monotonic())
    assert text == "Stop y on b? [y/N]" and not calls

    c.answer(False)
    assert c.confirm is None and not calls

    c.request_stop()
    c.answer(True)
    _wait_idle(c)
    assert calls == [("http://localhost:2/api/generate", {"model": "y", "keep_alive": 0})]
    assert c.notice(time.monotonic())[0] == "Stopped y"
    assert c.notice(time.monotonic() + mtop.ui.NOTICE_HOLD + 1) is None


def test_stop_failure_is_reported(monkeypatch):
    monkeypatch.setattr(mtop.util, "http_post_json",
                        lambda *a, **k: (False, "HTTP 401 Unauthorized"))
    c = mtop.ModelControl([mtop.Endpoint("localhost:11434")])
    c.sync(_snap(("local", ["a"])))
    c.request_stop()
    c.answer(True)
    _wait_idle(c)
    assert c.notice(time.monotonic())[0] == "Stop a failed: HTTP 401 Unauthorized"


def test_confirm_is_dropped_when_the_model_goes_away():
    c = mtop.ModelControl([mtop.Endpoint("localhost:11434")])
    c.sync(_snap(("local", ["a", "b"])))
    c.request_stop()
    c.sync(_snap(("local", ["b"])))
    assert c.confirm is None


def test_selected_row_is_drawn_reversed(win):
    snap = _snap(("a", ["x"]), ("b", ["y", "z"]))
    mtop.render_models(win, 0, snap, selected=(1, "z"))
    reversed_rows = [t for _, _, t, attr in win.calls if attr & curses.A_REVERSE]
    assert len(reversed_rows) == 1 and reversed_rows[0].startswith("z")


def test_footer_shows_control_keys_or_the_notice():
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, False, control=True)
    assert "s: stop model" in w.line(2)
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, False, control=True, notice=("Stop a? [y/N]", 0))
    assert w.line(2).strip() == "Stop a? [y/N]"
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, False)
    assert "stop" not in w.line(2)
