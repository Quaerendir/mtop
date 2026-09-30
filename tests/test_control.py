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
    assert c.notice(time.monotonic())[0] == "Stopped y on b"
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
    assert "s: stop │ L: load model" in w.line(2)
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, False, control=True, notice=("Stop a? [y/N]", 0))
    assert w.line(2).strip() == "Stop a? [y/N]"
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, False)
    assert "stop" not in w.line(2)


# ── load picker ──────────────────────────────────────────────────────────────

TAGS = {"models": [
    {"name": "qwen3.6:35b", "size": 24 * 2**30,
     "details": {"parameter_size": "35B", "quantization_level": "Q4_K_M"}},
    {"name": "bielik:11b", "size": 12 * 2**30,
     "details": {"parameter_size": "11B", "quantization_level": "Q8_0"}},
    {"name": "coder:32b", "size": 19 * 2**30, "details": {}},
]}


def _wait_items(c):
    deadline = time.monotonic() + 2
    while c.picker["items"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert c.picker["items"] is not None


@pytest.fixture
def api(monkeypatch):
    """GET /api/tags per base URL, POSTs recorded."""
    posts = []
    tags = {"http://localhost:1": TAGS, "http://localhost:2": {"models": [{"name": "far:7b"}]}}

    def get(url, *a, **k):
        base = url.rsplit("/api/", 1)[0]
        return (True, tags[base]) if base in tags else (False, "connection refused")

    monkeypatch.setattr(mtop.util, "http_get_json", get)
    def post(url, body, timeout, *a):
        posts.append((url, body, timeout))
        return True, {}

    monkeypatch.setattr(mtop.util, "http_post_json", post)
    return posts


def test_picker_lists_tags_sorted_and_loads_the_chosen_one(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.open_picker()
    _wait_items(c)
    assert [m["name"] for m in c.picker["items"]] == ["bielik:11b", "coder:32b", "qwen3.6:35b"]
    c.picker_key(curses.KEY_DOWN)
    c.picker_key(curses.KEY_DOWN)
    c.picker_key(curses.KEY_DOWN)                     # clamped at the end
    c.picker_key(10)                                  # Enter
    assert c.picker is None
    _wait_idle(c)
    url, body, timeout = api[-1]
    assert url == "http://localhost:1/api/generate" and body == {"model": "qwen3.6:35b"}
    assert timeout == mtop.ui.LOAD_TIMEOUT
    assert c.notice(time.monotonic())[0] == "Loaded qwen3.6:35b"


def test_picker_escape_closes_without_loading(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.open_picker()
    _wait_items(c)
    c.picker_key(27)
    assert c.picker is None and not api


def test_picker_opens_on_the_selected_endpoint_and_tab_cycles(api):
    c = mtop.ModelControl([mtop.Endpoint("a=localhost:1"), mtop.Endpoint("b=localhost:2")])
    c.sync(_snap(("a", []), ("b", ["y"])))
    c.open_picker()
    assert c.picker["ep"] == 1
    _wait_items(c)
    assert [m["name"] for m in c.picker["items"]] == ["far:7b"]
    c.picker_key(9)                                   # Tab
    assert c.picker["ep"] == 0
    _wait_items(c)
    c.picker_key(10)
    _wait_idle(c)
    assert api[-1][0] == "http://localhost:1/api/generate"
    assert c.notice(time.monotonic())[0] == "Loaded bielik:11b on a"


def test_picker_shows_fetch_errors(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:9")])
    c.open_picker()
    _wait_items(c)
    w = FakeWin(rows=20, cols=100)
    mtop.render_picker(w, c, {"endpoints": [{"models_ok": False, "models": []}]})
    assert "/api/tags failed: connection refused" in w.text()
    c.picker_key(10)                                  # nothing to load
    assert not api and c.busy is None


def test_picker_renders_details_marks_loaded_and_scrolls(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.open_picker()
    _wait_items(c)
    snap = _snap(("local", ["coder:32b"]))
    w = FakeWin(rows=11, cols=100)                    # room for all 3 rows
    mtop.render_picker(w, c, snap)
    text = w.text()
    assert "LOAD MODEL" in text and "1/3" in text
    assert "35B" in text and "Q4_K_M" in text and "24.00 G" in text
    line = next(ln for ln in text.splitlines() if "coder:32b" in ln)
    assert "●" in line
    w = FakeWin(rows=9, cols=100)                     # 2 body rows: must scroll
    c.picker_key(curses.KEY_END)
    mtop.render_picker(w, c, snap)
    assert "qwen3.6:35b" in w.text() and "bielik" not in w.text()


def test_one_action_at_a_time(api, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(mtop.util, "http_post_json",
                        lambda *a, **k: (gate.wait(2), (True, {}))[1])
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.sync(_snap(("local", ["a"])))
    c.request_stop()
    c.answer(True)
    assert c.busy and c.busy[0] == "Stopping"
    assert c.notice(time.monotonic())[0].startswith("Stopping a… ")
    c.open_picker()
    assert c.picker is None                           # refused while busy
    c.request_stop()
    assert c.confirm is None
    gate.set()
    _wait_idle(c)
