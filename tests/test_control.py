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
    assert c.notice(time.monotonic() + mtop.control.NOTICE_HOLD + 1) is None


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
    assert "↑↓ select  s stop  t keep  L load  P pull" in w.line(2)
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, False, control=True, notice=("Stop a? [y/N]", 0))
    assert w.line(2).strip() == "Stop a? [y/N]"
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, False)
    assert "stop" not in w.line(2)


def _footer(cols, **kw):
    w = FakeWin(rows=3, cols=cols)
    mtop.render_footer(w, 2.0, False, True, control=True, **kw)
    return w, w.line(2)


def test_footer_fits_136_columns_with_every_group():
    _, line = _footer(136)
    assert line.startswith(" q quit  +- 2.0s │ ↑↓ select")
    assert "l logs │ ? help" in line and len(line) < 136
    assert "mtop v" not in line and "via " not in line     # both live in the header


def test_footer_drops_whole_groups_when_narrow_but_keeps_help():
    _, line = _footer(80)
    assert "P pull │ ? help" in line and "runners" not in line
    _, line = _footer(40)
    assert line.strip() == "q quit  +- 2.0s │ ? help"
    _, line = _footer(12)                    # cut, never raises
    assert line.startswith(" q quit")


def test_footer_marks_toggles_that_are_on():
    w = FakeWin(rows=3, cols=160)
    mtop.render_footer(w, 1.0, False, True, runners=True, env=False, logs=True)
    bold = {t for _, _, t, a in w.calls if a & curses.A_BOLD}
    assert {"runners", "logs"} <= bold and not {"ps", "env"} & bold


def test_help_lists_every_key_and_the_toggle_state():
    rows = mtop.help_lines(2.0, False, True, True, False, True, True, "docker-api")
    keys = {k for k, _ in rows if k}
    assert {"q  Esc", "+  -", "?", "o", "r", "e", "l", "s", "t", "L", "P", "X"} <= keys
    text = {k: t for k, t in rows if k}
    assert text["e"].endswith("[off]") and text["l"].endswith("[on]")
    assert rows[-1][1].endswith("· via docker-api")
    swap = {k for k, _ in mtop.help_lines(2.0, False, False, True, True, False,
                                          "llama-swap", None) if k}
    assert "L" in swap and not {"t", "P", "X", "o"} & swap
    w = FakeWin(rows=30, cols=100)
    mtop.render_help(w, rows)
    assert "─ KEYS ─" in w.line(3) and "pull a model by name" in w.text()


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
    assert timeout == mtop.control.LOAD_TIMEOUT
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


# ── eviction warning ─────────────────────────────────────────────────────────

def _full_snap(env, loaded=("bielik:11b", "coder:32b"), gpus=1, source="container"):
    snap = _snap(("local", list(loaded)))
    snap["server"] = {"env": env, "env_source": source}
    snap["gpus"] = [{}] * gpus
    return snap


@pytest.mark.parametrize("env,gpus,source,expected", [
    ({"OLLAMA_MAX_LOADED_MODELS": "2"}, 1, "container", (2, "OLLAMA_MAX_LOADED_MODELS")),
    ({}, 2, "process", (6, "Ollama default")),
    ({"OLLAMA_MAX_LOADED_MODELS": "0"}, 0, "systemd", (3, "Ollama default")),
    ({"OLLAMA_MAX_LOADED_MODELS": "x"}, 1, "container", (3, "Ollama default")),
    ({"OLLAMA_MAX_LOADED_MODELS": "2"}, 1, None, None),        # env not readable
])
def test_max_loaded_models(env, gpus, source, expected):
    assert mtop.max_loaded_models(_full_snap(env, gpus=gpus, source=source)) == expected


def _pick(c, name):
    c.open_picker()
    _wait_items(c)
    c.picker["index"] = [m["name"] for m in c.picker["items"]].index(name)
    c.picker_key(10)


def test_load_at_the_limit_asks_and_names_the_candidates(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.sync(_full_snap({"OLLAMA_MAX_LOADED_MODELS": "2"}))
    c.open_picker()
    assert "2/2 loaded: a new model evicts one" in c.notice(time.monotonic())[0]
    c.picker = None

    _pick(c, "qwen3.6:35b")
    assert not api and c.asking
    assert c.notice(time.monotonic())[0] == (
        "Load qwen3.6:35b? [y/N] — 2/2 loaded (OLLAMA_MAX_LOADED_MODELS), "
        "Ollama will unload one of: bielik:11b, coder:32b")
    c.answer(False)
    assert not api and not c.asking

    _pick(c, "qwen3.6:35b")
    c.answer(True)
    _wait_idle(c)
    assert api[-1][1] == {"model": "qwen3.6:35b"}


def test_no_question_below_the_limit_or_for_a_resident_model(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.sync(_full_snap({"OLLAMA_MAX_LOADED_MODELS": "3"}))
    _pick(c, "qwen3.6:35b")
    assert not c.asking
    _wait_idle(c)
    c.sync(_full_snap({"OLLAMA_MAX_LOADED_MODELS": "2"}))
    _pick(c, "coder:32b")                             # already loaded: nothing evicted
    assert not c.asking
    _wait_idle(c)
    assert [b["model"] for _, b, _ in api] == ["qwen3.6:35b", "coder:32b"]


def test_no_question_when_the_limit_is_unknown(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.sync(_full_snap({"OLLAMA_MAX_LOADED_MODELS": "2"}, source=None))
    _pick(c, "qwen3.6:35b")
    assert not c.asking
    _wait_idle(c)


def test_control_is_accepted_with_the_llama_swap_backend(monkeypatch):
    seen = {}
    monkeypatch.setattr("sys.argv", ["mtop", "--control", "--backend", "llama-swap", "--json"])
    monkeypatch.setattr(mtop.cli, "headless_main",
                        lambda args: seen.update(control=args.control, backend=args.backend) or 0)
    with pytest.raises(SystemExit) as e:
        mtop.main()
    assert e.value.code == 0 and seen == {"control": True, "backend": "llama-swap"}


# ── keep loaded (TTL) ────────────────────────────────────────────────────────

def _ttl_snap(ctx=8192):
    snap = _snap(("a", ["x"]), ("b", ["y"]))
    snap["endpoints"][1]["models"][0]["context_length"] = ctx
    return snap


@pytest.mark.parametrize("key,value,text", [
    ("1", "30m", "Keeping y on b loaded for 30m"),
    ("3", "24h", "Keeping y on b loaded for 24h"),
    ("4", -1, "Keeping y on b loaded until unloaded"),
])
def test_keep_loaded_sends_keep_alive_with_the_current_context(api, key, value, text):
    c = mtop.ModelControl([mtop.Endpoint("a=localhost:1"), mtop.Endpoint("b=localhost:2")])
    snap = _ttl_snap()
    c.sync(snap)
    c.move(1, snap)
    c.request_ttl()
    assert c.notice(time.monotonic())[0].startswith("Keep y on b loaded for — 1: 30m │ 2: 2h")
    c.ttl_key(ord(key))
    assert c.ttl_target is None
    _wait_idle(c)
    url, body, timeout = api[-1]
    assert url == "http://localhost:2/api/generate"
    # Without num_ctx Ollama reloads the model with its default context.
    assert body == {"model": "y", "keep_alive": value, "options": {"num_ctx": 8192}}
    assert c.notice(time.monotonic())[0] == text


def test_keep_loaded_without_a_known_context_sends_no_options(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    snap = _snap(("local", ["x"]))
    del snap["endpoints"][0]["models"][0]["context_length"]   # older Ollama
    c.sync(snap)
    c.request_ttl()
    c.ttl_key(ord("2"))
    _wait_idle(c)
    assert api[-1][1] == {"model": "x", "keep_alive": "2h"}


def test_keep_loaded_other_keys_cancel_and_vanished_model_drops_the_question(api):
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.sync(_snap(("local", ["x"])))
    c.request_ttl()
    c.ttl_key(ord("9"))
    assert c.ttl_target is None and not api
    c.request_ttl()
    c.ttl_key(27)
    assert not api
    c.request_ttl()
    c.sync(_snap(("local", [])))
    assert c.ttl_target is None


def test_keep_loaded_failure_is_reported(monkeypatch):
    monkeypatch.setattr(mtop.util, "http_post_json", lambda *a, **k: (False, "HTTP 404 Not Found"))
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.sync(_snap(("local", ["x"])))
    c.request_ttl()
    c.ttl_key(ord("1"))
    _wait_idle(c)
    assert c.notice(time.monotonic())[0] == "Keep-alive of x failed: HTTP 404 Not Found"
