"""--control pull: the name prompt, the /api/pull stream, progress and cancel."""

import curses
import http.server
import json
import threading
import time

import pytest

import mtop
from conftest import FakeWin

GB = 1024**3


@pytest.fixture
def pull_server():
    """Answers /api/pull with `script`: a list of NDJSON dicts, or
    (status, body) for an HTTP error. `gate` holds the stream mid-way."""
    state = {"script": [], "seen": [], "gate": None}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["seen"].append((self.path, body))
            script = state["script"]
            if isinstance(script, tuple):
                code, err = script
                out = json.dumps(err).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for i, msg in enumerate(script):
                    if state["gate"] is not None and i == 2:
                        state["gate"].wait(5)
                    self.wfile.write(json.dumps(msg).encode() + b"\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", state
    finally:
        srv.shutdown()
        srv.server_close()


def _wait(c, timeout=5.0):
    end = time.monotonic() + timeout
    while c.pull is not None and time.monotonic() < end:
        time.sleep(0.01)
    assert c.pull is None


LAYER = "sha256:3a1b2c3d4e5f60718293a4b5c6d7e8f9"
OK_STREAM = [
    {"status": "pulling manifest"},
    {"status": f"pulling {LAYER}", "digest": LAYER, "total": 2 * GB, "completed": GB},
    {"status": f"pulling {LAYER}", "digest": LAYER, "total": 2 * GB, "completed": 2 * GB},
    {"status": "pulling small", "digest": "sha256:small", "total": 512, "completed": 512},
    {"status": "verifying sha256 digest"},
    {"status": "writing manifest"},
    {"status": "success"},
]


def test_pull_streams_to_success(pull_server):
    base, state = pull_server
    state["script"] = OK_STREAM
    c = mtop.ModelControl([mtop.Endpoint(base)])
    c.start_pull(0, "qwen3:8b")
    _wait(c)
    assert state["seen"] == [("/api/pull", {"model": "qwen3:8b"})]
    text, kind = c.notice(time.monotonic())
    assert kind == "ok" and text.startswith("Pulled qwen3:8b (2.0 GiB, ")


@pytest.mark.parametrize("script,expected", [
    ([{"status": "pulling manifest"},
      {"error": "pull model manifest: file does not exist"}],
     "Pull nope failed: pull model manifest: file does not exist"),
    ((404, {"error": "model not found"}), "Pull nope failed: model not found"),
    ((502, "not json"), "Pull nope failed: HTTP 502 Bad Gateway"),
    ([{"status": "pulling manifest"}], "Pull nope failed: stream ended before success"),
])
def test_pull_failures_say_why(pull_server, script, expected):
    base, state = pull_server
    state["script"] = script
    c = mtop.ModelControl([mtop.Endpoint(base)])
    c.start_pull(0, "nope")
    _wait(c)
    assert c.notice(time.monotonic()) == (expected, "err")


def test_cancel_closes_the_stream(pull_server):
    base, state = pull_server
    state["script"], state["gate"] = OK_STREAM, threading.Event()
    c = mtop.ModelControl([mtop.Endpoint(base)])
    c.start_pull(0, "big:70b")
    end = time.monotonic() + 5
    while not (c.pull and c.pull["layers"]) and time.monotonic() < end:
        time.sleep(0.01)
    assert c.pull_progress()["done"] == GB
    c.cancel_pull()
    state["gate"].set()
    _wait(c)
    assert c.notice(time.monotonic()) == ("Pull of big:70b cancelled", "err")


def test_prompt_typing_and_keys(monkeypatch):
    c = mtop.ModelControl([mtop.Endpoint("a=localhost:1"), mtop.Endpoint("b=localhost:2")])
    started = []
    monkeypatch.setattr(c, "start_pull", lambda ep, name: started.append((ep, name)))
    c.request_pull()
    for ch in "qwen3:8bx":
        c.pull_key(ord(ch))
    c.pull_key(curses.KEY_BACKSPACE)
    c.pull_key(ord(" "))                       # ignored: no spaces in names
    c.pull_key(9)                              # Tab: next endpoint
    assert c.notice(0)[0].startswith("Pull model on b: qwen3:8b▏ │ Enter: pull")
    c.pull_key(10)
    assert started == [(1, "qwen3:8b")] and c.pull_prompt is None
    c.request_pull()
    c.pull_key(ord("x"))
    c.pull_key(27)
    c.request_pull()
    c.pull_key(10)                             # empty name: nothing
    assert started == [(1, "qwen3:8b")]


def test_one_pull_at_a_time_and_not_under_llama_swap():
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.pull = {"ep": 0, "name": "a"}
    c.request_pull()
    assert c.pull_prompt is None and c.notice(0)[0] == "Already pulling a"
    s = mtop.ModelControl([mtop.Endpoint("localhost:8001")], backend="llama-swap")
    s.request_pull()
    assert s.pull_prompt is None
    assert s.notice(0)[0] == "llama-swap has no pull: its models come from its config"


def test_progress_rate_and_eta():
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    state = {"ep": 0, "name": "m", "status": "", "layers": {}, "rate": None,
             "sample": None, "started": 0.0}
    c.pull = state
    c._pull_update(state, {"status": f"pulling {LAYER}", "digest": LAYER,
                           "total": 4 * GB, "completed": 0}, 10.0)
    c._pull_update(state, {"status": f"pulling {LAYER}", "digest": LAYER,
                           "total": 4 * GB, "completed": GB}, 12.0)
    pr = c.pull_progress()
    assert pr["rate"] == GB / 2 and pr["eta"] == 6.0
    assert pr["status"] == "pulling sha256:3a1b2…" and pr["done"] == GB


def test_render_pull_row():
    c = mtop.ModelControl([mtop.Endpoint("localhost:1")])
    c.pull = {"ep": 0, "name": "qwen3:8b", "status": "pulling manifest",
              "layers": {"d": (GB, 4 * GB)}, "rate": 40 * 1024**2, "sample": None}
    w = FakeWin(rows=10, cols=140)
    mtop.ui.render_pull(w, c)
    line = w.line(8)
    assert line.startswith(" PULL qwen3:8b [█████░░░░░") and " 25.0%" in line
    assert "1.0 GiB / 4.0 GiB  40 MiB/s  ETA 1m 16s  pulling manifest  │ X: cancel" in line


@pytest.mark.parametrize("n,text", [(512, "512 B"), (2048, "2 KiB"), (340 * 1024**2, "340 MiB"),
                                    (int(1.25 * GB), "1.2 GiB")])
def test_fmt_bytes(n, text):
    assert mtop.ui.fmt_bytes(n) == text
