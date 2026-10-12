"""ComfyUI live progress (/ws) and --control: the WebSocket client, the event
follower, the job line, the actions, the job metrics."""

import json
import socket
import struct
import threading
import time

import mtop
from conftest import FakeWin

from mtop import comfyui, export, wsclient
from mtop.comfywatch import ComfyProgress


# ── WebSocket client ────────────────────────────────────────────────────────

def _server_frame(op: int, payload: bytes, fin: bool = True) -> bytes:
    """An unmasked server frame."""
    n = len(payload)
    head = bytes([(0x80 if fin else 0) | op])
    if n < 126:
        head += bytes([n])
    elif n < 1 << 16:
        head += bytes([126]) + struct.pack("!H", n)
    else:
        head += bytes([127]) + struct.pack("!Q", n)
    return head + payload


def _read_client_frame(sock: socket.socket) -> tuple[int, bytes]:
    b0, b1 = sock.recv(2)
    n = b1 & 0x7F
    mask = sock.recv(4)
    data = sock.recv(n) if n else b""
    return b0 & 0x0F, bytes(b ^ mask[i % 4] for i, b in enumerate(data))


class _WsServer:
    """One-connection WebSocket server: handshake, then `script(conn)`."""

    def __init__(self, script, status="101 Switching Protocols", accept=None):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.request = b""
        self.script, self.status, self.accept = script, status, accept
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.sock.accept()
        with conn:
            while b"\r\n\r\n" not in self.request:
                self.request += conn.recv(4096)
            key = next(ln.split(":", 1)[1].strip() for ln in self.request.decode().split("\r\n")
                       if ln.lower().startswith("sec-websocket-key"))
            accept = self.accept or wsclient.accept_key(key)
            conn.sendall(f"HTTP/1.1 {self.status}\r\nUpgrade: websocket\r\n"
                         f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                         .encode())
            self.script(conn)
        self.sock.close()


def test_encode_frame_masks_and_sizes():
    f = wsclient.encode_frame(wsclient.OP_PING, b"abc", mask=b"\x01\x02\x03\x04")
    assert f[:2] == bytes([0x89, 0x83]) and f[2:6] == b"\x01\x02\x03\x04"
    assert bytes(b ^ f[2 + i % 4] for i, b in enumerate(f[6:])) == b"abc"
    assert wsclient.encode_frame(1, b"x" * 300)[1] == 0x80 | 126
    assert wsclient.encode_frame(1, b"x" * 70000)[1] == 0x80 | 127


def test_ws_url():
    assert wsclient.ws_url("http://h:8188", "/ws?clientId=a") == "ws://h:8188/ws?clientId=a"
    assert wsclient.ws_url("https://h/comfy/", "/ws") == "wss://h/comfy/ws"


def test_recv_text_skips_binary_joins_fragments_and_answers_ping():
    got = {}

    def script(conn):
        conn.sendall(_server_frame(wsclient.OP_BIN, b"\x00" * 70000)       # a preview image
                     + _server_frame(wsclient.OP_TEXT, b'{"a":', fin=False)
                     + _server_frame(wsclient.OP_PING, b"hi")             # between fragments
                     + _server_frame(wsclient.OP_CONT, b"1}"))
        got["pong"] = _read_client_frame(conn)
        conn.sendall(_server_frame(wsclient.OP_CLOSE, b""))
        time.sleep(0.2)

    srv = _WsServer(script)
    ws = wsclient.WebSocket.connect(f"ws://127.0.0.1:{srv.port}/ws?clientId=x", 5,
                                    {"Authorization": "Bearer t"})
    assert ws.recv() == '{"a":1}' and not ws.mid_frame
    srv.thread.join(2)
    assert got["pong"] == (wsclient.OP_PONG, b"hi")
    req = srv.request.decode()
    assert req.startswith("GET /ws?clientId=x HTTP/1.1") and "Authorization: Bearer t" in req
    try:
        ws.recv()
        raise AssertionError("close frame must raise")
    except wsclient.WebSocketError as e:
        assert "closed" in str(e)
    ws.close()


def test_handshake_rejects_non_101_and_a_bad_accept():
    for kwargs, msg in (({"status": "403 Forbidden"}, "403"),
                        ({"accept": "bogus"}, "Accept")):
        srv = _WsServer(lambda conn: None, **kwargs)
        try:
            wsclient.WebSocket.connect(f"ws://127.0.0.1:{srv.port}/ws", 5)
            raise AssertionError("handshake must fail")
        except wsclient.WebSocketError as e:
            assert msg in str(e)


# ── following /ws ───────────────────────────────────────────────────────────

class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _msg(kind, **data):
    return {"type": kind, "data": data}


def test_feed_tracks_node_steps_and_end():
    clock = _Clock()
    w = ComfyProgress(mtop.Endpoint("localhost:8188"), clock)
    w.feed(_msg("status", status={"exec_info": {"queue_remaining": 1}}))
    w.feed(_msg("execution_start", prompt_id="p1", timestamp=1))
    w.feed(_msg("execution_cached", prompt_id="p1", nodes=["1", "2"]))
    clock.t = 101.0
    w.feed(_msg("executing", prompt_id="p1", node="5", display_node="5"))
    clock.t = 105.0
    w.feed(_msg("progress", prompt_id="p1", node="5", value=4, max=20))
    job = w.job("p1")
    assert job["started"] == 100.0 and job["node"] == "5" and job["node_started"] == 101.0
    assert (job["value"], job["max"], job["cached"]) == (4, 20, 2)
    assert w.job("other") is None
    w.feed(_msg("executing", prompt_id="p1", node="8"))        # VAE decode: steps reset
    assert w.job()["value"] is None and w.job()["node"] == "8"
    w.feed(_msg("executing", prompt_id="p1", node=None))       # end marker: nothing yet
    assert w.job() is not None
    w.feed(_msg("execution_success", prompt_id="p1"))
    assert w.job() is None


def test_events_without_a_seen_start_leave_the_start_unknown():
    w = ComfyProgress(mtop.Endpoint("localhost:8188"), _Clock())
    w.feed(_msg("progress", prompt_id="p2", node="3", value=1, max=8))
    assert w.job("p2")["started"] is None and w.job("p2")["value"] == 1
    w.feed(_msg("execution_interrupted", prompt_id="other"))
    assert w.job("p2") is not None
    w.feed(_msg("execution_error", prompt_id="p2", exception_message="OOM"))
    assert w.job() is None


def test_follower_reads_a_live_socket_and_stops():
    def script(conn):
        for m in (_msg("execution_start", prompt_id="p9"),
                  _msg("progress", prompt_id="p9", node="5", value=7, max=10)):
            conn.sendall(_server_frame(wsclient.OP_TEXT, json.dumps(m).encode()))
        time.sleep(1.0)

    srv = _WsServer(script)
    w = ComfyProgress(mtop.Endpoint(f"http://127.0.0.1:{srv.port}"))
    w.start()
    deadline = time.monotonic() + 3
    while (w.job("p9") or {}).get("value") != 7 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert w.connected and w.job("p9")["max"] == 10
    assert f"clientId={w.client_id}" in srv.request.decode()
    w.stop()
    w.join(2)
    assert not w.is_alive()


# ── queue / history helpers ─────────────────────────────────────────────────

GRAPH = {"5": {"class_type": "KSampler", "inputs": {}},
         "8": {"class_type": "VAEDecode", "inputs": {}}}
QUEUE = {"queue_running": [[3, "p1", GRAPH, {"client_id": "tab"}, ["9"]]],
         "queue_pending": []}


def test_node_class_and_client():
    assert comfyui.node_class(QUEUE, "p1", "5") == "KSampler"
    assert comfyui.node_class(QUEUE, "p1", "77") is None
    assert comfyui.node_class(QUEUE, "nope", "5") is None
    assert comfyui.queued_by_client(QUEUE, "p1")
    assert not comfyui.queued_by_client({"queue_running": [[3, "p1", GRAPH, {}, []]]}, "p1")


def _entry(status, msgs):
    return {"prompt": [1, "x", {}, {}, []],
            "status": {"status_str": status, "completed": status == "success",
                       "messages": [[k, {"timestamp": t}] for k, t in msgs]}}


def test_last_job():
    hist = {"old": _entry("success", [("execution_start", 0), ("execution_success", 5000)]),
            "new": _entry("error", [("execution_start", 1_700_000_000_000),
                                    ("execution_interrupted", 1_700_000_012_500)])}
    last = comfyui.last_job(hist)
    assert last == {"prompt_id": "new", "outcome": "interrupted", "took": 12.5,
                    "ended": "2023-11-14T22:13:32+00:00"}
    assert comfyui.last_job({}) is None and comfyui.last_job(None) is None


# ── the collector's job view ────────────────────────────────────────────────

class _FakeWatch:
    def __init__(self, job=None, connected=True):
        self._j, self.connected, self.error = job, connected, ""

    def job(self, pid=None):
        return dict(self._j) if self._j and self._j["prompt_id"] == pid else None

    def stop(self):
        pass


def _col():
    return mtop.Collector(container="comfyui", api_url="http://localhost:8188", interval=1.0,
                          show_gpu=False, mode="api", backend="comfyui")


def test_job_live_with_eta(monkeypatch):
    monkeypatch.setattr(mtop.collector, "ComfyProgress", _FakeWatch)
    c = _col()
    now = time.monotonic()
    c._activity[c.primary.label] = _FakeWatch({
        "prompt_id": "p1", "started": now - 60, "node": "5", "node_started": now - 40,
        "value": 10, "max": 30, "cached": 2})
    job = c._comfyui_job(c.primary, QUEUE, "p1")
    assert job["live"] and job["node_class"] == "KSampler" and job["elapsed_exact"]
    assert 59 <= job["elapsed"] <= 61
    assert 79 <= job["eta"] <= 81                   # 4 s/step, 20 steps to go
    assert c._comfyui_ws_state(c.primary) == "connected"


def test_job_unseen_counts_from_first_sight(monkeypatch):
    monkeypatch.setattr(mtop.collector, "ComfyProgress", _FakeWatch)
    c = _col()
    c._activity[c.primary.label] = _FakeWatch(None)
    c._job_seen[c.primary.label] = ("p1", time.monotonic() - 30, False)
    job = c._comfyui_job(c.primary, QUEUE, "p1")
    assert not job["live"] and not job["elapsed_exact"] and job["elapsed"] >= 30
    assert job["queued_by_client"] and job["eta"] is None
    assert c._comfyui_job(c.primary, QUEUE, None) is None
    assert c.primary.label not in c._job_seen        # idle forgets the job


def test_job_first_seen_after_an_earlier_poll_is_exact():
    c = _col()
    assert c._comfyui_job(c.primary, QUEUE, "p1")["elapsed_exact"] is False   # first read
    assert c._comfyui_job(c.primary, QUEUE, "p1")["elapsed_exact"] is False   # same job
    c._comfyui_job(c.primary, QUEUE, None)                                    # idle
    assert c._comfyui_job(c.primary, QUEUE, "p1")["elapsed_exact"] is True


def test_one_shot_has_no_websocket():
    c = _col()
    assert c._comfyui_ws_state(c.primary) is None
    job = c._comfyui_job(c.primary, QUEUE, "p1")
    assert not job["live"] and job["elapsed"] < 1


# ── the job line ────────────────────────────────────────────────────────────

def _job(**kw):
    j = {"prompt_id": "694df408-dc51", "live": True, "queued_by_client": False, "node": "5",
         "node_class": "KSampler", "value": 12, "max": 30, "elapsed": 65.0,
         "elapsed_exact": True, "eta": 25.0, "cached": 0}
    j.update(kw)
    return j


def test_job_line_live():
    text, kind = mtop.ui.comfyui_job_line({"job": _job(), "websocket": "connected"})
    assert kind == "ok"
    assert text == ("job 694df408 · KSampler #5 [██████░░░░░░░░░░] 12/30  40% · 1m 5s · "
                    "ETA 0m 25s")


def test_job_line_without_progress_says_why():
    ep = {"job": _job(live=False, elapsed_exact=False, queued_by_client=True),
          "websocket": "connected"}
    text, kind = mtop.ui.comfyui_job_line(ep)
    assert kind == "warn" and "running ≥1m 5s" in text
    assert "progress goes only to the client that queued it" in text
    ep["websocket"] = "connection refused"
    assert "no progress: websocket connection refused" in mtop.ui.comfyui_job_line(ep)[0]
    ep["websocket"] = None
    assert mtop.ui.comfyui_job_line(ep)[0].endswith("running ≥1m 5s")


def test_job_line_last_job_when_idle():
    ep = {"job": None, "last_job": {"outcome": "success", "took": 116.0,
                                    "ended": "2020-01-01T00:00:00+00:00"}}
    text, kind = mtop.ui.comfyui_job_line(ep)
    assert text.startswith("last job: success in 1m 56s, ") and text.endswith("ago")
    assert kind == "dim"
    ep["last_job"]["outcome"] = "interrupted"
    assert mtop.ui.comfyui_job_line(ep)[1] == "err"
    assert mtop.ui.comfyui_job_line({}) is None


def test_job_line_on_screen():
    w = FakeWin(rows=10, cols=160)
    mtop.render_models(w, 0, {"backend": "comfyui", "endpoints": [{
        "models_ok": True, "queue": {"running": 1, "pending": 0}, "models": [],
        "job": _job(), "websocket": "connected"}]})
    assert "KSampler #5" in w.text() and "ETA" in w.text()


# ── --control ───────────────────────────────────────────────────────────────

def _control(monkeypatch, replies):
    calls = []

    def fake(method):
        def f(url, *a, **k):
            path = url.split("8188", 1)[1]
            body = a[0] if method == "POST-JSON" and a else None
            calls.append((method, path, body))
            return replies.get(path, (True, ""))
        return f
    monkeypatch.setattr(mtop.util, "http_post_json", fake("POST-JSON"))
    monkeypatch.setattr(mtop.util, "http_request", fake("REQ"))
    c = mtop.ModelControl([mtop.Endpoint("http://localhost:8188")], "comfyui")
    return c, calls


def _snap(running=None, pending=0):
    return {"backend": "comfyui", "endpoints": [{
        "models_ok": True, "queue": {"running": int(bool(running)), "pending": pending,
                                     "prompt_id": running},
        "models": [{"name": "m.safetensors"}]}]}


def _wait_idle(c):
    deadline = time.monotonic() + 2
    while c.busy and time.monotonic() < deadline:
        time.sleep(0.01)


def test_no_cursor_and_nothing_to_do(monkeypatch):
    c, calls = _control(monkeypatch, {})
    c.sync(_snap())
    assert c.selected is None
    c.request_stop()
    assert c.notice(time.monotonic()) == ("No job is running", "err")
    c.request_clear()
    assert c.notice(time.monotonic()) == ("No jobs are waiting", "err")
    c.request_load()
    c.request_ttl()
    assert not c.asking and not calls


def test_cancel_running_job(monkeypatch):
    c, calls = _control(monkeypatch, {})
    c.sync(_snap(running="694df408-dc51"))
    c.request_stop()
    assert c.notice(time.monotonic()) == ("Cancel the running job 694df408? [y/N]", "ask")
    c.answer(False)
    assert not calls
    c.request_stop()
    c.answer(True)
    _wait_idle(c)
    assert calls == [("REQ", "/api/jobs/694df408-dc51/cancel", None)]
    c.sync({**c._snap, "ts": time.monotonic()})       # a snapshot after the action
    assert c.notice(time.monotonic()) == ("Cancelled job 694df408", "ok")


def test_cancel_falls_back_to_interrupt(monkeypatch):
    c, calls = _control(monkeypatch, {"/api/jobs/p1/cancel": (False, "HTTP 404 Not Found")})
    c.sync(_snap(running="p1"))
    c.request_stop()
    c.answer(True)
    _wait_idle(c)
    assert calls[1] == ("POST-JSON", "/interrupt", {"prompt_id": "p1"})


def test_clear_and_free(monkeypatch):
    c, calls = _control(monkeypatch, {})
    c.sync(_snap(pending=3))
    c.request_clear()
    assert c.notice(time.monotonic()) == ("Clear the queue (3 pending jobs)? [y/N]", "ask")
    c.answer(True)
    _wait_idle(c)
    c.request_free()
    assert c.notice(time.monotonic())[0].startswith("Unload all models")
    c.answer(True)
    _wait_idle(c)
    assert calls == [
        ("POST-JSON", "/queue", {"clear": True}),
        ("POST-JSON", "/free", {"unload_models": True, "free_memory": True})]


def test_failure_is_reported(monkeypatch):
    c, _ = _control(monkeypatch, {"/free": (False, "HTTP 500 Internal Server Error")})
    c.sync(_snap())
    c.request_free()
    c.answer(True)
    _wait_idle(c)
    assert c.notice(time.monotonic()) == (
        "Free of all models failed: HTTP 500 Internal Server Error", "err")


def test_footer_and_help_for_comfyui():
    groups = mtop.ui.footer_groups(1.0, False, True, control="comfyui")
    assert [k for k, _, _ in groups[1]] == ["s", "C", "F"]
    rows = mtop.ui.help_lines(1.0, False, True, True, True, False, "comfyui", None)
    keys = [k for k, _ in rows]
    assert "C" in keys and "F" in keys and "t" not in keys and "L" not in keys


# ── Prometheus ──────────────────────────────────────────────────────────────

def test_job_metrics():
    snap = {"backend": "comfyui", "status": "api-only", "models_ok": True, "endpoints": [{
        "label": "l", "url": "u", "models_ok": True, "models": [],
        "queue": {"running": 1, "pending": 0}, "job": _job(), "websocket": "connected"}]}
    text = export.prometheus_text(snap, "9.9.9", now=0.0)
    assert ('mtop_comfyui_job_elapsed_seconds{endpoint="l",url="u",'
            'prompt_id="694df408-dc51"} 65') in text
    assert ('mtop_comfyui_job_progress_ratio{endpoint="l",url="u",prompt_id="694df408-dc51",'
            'node="5"} 0.4') in text
    assert 'mtop_comfyui_websocket_up{endpoint="l",url="u"} 1' in text
