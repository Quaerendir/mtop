"""
mtop.comfywatch — live progress of the running ComfyUI job, from `/ws`.

ComfyUI reports a job's progress on its websocket: `execution_start`,
`executing` (the node now running), `progress` (sampler step x/y),
`execution_success` / `_error` / `_interrupted`. It sends those to the
`client_id` the job was queued with, and to every connected client only
when the job had none — which is the case for jobs posted to `/prompt` by a
script without one. A job queued from the web UI carries the browser tab's
id, so mtop sees its existence (`/queue`) but none of its progress. mtop
never connects under another client's id: that would take the tab's events
away from it.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable

from .util import Endpoint
from .wsclient import OP_PING, WebSocket, ws_url

IDLE_TIMEOUT = 30.0     # s without a frame before a ping; a second one reconnects
RETRY_DELAY = 5.0       # s between connection attempts


class ComfyProgress(threading.Thread):
    """Follows one endpoint's `/ws`; `job()` returns the progress it saw."""

    def __init__(self, ep: Endpoint, clock: Callable[[], float] = time.monotonic):
        super().__init__(daemon=True, name=f"mtop-comfyws-{ep.label}")
        self.ep = ep
        self.clock = clock
        self.client_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._ws: WebSocket | None = None
        self.connected = False
        self.error = ""
        self._cur: dict | None = None       # the job being executed

    # -- queries (any thread) --------------------------------------------------

    def job(self, prompt_id: str | None = None) -> dict | None:
        """A copy of the running job's progress — of `prompt_id` when given,
        None when mtop has seen no event of it."""
        with self._lock:
            cur = self._cur
            if cur is None or (prompt_id and cur["prompt_id"] != prompt_id):
                return None
            return dict(cur)

    # -- event handling --------------------------------------------------------

    def feed(self, msg: dict) -> None:
        """One decoded text message: `{"type": ..., "data": {...}}`."""
        kind, data = msg.get("type"), msg.get("data")
        if not isinstance(data, dict):
            return
        pid = data.get("prompt_id")
        now = self.clock()
        with self._lock:
            if kind == "execution_start" and pid:
                self._cur = self._new(pid, now)
            elif kind == "execution_cached" and pid:
                self._job(pid, now)["cached"] = len(data.get("nodes") or [])
            elif kind == "executing" and pid:
                node = data.get("display_node") or data.get("node")
                if node is None:            # the job's end; execution_* says how
                    return
                cur = self._job(pid, now)
                if cur["node"] != str(node):
                    cur.update(node=str(node), node_started=now, value=None, max=None)
            elif kind == "progress" and pid:
                cur = self._job(pid, now)
                node = data.get("node")
                if node is not None and cur["node"] != str(node):
                    cur.update(node=str(node), node_started=now)
                cur["value"], cur["max"] = data.get("value"), data.get("max")
            elif (kind in ("execution_success", "execution_error", "execution_interrupted")
                  and self._cur and self._cur["prompt_id"] == pid):
                self._cur = None

    @staticmethod
    def _new(pid: str, now: float | None) -> dict:
        return {"prompt_id": pid, "started": now, "node": None, "node_started": None,
                "value": None, "max": None, "cached": 0}

    def _job(self, pid: str, now: float) -> dict:
        """The current job, or a new one when its start went unseen (mtop
        connected mid-job): its elapsed time is then unknown."""
        if self._cur is None or self._cur["prompt_id"] != pid:
            self._cur = self._new(pid, None)
        return self._cur

    # -- the socket ------------------------------------------------------------

    def run(self) -> None:
        url = ws_url(self.ep.url, f"/ws?clientId={self.client_id}")
        while not self._halt.is_set():
            ws = None
            try:
                ws = WebSocket.connect(url, IDLE_TIMEOUT, self.ep.headers or None,
                                       self.ep.context)
                self._ws = ws
                self.connected, self.error = True, ""
                self._read(ws)
            except Exception as e:
                if not self._halt.is_set():
                    self.error = str(e) or type(e).__name__
            finally:
                self.connected = False
                self._ws = None
                if ws is not None:
                    ws.close()
                with self._lock:
                    self._cur = None        # whatever ran may end unseen
            self._halt.wait(RETRY_DELAY)

    def _read(self, ws: WebSocket) -> None:
        pinged: int | None = None           # ws.frames when the ping went out
        while not self._halt.is_set():
            try:
                text = ws.recv()
            except TimeoutError:
                if ws.mid_frame or (pinged is not None and ws.frames == pinged):
                    raise                   # not even a pong: the connection is gone
                pinged = ws.frames
                ws.send(OP_PING)
                continue
            pinged = None
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if isinstance(msg, dict):
                self.feed(msg)

    def stop(self) -> None:
        self._halt.set()
        ws = self._ws
        if ws is not None:
            ws.shutdown()
