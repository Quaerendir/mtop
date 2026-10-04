"""
mtop.swapwatch — when each llama-swap model was last used, for a live TTL.

llama-swap's `/running` reports a model's *configured* `ttl`, not the time
left: the countdown restarts after every proxied request (and when the model
turns ready), and the API exposes no "last request" time. What it does expose
is `/api/events`, the Server-Sent Events stream its web UI lives on. Three
kinds of event mark activity:

  inflight    `upsert`/`remove` of an OpenAI-style request, with its model —
              while one is open the model cannot expire
  logData     llama-swap's own access log (`source: proxy`), one line per
              finished request; `/upstream/<model>/...` calls (which never
              show up as inflight) name their model only here
  modelStatus every model's state; the TTL window opens on → `ready`

The first logData per source after a connect is the log backlog, with no
timestamps — it is skipped, so a model's last use stays unknown until mtop
sees it happen.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from collections.abc import Callable

from .logs import parse_gin
from .util import Endpoint

STREAM_TIMEOUT = 60.0   # s without a byte before reconnecting
RETRY_DELAY = 5.0       # s between connection attempts


class SwapActivity(threading.Thread):
    """Follows one endpoint's `/api/events`; answers `idle_since(model)`."""

    def __init__(self, ep: Endpoint, clock: Callable[[], float] = time.monotonic):
        super().__init__(daemon=True, name=f"mtop-events-{ep.label}")
        self.ep = ep
        self.clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._resp = None
        self._last: dict[str, float] = {}       # model -> clock() of last activity
        self._inflight: dict[str, str] = {}     # request id -> model
        self._states: dict[str, str] = {}       # model -> last modelStatus state
        self._backlog: set[str] = set()         # logData sources still owed a backlog

    # -- queries (any thread) --------------------------------------------------

    def idle_since(self, model: str) -> float | None:
        """clock() time the model's TTL window opened; now while a request is
        in flight; None when mtop has not seen it used since connecting."""
        with self._lock:
            if model in self._inflight.values():
                return self.clock()
            return self._last.get(model)

    # -- event handling --------------------------------------------------------

    def connected(self) -> None:
        with self._lock:
            self._backlog = {"proxy", "upstream"}
            self._inflight.clear()

    def feed(self, event: dict) -> None:
        """One decoded `data:` payload — `{"type": ..., "data": "<json>"}`."""
        try:
            data = json.loads(event.get("data") or "null")
        except (TypeError, ValueError):
            return
        kind = event.get("type")
        now = self.clock()
        with self._lock:
            if kind == "logData" and isinstance(data, dict):
                self._log(data, now)
            elif kind == "inflight" and isinstance(data, dict):
                self._inflight_op(data, now)
            elif kind == "modelStatus" and isinstance(data, list):
                self._status(data, now)

    def _log(self, data: dict, now: float) -> None:
        source = data.get("source")
        if source in self._backlog:
            self._backlog.discard(source)
            return
        if source != "proxy":
            return                      # the models' own output
        for line in str(data.get("data") or "").splitlines():
            g = parse_gin(line)
            if g and g["path"].startswith("/upstream/"):
                model = self._upstream_model(g["path"])
                if model:
                    self._last[model] = now

    def _upstream_model(self, path: str) -> str | None:
        """`/upstream/<model>/rest` -> model; an id may itself hold '/', so
        the longest prefix llama-swap knows wins."""
        parts = [urllib.parse.unquote(p) for p in path.split("?", 1)[0].split("/")[2:]]
        for n in range(len(parts), 0, -1):
            name = "/".join(parts[:n])
            if name in self._states:
                return name
        return parts[0] if parts and parts[0] else None

    def _inflight_op(self, data: dict, now: float) -> None:
        op = data.get("operation")
        if op == "snapshot":
            self._inflight = {str(r.get("id")): r.get("model", "")
                              for r in data.get("requests") or [] if isinstance(r, dict)}
        elif op == "upsert" and isinstance(data.get("request"), dict):
            r = data["request"]
            if r.get("model"):
                self._inflight[str(r.get("id"))] = r["model"]
        elif op == "remove":
            model = self._inflight.pop(str(data.get("id")), None)
            if model:
                self._last[model] = now

    def _status(self, models: list, now: float) -> None:
        for m in models:
            if not isinstance(m, dict) or not m.get("id"):
                continue
            name, state = m["id"], m.get("state")
            prev = self._states.get(name)
            if state == "ready" and prev is not None and prev != "ready":
                self._last[name] = now
            elif state != "ready":
                self._last.pop(name, None)
            self._states[name] = state

    # -- the stream ------------------------------------------------------------

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._resp = self.ep.open_stream("/api/events", STREAM_TIMEOUT)
                self.connected()
                self._read(self._resp)
            except Exception:
                pass
            finally:
                resp, self._resp = self._resp, None
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            self._stop.wait(RETRY_DELAY)

    def _read(self, resp) -> None:
        data: list[str] = []
        for raw in resp:
            if self._stop.is_set():
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                try:
                    event = json.loads("\n".join(data))
                except ValueError:
                    event = None
                data = []
                if isinstance(event, dict):
                    self.feed(event)

    def stop(self) -> None:
        self._stop.set()
        resp = self._resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
