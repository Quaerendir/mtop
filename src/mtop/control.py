"""
mtop.control — the --control actions: stop, load, keep-alive, pull.

State and HTTP only; ui.py draws it. The requests run on threads so the
curses loop keeps drawing while a 100B model loads or a pull streams in.
`notice()` says *what kind* of message it has (NOTICE_KINDS) and the UI
picks the color.
"""

from __future__ import annotations

import curses
import json
import threading
import time
import urllib.error
import urllib.parse

from .util import fmt_duration, http_error_text, shutdown_stream

UNLOAD_TIMEOUT = 30       # s — a busy server finishes in-flight requests before unloading
LOAD_TIMEOUT = 900        # s — a 100B+ model from a cold disk takes minutes
# `t` on a loaded model: key -> (label, keep_alive value Ollama accepts)
TTL_CHOICES = (("30m", "30m"), ("2h", "2h"), ("24h", "24h"), ("forever", -1))
NOTICE_HOLD = 5.0         # s a model-action result stays in the footer
PULL_READ_TIMEOUT = 120   # s with no progress line before a pull counts as stalled
# What a footer notice is, for the UI to color: key hints, a warning, a y/N
# question, an action in progress, its success or its failure.
NOTICE_KINDS = ("info", "warn", "ask", "ok", "err")


def max_loaded_models(snap: dict) -> tuple[int, str] | None:
    """How many models the primary server keeps resident, and where that
    number comes from — None when its environment is not readable (api mode,
    a local server owned by another user): then there is nothing to warn with.

    Unset or 0 means Ollama's default, 3 per GPU (3 on CPU).
    """
    server = snap.get("server") or {}
    if not server.get("env_source"):
        return None
    raw = (server.get("env") or {}).get("OLLAMA_MAX_LOADED_MODELS", "")
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n > 0:
        return n, "OLLAMA_MAX_LOADED_MODELS"
    return 3 * max(1, len(snap.get("gpus") or [])), "Ollama default"


class ModelControl:
    """The --control cursor over loaded models, stop (unload) and load.

    The selection is (endpoint index, model name), not a row number: /api/ps
    order can change between snapshots, and the cursor must stay on the model
    the user picked instead of sliding onto a neighbour right before `y`.
    Stop is `POST /api/generate {"model": name, "keep_alive": 0}` — the same
    request `ollama stop` sends. Load is the same call without `keep_alive`
    (the server's OLLAMA_KEEP_ALIVE applies), picked from `/api/tags` in an
    overlay. Keep-loaded (`t`) sends the same call with a new `keep_alive`
    *and* the model's current `num_ctx`: without it Ollama treats the
    request as asking for default options and reloads the model with the
    Modelfile's context (seen on 0.34: 8192 -> 65536, new runner). All run
    on a thread so the UI keeps drawing; one at a time.

    With llama-swap the cursor walks the whole catalog (every configured
    model). Stop is `POST /api/models/unload/<model>`; load is
    `GET /upstream/<model>/health`, which starts the model and answers once
    it is up — and keeps starting it if mtop goes away. llama-swap runs one
    model at a time unless its config groups them, so a load while another
    model runs asks first. There is no keep-alive to change: TTL is config.

    Pull (`P`, Ollama only) asks for a name and streams `POST /api/pull`.
    It runs beside the other actions — a 70 GB download should not block a
    stop — one pull at a time, and `X` cancels it by closing the stream,
    which Ollama takes as a cancel (the blobs already fetched stay, so the
    next pull resumes).
    """

    def __init__(self, endpoints: list, backend: str = "ollama"):
        self.endpoints = endpoints
        self.swap = backend == "llama-swap"
        self.selected: tuple[int, str] | None = None
        self.confirm: tuple[int, str] | None = None
        self.busy: tuple[str, int, str, float] | None = None  # verb, ep, name, started
        self.picker: dict | None = None
        # (endpoint, model, models that may be evicted, limit, limit source)
        self.load_confirm: tuple[int, str, list[str], int, str] | None = None
        self.ttl_target: tuple[int, str] | None = None
        # llama-swap load: (endpoint, model, running models it would stop)
        self.swap_confirm: tuple[int, str, list[str]] | None = None
        self._snap: dict = {}
        self._index = 0
        self._result: tuple[str, bool, float] | None = None   # text, ok, monotonic until
        self._lock = threading.Lock()
        self.pull_prompt: dict | None = None    # {"ep", "text"} while typing a name
        self.pull: dict | None = None           # the running pull, see _pull_run

    @staticmethod
    def rows(snap: dict) -> list[tuple[int, str]]:
        endpoints = snap.get("endpoints") or [snap]
        return [(i, m.get("name", "?")) for i, ep in enumerate(endpoints)
                if ep.get("models_ok") for m in ep.get("models", [])]

    def sync(self, snap: dict) -> None:
        """Re-anchor the cursor after a new snapshot; a vanished model hands
        the cursor to whatever now sits at its position."""
        self._snap = snap
        rows = self.rows(snap)
        if self.selected in rows:
            self._index = rows.index(self.selected)
        elif rows:
            self._index = min(self._index, len(rows) - 1)
            self.selected = rows[self._index]
        else:
            self.selected = None
        if self.confirm not in rows:
            self.confirm = None
        if self.ttl_target not in rows:
            self.ttl_target = None
        if self.swap_confirm and self.swap_confirm[:2] not in rows:
            self.swap_confirm = None

    def move(self, delta: int, snap: dict) -> None:
        rows = self.rows(snap)
        if not rows:
            return
        self._index = max(0, min(len(rows) - 1, self._index + delta))
        self.selected = rows[self._index]

    def request_stop(self) -> None:
        if not self.selected or self.busy:
            return
        if self.swap and not self._running(self.selected):
            self._say(f"{self.label(self.selected)} is not running", ok=False)
            return
        self.confirm = self.selected

    def request_ttl(self) -> None:
        if self.swap:
            self._say("llama-swap sets each model's TTL in its config", ok=False)
            return
        if self.selected and not self.busy:
            self.ttl_target = self.selected

    def request_load(self) -> None:
        """`L`: the /api/tags picker (Ollama), or the selected catalog entry
        (llama-swap — its whole catalog is already on screen)."""
        if not self.swap:
            self.open_picker()
            return
        if not self.selected or self.busy:
            return
        if self._running(self.selected):
            self._say(f"{self.label(self.selected)} is already running", ok=True)
            return
        i = self.selected[0]
        others = [m.get("name", "?") for m in self._models(i)
                  if m.get("running") and m.get("name") != self.selected[1]]
        if others:
            self.swap_confirm = (i, self.selected[1], others)
        else:
            self._load(self.selected)

    def _models(self, ep: int) -> list[dict]:
        endpoints = self._snap.get("endpoints") or [self._snap]
        return (endpoints[ep].get("models") or []) if ep < len(endpoints) else []

    def _running(self, target: tuple[int, str]) -> bool:
        return bool((self._model(target) or {}).get("running", True))

    def _say(self, text: str, ok: bool) -> None:
        with self._lock:
            self._result = (text, ok, time.monotonic() + NOTICE_HOLD)

    def _load(self, target: tuple[int, str]) -> None:
        if self.swap:
            path = f"/upstream/{urllib.parse.quote(target[1])}/health"
            self._start("Loading", target, None, LOAD_TIMEOUT, request=("GET", path))
        else:
            self._start("Loading", target, {}, LOAD_TIMEOUT)

    def ttl_key(self, key: int) -> None:
        """1-4 picks a keep-alive from TTL_CHOICES; any other key cancels."""
        target, self.ttl_target = self.ttl_target, None
        idx = key - ord("1")
        if target is None or not 0 <= idx < len(TTL_CHOICES):
            return
        label, value = TTL_CHOICES[idx]
        extra: dict = {"keep_alive": value}
        ctx = (self._model(target) or {}).get("context_length")
        if ctx:
            extra["options"] = {"num_ctx": int(ctx)}
        when = "until unloaded" if value == -1 else f"for {label}"
        self._start("Extending", target, extra, UNLOAD_TIMEOUT,
                    done=f"Keeping {self.label(target)} loaded {when}")

    def _model(self, target: tuple[int, str]) -> dict | None:
        """The /api/ps entry of `target` in the latest snapshot."""
        endpoints = self._snap.get("endpoints") or [self._snap]
        if target[0] >= len(endpoints):
            return None
        return next((m for m in endpoints[target[0]].get("models") or []
                     if m.get("name") == target[1]), None)

    def answer(self, yes: bool) -> None:
        """y/N for whichever question is open: stop, or a load that evicts."""
        target, self.confirm = self.confirm, None
        load, self.load_confirm = self.load_confirm, None
        swap, self.swap_confirm = self.swap_confirm, None
        if yes and target and self.swap:
            path = f"/api/models/unload/{urllib.parse.quote(target[1])}"
            self._start("Stopping", target, None, UNLOAD_TIMEOUT, request=("POST", path))
        elif yes and target:
            self._start("Stopping", target, {"keep_alive": 0}, UNLOAD_TIMEOUT)
        elif yes and (load or swap):
            self._load((load or swap)[:2])

    @property
    def asking(self) -> bool:
        return bool(self.confirm or self.load_confirm or self.swap_confirm)

    def evictions(self, ep: int) -> tuple[list[str], int, str] | None:
        """Resident models on `ep` when it is at its model-count limit.

        Only the primary's limit is known (its environment is what mtop
        reads). Which model Ollama drops is its scheduler's choice, so the
        warning names the candidates, not a victim.
        """
        if ep != 0 or self.swap:
            return None
        limit = max_loaded_models(self._snap)
        if limit is None:
            return None
        endpoints = self._snap.get("endpoints") or [self._snap]
        loaded = [m.get("name", "?") for m in (endpoints[0].get("models") or [])]
        return (loaded, *limit) if len(loaded) >= limit[0] else None

    # ── load picker ──────────────────────────────────────────────────────────

    def open_picker(self, ep: int | None = None) -> None:
        """Overlay listing /api/tags of the selected model's endpoint (or the
        primary). The list is fetched on a thread; `items` is None until then."""
        if self.busy:
            return
        if ep is None:
            ep = self.selected[0] if self.selected else 0
        picker = {"ep": ep, "items": None, "err": "", "index": 0, "top": 0}
        self.picker = picker
        threading.Thread(target=self._fetch_tags, args=(picker,), daemon=True,
                         name="mtop-tags").start()

    def _fetch_tags(self, picker: dict) -> None:
        ok, data = self.endpoints[picker["ep"]].get_json("/api/tags")
        models = data.get("models") if ok and isinstance(data, dict) else None
        with self._lock:
            if isinstance(models, list):
                picker["items"] = sorted(models, key=lambda m: m.get("name", "").casefold())
            else:
                picker["items"] = []
                picker["err"] = str(data) if not ok else "unexpected /api/tags reply"

    def picker_key(self, key: int, page: int = 10) -> None:
        """Keys while the overlay is open; everything else is swallowed."""
        pk = self.picker
        if pk is None:
            return
        if key in (27, ord("q"), ord("L")):
            self.picker = None
            return
        if key == 9 and len(self.endpoints) > 1:                  # Tab: next endpoint
            self.open_picker((pk["ep"] + 1) % len(self.endpoints))
            return
        with self._lock:
            items = pk["items"] or []
        if not items:
            return
        delta = {curses.KEY_UP: -1, curses.KEY_DOWN: 1, curses.KEY_PPAGE: -page,
                 curses.KEY_NPAGE: page, curses.KEY_HOME: -len(items),
                 curses.KEY_END: len(items)}.get(key)
        if delta is not None:
            pk["index"] = max(0, min(len(items) - 1, pk["index"] + delta))
        elif key in (10, 13, curses.KEY_ENTER):
            name = items[pk["index"]].get("name", "?")
            self.picker = None
            full = self.evictions(pk["ep"])
            if full and name not in full[0]:      # reloading a resident model evicts nothing
                self.load_confirm = (pk["ep"], name, *full)
            else:
                self._load((pk["ep"], name))

    # ── pull ─────────────────────────────────────────────────────────────────

    def request_pull(self) -> None:
        """`P`: ask for a model name to pull on the selected model's endpoint."""
        if self.swap:
            self._say("llama-swap has no pull: its models come from its config", ok=False)
            return
        if self.pull:
            self._say(f"Already pulling {self.label((self.pull['ep'], self.pull['name']))}",
                      ok=False)
            return
        self.pull_prompt = {"ep": self.selected[0] if self.selected else 0, "text": ""}

    def pull_key(self, key: int) -> None:
        """Keys while the name prompt is open: type, Backspace, Tab, Enter, Esc."""
        pp = self.pull_prompt
        if pp is None:
            return
        if key == 27:
            self.pull_prompt = None
        elif key in (10, 13, curses.KEY_ENTER):
            self.pull_prompt = None
            name = pp["text"].strip()
            if name:
                self.start_pull(pp["ep"], name)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            pp["text"] = pp["text"][:-1]
        elif key == 9 and len(self.endpoints) > 1:
            pp["ep"] = (pp["ep"] + 1) % len(self.endpoints)
        elif 32 < key < 127:                        # model names have no spaces
            pp["text"] += chr(key)

    def start_pull(self, ep: int, name: str) -> None:
        if self.pull:
            return
        state = {"ep": ep, "name": name, "status": "connecting", "layers": {},
                 "started": time.monotonic(), "rate": None, "sample": None,
                 "cancel": threading.Event(), "resp": None}
        self.pull = state
        threading.Thread(target=self._pull_run, args=(state,), daemon=True,
                         name="mtop-pull").start()

    def cancel_pull(self) -> None:
        """Report the cancel at once; the reader thread drops the connection
        (which is what makes Ollama stop) on its next line, or right away
        when the socket can be shut down under it."""
        state = self.pull
        if not state:
            return
        state["cancel"].set()
        shutdown_stream(state.get("resp"))
        with self._lock:
            if self.pull is state:
                self.pull = None
            self._result = (f"Pull of {self.label((state['ep'], state['name']))} cancelled",
                            False, time.monotonic() + NOTICE_HOLD)

    def _pull_run(self, state: dict) -> None:
        """Stream /api/pull. Each NDJSON line is a status, and while a layer
        downloads also its digest with total and completed bytes."""
        target = (state["ep"], state["name"])
        ok, err = False, "stream ended before success"
        try:
            resp = self.endpoints[state["ep"]].open_post_stream(
                "/api/pull", {"model": state["name"]}, PULL_READ_TIMEOUT)
            state["resp"] = resp
            with resp:
                for raw in resp:
                    if state["cancel"].is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    if msg.get("error"):
                        err = msg["error"]
                        break
                    with self._lock:
                        self._pull_update(state, msg, time.monotonic())
                    if msg.get("status") == "success":
                        ok = True
                        break
        except urllib.error.HTTPError as e:
            err = _http_error_body(e)
        except Exception as e:
            err = http_error_text(e)
        if state["cancel"].is_set():
            return                     # cancel_pull already said so
        took = fmt_duration(time.monotonic() - state["started"])
        if ok:
            total = sum(t for _, t in state["layers"].values())
            size = f"{total / 1024**3:.1f} GiB, " if total else ""
            text = f"Pulled {self.label(target)} ({size}{took})"
        else:
            text = f"Pull {state['name']} failed: {err}"
        with self._lock:
            self._result = (text, ok, time.monotonic() + NOTICE_HOLD)
            if self.pull is state:
                self.pull = None

    @staticmethod
    def _pull_update(state: dict, msg: dict, now: float) -> None:
        state["status"] = msg.get("status") or state["status"]
        digest, total = msg.get("digest"), msg.get("total")
        if digest and total:
            state["layers"][digest] = (msg.get("completed") or 0, total)
        done = sum(c for c, _ in state["layers"].values())
        # Rate over >= 1 s windows, smoothed: per-line deltas are far too jumpy.
        sample = state["sample"]
        if sample is None:
            state["sample"] = (now, done)
        elif now - sample[0] >= 1.0:
            rate = max(0.0, (done - sample[1]) / (now - sample[0]))
            prev = state["rate"]
            state["rate"] = rate if prev is None else 0.7 * prev + 0.3 * rate
            state["sample"] = (now, done)

    def pull_progress(self) -> dict | None:
        """What the pull bar shows: label, status, bytes done/total, rate, eta."""
        state = self.pull
        if not state:
            return None
        with self._lock:
            layers = list(state["layers"].values())
            rate, status = state["rate"], state["status"]
        done = sum(c for c, _ in layers)
        total = sum(t for _, t in layers)
        eta = (total - done) / rate if rate and total > done else None
        if status.startswith("pulling ") and len(status) > 20:
            status = status[:20] + "…"        # "pulling <sha256 digest>"
        return {"label": self.label((state["ep"], state["name"])), "status": status,
                "done": done, "total": total, "rate": rate, "eta": eta}

    # ── actions ──────────────────────────────────────────────────────────────

    _VERBS = {"Stopping": ("Stopped", "Stop"), "Loading": ("Loaded", "Load"),
              "Extending": ("Extended", "Keep-alive of")}

    def _start(self, verb: str, target: tuple[int, str], extra: dict | None, timeout: int,
               done: str | None = None, request: tuple[str, str] | None = None) -> None:
        """`extra` goes into Ollama's /api/generate body; `request` is a
        (method, path) with no body instead — llama-swap's endpoints."""
        if self.busy:
            return
        self.busy = (verb, target[0], target[1], time.monotonic())
        threading.Thread(target=self._run,
                         args=(verb, target, extra, timeout, done, request),
                         daemon=True, name="mtop-action").start()

    def _run(self, verb: str, target: tuple[int, str], extra: dict | None, timeout: int,
             done: str | None = None, request: tuple[str, str] | None = None) -> None:
        i, name = target
        if request is not None:
            ok, data = self.endpoints[i].request(*request, timeout=timeout)
        else:
            ok, data = self.endpoints[i].post_json(
                "/api/generate", {"model": name, **(extra or {})}, timeout=timeout)
        past, what = self._VERBS[verb]
        text = ((done or f"{past} {self.label(target)}") if ok
                else f"{what} {name} failed: {data}")
        with self._lock:
            self._result = (text, ok, time.monotonic() + NOTICE_HOLD)
            self.busy = None

    def label(self, target: tuple[int, str]) -> str:
        i, name = target
        return name if len(self.endpoints) == 1 else f"{name} on {self.endpoints[i].label}"

    def notice(self, now: float) -> tuple[str, str] | None:
        """Footer override: the picker keys, the y/n question, the running
        action, or its result — (text, kind), kind one of NOTICE_KINDS."""
        if self.pull_prompt:
            pp = self.pull_prompt
            where = f" on {self.endpoints[pp['ep']].label}" if len(self.endpoints) > 1 else ""
            tab = " │ Tab: next endpoint" if len(self.endpoints) > 1 else ""
            return (f"Pull model{where}: {pp['text']}▏ │ Enter: pull │ Esc: cancel{tab}",
                    "ask")
        if self.picker:
            tab = " │ Tab: next endpoint" if len(self.endpoints) > 1 else ""
            full = self.evictions(self.picker["ep"])
            if full:
                return (f"↑/↓ PgUp/PgDn: choose │ Enter: load │ Esc: cancel{tab} │ "
                        f"{len(full[0])}/{full[1]} loaded: a new model evicts one",
                        "warn")
            return (f"↑/↓ PgUp/PgDn: choose │ Enter: load │ Esc: cancel{tab}",
                    "info")
        if self.load_confirm:
            ep, name, loaded, limit, source = self.load_confirm
            # The question first: long model names get cut at the right edge.
            return (f"Load {self.label((ep, name))}? [y/N] — {len(loaded)}/{limit} loaded "
                    f"({source}), Ollama will unload one of: {', '.join(loaded)}",
                    "ask")
        if self.swap_confirm:
            ep, name, others = self.swap_confirm
            return (f"Load {self.label((ep, name))}? [y/N] — llama-swap will stop "
                    f"{', '.join(others)} unless its config groups them",
                    "ask")
        if self.ttl_target:
            keys = " │ ".join(f"{n}: {label}" for n, (label, _) in enumerate(TTL_CHOICES, 1))
            return (f"Keep {self.label(self.ttl_target)} loaded for — {keys} │ other key: cancel",
                    "ask")
        if self.confirm:
            return (f"Stop {self.label(self.confirm)}? [y/N]",
                    "ask")
        busy = self.busy
        if busy:
            verb, i, name, started = busy
            return (f"{verb} {self.label((i, name))}… {now - started:.0f}s",
                    "info")
        with self._lock:
            result = self._result
        if result and now < result[2]:
            return result[0], "ok" if result[1] else "err"
        return None


def _http_error_body(e: urllib.error.HTTPError) -> str:
    """Ollama puts the reason in a JSON body: `{"error": "pull model manifest:
    file does not exist"}` says more than `HTTP 500`."""
    try:
        msg = json.loads(e.read().decode(errors="replace")).get("error")
    except Exception:
        msg = None
    return msg or http_error_text(e)

