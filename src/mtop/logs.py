"""
mtop.logs — tail the Ollama server log and turn its request lines into numbers.

Ollama has no metrics endpoint. What it does have is a GIN access log on
stderr, one line per API request with status, latency, client and path:

    [GIN] 2026/09/13 - 20:44:07 | 200 |  1.234567s |  127.0.0.1 | POST  "/api/generate"

Tailing that is the only universal source of request rate and error rate —
it works for a container (Engine API / CLI logs), a systemd unit
(journalctl) and needs nothing from Ollama itself. Tokens/s is *not* in the
log; that is per-response data the API returns to the caller only.

llama-swap logs its own access line, in a different shape and with no time:

    [INFO] Request 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200 1234 "curl/8.5.0" 1.2s

A unit may also send its output to a file (`StandardOutput=append:/path`)
instead of the journal; FileLogs follows that file.

Line timestamps come from the transport (docker --timestamps, journalctl
short-iso), not from the GIN field: the transport stamp is RFC3339 with a
zone, the GIN one is the server's local wall clock with no zone.

Stdlib only; no imports from the package (bundler embeds this as a module).
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone

Runner = Callable[..., tuple[bool, str]]

_GIN_RE = re.compile(
    r"\[GIN\]\s+\S+\s+-\s+\S+\s+\|\s*(?P<status>\d{3})\s*\|"
    r"\s*(?P<latency>[^|]+?)\s*\|"
    r"\s*(?P<client>[^|]+?)\s*\|\s*(?P<method>[A-Z]+)\s+\"(?P<path>[^\"]*)\"")
_SWAP_RE = re.compile(
    r"\[[A-Z]+\]\s+Request\s+(?P<client>\S+)\s+\"(?P<method>[A-Z]+)\s+(?P<path>\S+)"
    r"\s+HTTP/[\d.]+\"\s+(?P<status>\d{3})\s+\d+\s+\"[^\"]*\"\s+(?P<latency>\S+)")
_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)(µs|us|ns|ms|s|m|h)")
_DUR_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}
_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+")
_LEVEL_RE = re.compile(r"\b(level=|\[)?(ERROR|WARN(?:ING)?|FATAL|PANIC)\b", re.IGNORECASE)


def parse_go_duration(text: str) -> float | None:
    """'1.234567s', '12.5ms', '1m2.3s' -> seconds."""
    total = 0.0
    found = False
    for num, unit in _DUR_RE.findall(text.strip()):
        total += float(num) * _DUR_UNITS[unit]
        found = True
    return total if found else None


def parse_timestamp(text: str) -> float | None:
    """RFC3339 / journal short-iso prefix -> unix seconds, or None."""
    s = text.strip().replace(",", ".")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # older journalctl short-iso: '+0100' -> '+01:00'
    m = re.search(r"([+-]\d{2})(\d{2})$", s)
    if m and ":" not in s[m.start():]:
        s = s[:m.start()] + m.group(1) + ":" + m.group(2)
    if "." in s:
        dot = s.index(".")
        end = dot + 1
        while end < len(s) and s[end].isdigit():
            end += 1
        s = s[:dot + 1] + s[dot + 1:end][:6].ljust(6, "0") + s[end:]
    try:
        t = datetime.fromisoformat(s.replace(" ", "T", 1))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


def split_line(raw: str) -> tuple[float | None, str]:
    """Strip a leading transport timestamp; return (unix_ts_or_None, text)."""
    m = _TS_RE.match(raw)
    if not m:
        return None, raw.rstrip()
    return parse_timestamp(m.group(1)), raw[m.end():].rstrip()


def parse_gin(text: str) -> dict | None:
    """An access-log line — Ollama's GIN or llama-swap's `Request` — as a dict."""
    m = _GIN_RE.search(text) or _SWAP_RE.search(text)
    if not m:
        return None
    return {
        "status": int(m.group("status")),
        "latency": parse_go_duration(m.group("latency")),
        "client": m.group("client"),
        "method": m.group("method"),
        "path": m.group("path"),
    }


def line_level(text: str) -> str:
    """'error' / 'warn' / 'info' for coloring."""
    g = parse_gin(text)
    if g is not None:
        return "error" if g["status"] >= 500 else "warn" if g["status"] >= 400 else "info"
    m = _LEVEL_RE.search(text)
    if not m:
        return "info"
    lvl = m.group(2).upper()
    return "error" if lvl in ("ERROR", "FATAL", "PANIC") else "warn"


# Paths that monitors (mtop itself, health checks, model listers) hit; they
# would swamp the inference traffic that the stats are meant to show.
MONITOR_PATHS = frozenset({"/", "/api/ps", "/api/version", "/api/tags",
                           "/running", "/v1/models", "/health"})


def request_stats(lines: list[tuple[float | None, str]], window: float = 60.0,
                  now: float | None = None,
                  ignore_paths: frozenset[str] = MONITOR_PATHS) -> dict:
    """Request counts and latency over the last `window` seconds of GIN lines.

    Monitoring paths (see MONITOR_PATHS) are counted separately under
    'monitor' so mtop's own /api/ps polling does not read as traffic. Lines
    without a transport timestamp cannot be placed in the window and are
    counted under 'untimed' instead of being guessed at.
    """
    now = time.time() if now is None else now
    total = 0
    monitor = 0
    by_class: dict[str, int] = {}
    by_path: dict[str, int] = {}
    latencies: list[float] = []
    untimed = 0
    last: dict | None = None
    for ts, text in lines:
        g = parse_gin(text)
        if g is None:
            continue
        if ts is None:
            untimed += 1
            continue
        if now - ts > window:
            continue
        if g["path"].split("?", 1)[0] in ignore_paths:
            monitor += 1
            continue
        last = {**g, "ts": ts}
        total += 1
        cls = f"{g['status'] // 100}xx"
        by_class[cls] = by_class.get(cls, 0) + 1
        by_path[g["path"]] = by_path.get(g["path"], 0) + 1
        if g["latency"] is not None:
            latencies.append(g["latency"])
    latencies.sort()
    out = {"window_sec": window, "total": total, "by_status": by_class,
           "by_path": dict(sorted(by_path.items(), key=lambda kv: -kv[1])[:5]),
           "monitor": monitor, "untimed": untimed, "last": last}
    if latencies:
        out["latency_p50"] = latencies[len(latencies) // 2]
        out["latency_max"] = latencies[-1]
    return out


# ── sources ───────────────────────────────────────────────────────────────────

class LogSource:
    name = "base"

    def tail(self, n: int) -> tuple[bool, list[tuple[float | None, str]] | str]:
        """(True, [(ts, text), ...]) or (False, 'why not')."""
        raise NotImplementedError


class ContainerLogs(LogSource):
    """Whatever the container runtime gives (Engine API or CLI, both stamp lines)."""

    def __init__(self, runtime, container: str):
        self._rt = runtime
        self.container = container
        self.name = f"{runtime.name} logs"

    def tail(self, n: int):
        ok, out = self._rt.logs(self.container, n)
        if not ok:
            return False, out
        return True, [split_line(ln) for ln in out.splitlines() if ln.strip()]


class JournalLogs(LogSource):
    """journalctl for the systemd unit. Needs the systemd-journal/adm group."""

    name = "journalctl"

    def __init__(self, runner: Runner, unit: str = "ollama.service", user: bool = False):
        self._run = runner
        self.unit = unit
        self.user = user   # llama-swap is a `systemctl --user` unit, not a system one

    def tail(self, n: int):
        unit_flag = ["--user", "--user-unit", self.unit] if self.user else ["-u", self.unit]
        ok, out = self._run(["journalctl", *unit_flag, "-n", str(n), "-o", "short-iso",
                             "--no-pager", "-q"], timeout=4)
        if not ok:
            return False, out or "journalctl failed"
        lines = []
        for ln in out.splitlines():
            if not ln.strip() or ln.startswith("-- "):
                continue
            ts, rest = split_line(ln)
            # short-iso: '<ts> <host> <unit>[pid]: <message>'
            rest = re.sub(r"^\S+\s+\S+?\[\d+\]:\s?", "", rest, count=1)
            lines.append((ts, rest))
        return True, lines


class FileLogs(LogSource):
    """Follow a log file (a unit's `StandardOutput=append:/path`).

    The first read takes the last lines from the end of the file — no need
    to read 100 MB of history — and they stay untimed: llama-swap writes no
    time, and guessing one would put old requests into the rate window.
    After that, each read picks up what was appended since and stamps it
    with the time mtop saw it, so the request rate works from then on.
    Truncation or rotation (smaller file, new inode) starts over at the end.
    """

    INITIAL_BYTES = 64 * 1024
    MAX_READ = 1024 * 1024     # per call; a larger burst keeps only its newest part

    def __init__(self, path: str, keep: int = 500):
        self.path = path
        self.name = f"file {os.path.basename(path)}"
        self._lines: deque[tuple[float | None, str]] = deque(maxlen=keep)
        self._offset: int | None = None
        self._inode: int | None = None
        self._partial = ""

    def tail(self, n: int):
        try:
            with open(self.path, "rb") as f:
                st = os.fstat(f.fileno())
                fresh = (self._offset is None or st.st_ino != self._inode
                         or st.st_size < self._offset)
                if fresh:
                    self._lines.clear()
                    self._partial = ""
                    start, stamp = max(0, st.st_size - self.INITIAL_BYTES), None
                    gap = start > 0
                else:
                    start, stamp = max(self._offset, st.st_size - self.MAX_READ), time.time()
                    gap = start > self._offset
                if gap:
                    # One byte earlier, so the cut below keeps the first line
                    # when `start` happens to sit right after a newline.
                    start -= 1
                f.seek(start)
                chunk = f.read(st.st_size - start).decode(errors="replace")
                self._offset, self._inode = st.st_size, st.st_ino
        except OSError as e:
            return False, f"{self.path}: {e.strerror or e}"
        if gap:
            # Started mid-file: the first line is cut, and a pending partial
            # line no longer continues into this chunk.
            chunk = chunk.split("\n", 1)[1] if "\n" in chunk else ""
            self._partial = ""
        *complete, self._partial = (self._partial + chunk).split("\n")
        for ln in complete:
            if ln.strip():
                ts, text = split_line(ln)
                self._lines.append((ts if ts is not None else stamp, text))
        return True, list(self._lines)[-n:]
