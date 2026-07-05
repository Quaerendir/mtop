#!/usr/bin/env python3
"""
mtop — Ollama model monitor for Docker containers.
curses-based TUI with zero flicker, color-coded status, GPU/container stats.

Architecture (v0.2.0): a background collector thread gathers all data
(docker inspect/stats, nvidia-smi, Ollama API) and publishes immutable
snapshots; the curses loop only draws the latest snapshot and handles
keys at a fixed 100 ms poll. Slow or hung data sources can no longer
freeze the UI — stale data is flagged instead.

Usage:
    mtop [-c CONTAINER] [-i INTERVAL] [-u URL] [--no-gpu] [--no-docker]
         [--json] [-h]
"""

import argparse
import curses
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

__version__ = "0.2.0"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONTAINER = "ollama"
DEFAULT_INTERVAL = 1.0
DEFAULT_API_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

UI_POLL_MS = 100          # curses getch timeout — UI responsiveness, not data rate
SLOW_FLOOR = 2.0          # minimum cadence for docker stats / nvidia-smi
STALE_FACTOR = 3.0        # snapshot older than interval*factor => flagged stale

# ── Color pairs (initialized in curses_main) ─────────────────────────────────

C_HEADER = 1
C_OK = 2
C_WARN = 3
C_ERR = 4
C_DIM = 5
C_ACCENT = 6
C_TABLE_HDR = 7
C_GPU = 8


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(C_OK, curses.COLOR_GREEN, -1)
    curses.init_pair(C_WARN, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_ERR, curses.COLOR_RED, -1)
    curses.init_pair(C_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(C_ACCENT, curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_TABLE_HDR, curses.COLOR_WHITE, -1)
    curses.init_pair(C_GPU, curses.COLOR_GREEN, -1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_api_url(url: str) -> str:
    """Accept Ollama-style OLLAMA_HOST values without a scheme.

    Ollama itself treats ``OLLAMA_HOST=0.0.0.0:11434`` or ``gpu-rig:11434``
    as valid; urllib does not. Prepend http:// when no scheme is present.
    """
    url = url.strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    return url


def run_cmd(cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
    """Run a command, return (success, stdout_or_stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, r.stderr.strip() or r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, f"command not found: {cmd[0]}"
    except Exception as e:
        return False, str(e)


def http_get_json(url: str, timeout: int = 5) -> tuple[bool, Any]:
    """GET JSON from URL, return (success, data_or_error_string)."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def bytes_to_gib(b: int | float) -> str:
    return f"{b / (1024**3):.2f}"


def to_float(s: Any) -> float | None:
    """Parse a numeric field that may be '[N/A]', 'N/A', '' or garbage."""
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


def relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to relative future/past string."""
    if not iso_str:
        return "—"
    try:
        # Handle various ISO formats from Ollama
        iso_str = iso_str.replace("Z", "+00:00")
        if "." in iso_str:
            # Truncate nanoseconds to 6 digits for fromisoformat
            dot_pos = iso_str.index(".")
            plus_pos = iso_str.find("+", dot_pos)
            if plus_pos == -1:
                plus_pos = iso_str.find("-", dot_pos + 1)
            if plus_pos != -1:
                frac = iso_str[dot_pos + 1 : plus_pos][:6]
                iso_str = iso_str[: dot_pos + 1] + frac + iso_str[plus_pos:]
        target = datetime.fromisoformat(iso_str)
        now = datetime.now(timezone.utc)
        delta = target - now
        total_sec = int(delta.total_seconds())
        suffix = " left" if total_sec >= 0 else " ago"
        total_sec = abs(total_sec)
        if total_sec < 60:
            return f"{total_sec}s{suffix}"
        elif total_sec < 3600:
            m, s = divmod(total_sec, 60)
            return f"{m}m {s}s{suffix}"
        elif total_sec < 86400:
            h, rem = divmod(total_sec, 3600)
            m = rem // 60
            return f"{h}h {m}m{suffix}"
        else:
            d = total_sec // 86400
            return f"{d}d{suffix}"
    except Exception:
        return iso_str[:19]


def read_unified_memory() -> tuple[str, int, int] | None:
    """Detect Tegra/Jetson/Spark unified-memory platforms via device-tree.

    Returns (model_name, used_mib, total_mib) or None.
    """
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().strip().replace("\x00", "")
        if not any(k in model.lower() for k in ("jetson", "tegra", "spark", "orin")):
            return None
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        used_kb = total_kb - avail_kb
        return model, used_kb // 1024, total_kb // 1024
    except (FileNotFoundError, PermissionError, KeyError, ValueError):
        return None


# ── Collector (background thread) ─────────────────────────────────────────────

class Collector(threading.Thread):
    """Gathers container/GPU/API data in the background.

    Publishes point-in-time snapshot dicts; the UI thread reads the latest
    one under a lock. All blocking I/O (subprocess, HTTP with up to 5 s
    timeouts) lives here so the curses loop stays responsive.

    ``interval`` and ``show_raw_ps`` are mutated from the UI thread; both
    are single-reference reads/writes so the GIL makes them safe without
    additional locking.
    """

    def __init__(self, container: str, api_url: str, interval: float,
                 show_gpu: bool, use_docker: bool, show_raw_ps: bool = False):
        super().__init__(daemon=True, name="mtop-collector")
        self.container = container
        self.api_url = api_url
        self.interval = interval
        self.show_gpu = show_gpu
        self.use_docker = use_docker
        self.show_raw_ps = show_raw_ps

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot: dict = {"ts": 0.0, "status": "starting", "uptime": ""}

        # Slow-path caches (docker stats / nvidia-smi), refreshed at
        # max(SLOW_FLOOR, self.interval) — recomputed each cycle so
        # runtime +/- interval changes take effect (fixes v0.1.0 bug
        # where the cap was frozen at startup).
        self._slow_ts: float = 0.0
        self._docker_stats: dict | None = None
        self._gpu_cache: list[dict] | None = None

        # Memoized GPU probe strategy: "host" | "container" | "unified" | None.
        # Avoids re-forking failed nvidia-smi probes every cycle on boxes
        # without a GPU; reset on failure so hotplug/driver restarts recover.
        self._gpu_mode: str | None = None

    # -- lifecycle -------------------------------------------------------------

    def run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            snap = self.collect(t0)
            with self._lock:
                self._snapshot = snap
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.05, self.interval - elapsed))

    def stop(self):
        self._stop.set()

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot

    # -- collection ------------------------------------------------------------

    def collect(self, now: float) -> dict:
        """One full collection pass. Also used synchronously by --json."""
        snap: dict = {
            "ts": time.monotonic(),
            "wallclock": datetime.now().astimezone().isoformat(timespec="seconds"),
            "container": self.container,
            "api_url": self.api_url,
        }

        if self.use_docker:
            status, uptime, cpu_limit = self._inspect_container()
            snap.update(status=status, uptime=uptime, cpu_limit=cpu_limit)
            if status != "running":
                return snap
        else:
            snap.update(status="api-only", uptime="", cpu_limit=None)

        slow_due = now - self._slow_ts >= max(SLOW_FLOOR, self.interval)
        if slow_due:
            if self.use_docker:
                self._docker_stats = self._docker_stats_read()
            if self.show_gpu:
                self._gpu_cache = self._gpu_read()
            self._slow_ts = now

        snap["docker_stats"] = self._docker_stats if self.use_docker else None
        snap["gpus"] = self._gpu_cache if self.show_gpu else None

        ok, data = http_get_json(f"{self.api_url}/api/ps")
        snap["models_ok"] = ok
        snap["models"] = data.get("models", []) if ok else []
        snap["models_err"] = "" if ok else str(data)

        if self.use_docker and self.show_raw_ps:
            ok2, out2 = run_cmd(["docker", "exec", self.container, "ollama", "ps"])
            snap["raw_ps_ok"] = ok2
            snap["raw_ps"] = out2

        return snap

    def _inspect_container(self) -> tuple[str, str, float | None]:
        """(status, uptime, effective_cpu_limit) in a single inspect call.

        CPU limit comes from HostConfig (NanoCpus for --cpus, quota/period
        for --cpu-quota); falls back to host core count. This makes the CPU
        bar normalize against what the container can actually use instead
        of the host total (a --cpus=4 container saturating on a 32-core
        host previously showed 12.5%).
        """
        fmt = ("{{.State.Status}}\t{{.State.StartedAt}}\t"
               "{{.HostConfig.NanoCpus}}\t{{.HostConfig.CpuQuota}}\t"
               "{{.HostConfig.CpuPeriod}}")
        ok, out = run_cmd(["docker", "inspect", "--format", fmt, self.container])
        if not ok:
            return "not found", "", None
        parts = out.split("\t")
        status = parts[0].strip()
        uptime = ""
        if status == "running" and len(parts) > 1 and parts[1].strip():
            uptime = relative_time(parts[1].strip()).replace(" ago", "")

        cpu_limit: float | None = None
        try:
            nano = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 0
            quota = int(parts[3]) if len(parts) > 3 and parts[3].strip() else 0
            period = int(parts[4]) if len(parts) > 4 and parts[4].strip() else 100000
            if nano > 0:
                cpu_limit = nano / 1e9
            elif quota > 0 and period > 0:
                cpu_limit = quota / period
        except (ValueError, IndexError):
            pass
        if not cpu_limit or cpu_limit <= 0:
            cpu_limit = float(os.cpu_count() or 1)
        return status, uptime, cpu_limit

    def _docker_stats_read(self) -> dict | None:
        fmt = "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        ok, out = run_cmd(
            ["docker", "stats", "--no-stream", "--format", fmt, self.container]
        )
        if not ok or not out:
            return None
        parts = out.split("\t")
        if len(parts) < 3:
            return None
        return {
            "cpu": parts[0].strip(),
            "mem_usage": parts[1].strip(),
            "mem_pct": parts[2].strip(),
        }

    # -- GPU -------------------------------------------------------------------

    def _nvidia_smi(self, prefix: list[str]) -> list[dict] | None:
        query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
        cmd = prefix + [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        ok, out = run_cmd(cmd, timeout=3)
        if not ok or not out:
            return None
        gpus = []
        for line in out.strip().split("\n"):
            fields = [f.strip() for f in line.split(",")]
            if len(fields) >= 5:
                gpus.append({
                    "name": fields[0],
                    "util": fields[1],
                    "mem_used": fields[2],
                    "mem_total": fields[3],
                    "temp": fields[4],
                })
        return gpus or None

    def _gpu_read(self) -> list[dict] | None:
        """Probe GPU stats, memoizing which strategy worked.

        On unified-memory platforms (GB10 Spark, Jetson, Orin under recent
        DGX OS / JetPack) nvidia-smi is present but reports memory fields
        as '[N/A]'. v0.1.0 accepted that output verbatim and the VRAM bar
        silently vanished; now incomplete memory data gets patched from
        /proc/meminfo while util/temp (when numeric) are kept.
        """
        host_prefix: list[str] = []
        cont_prefix = ["docker", "exec", self.container] if self.use_docker else None

        if self._gpu_mode == "host":
            candidates = [("host", host_prefix)]
        elif self._gpu_mode == "container" and cont_prefix:
            candidates = [("container", cont_prefix)]
        elif self._gpu_mode == "unified":
            candidates = []
        else:
            candidates = [("host", host_prefix)]
            if cont_prefix:
                candidates.append(("container", cont_prefix))

        for mode, prefix in candidates:
            gpus = self._nvidia_smi(prefix)
            if gpus:
                self._gpu_mode = mode
                return self._patch_unified(gpus)

        uni = read_unified_memory()
        if uni:
            self._gpu_mode = "unified"
            model, used_mib, total_mib = uni
            return [{
                "name": model,
                "util": "N/A",
                "mem_used": str(used_mib),
                "mem_total": str(total_mib),
                "temp": "N/A",
            }]

        # Nothing worked — forget the memo so a driver restart / container
        # start is picked up on the next slow cycle.
        self._gpu_mode = None
        return None

    def _patch_unified(self, gpus: list[dict]) -> list[dict]:
        """Fill '[N/A]' memory fields from /proc/meminfo on unified platforms."""
        if len(gpus) == 1 and to_float(gpus[0]["mem_total"]) is None:
            uni = read_unified_memory()
            if uni:
                _, used_mib, total_mib = uni
                gpus[0]["mem_used"] = str(used_mib)
                gpus[0]["mem_total"] = str(total_mib)
                gpus[0]["unified"] = True
        return gpus


# ── Curses drawing helpers ────────────────────────────────────────────────────

def safe_addstr(win, y: int, x: int, text: str, attr=0) -> int:
    """Write string to window, clipping to window bounds. Returns next y.

    Guards both axes: rows past the usable area stop vertical flow (return y),
    while an x beyond the right edge skips the draw but still advances the row
    so the surrounding layout stays intact. Clipping uses a real available-width
    computation rather than a slice that can go negative when x >= max_x.
    """
    max_y, max_x = win.getmaxyx()
    if y >= max_y - 1:
        return y
    if x < 0 or x >= max_x - 1:
        return y + 1
    avail = max_x - x - 1
    if len(text) > avail:
        text = text[:avail]
    if text:
        try:
            win.addstr(y, x, text, attr)
        except curses.error:
            pass
    return y + 1


def draw_detail_right(win, y: int, min_x: int, text: str, attr=0):
    """Right-align a detail string to the frame, but never left of min_x.

    Replaces the v0.1.0 hardcoded x=60/62 detail columns that overlapped
    the bars (or vanished) on terminals narrower than ~80 cols.
    """
    _, max_x = win.getmaxyx()
    x = max_x - len(text) - 2
    if x <= min_x:
        return  # not enough room — drop the detail, keep the bar intact
    safe_addstr(win, y, x, text, attr)


def draw_table(win, y: int, x: int, headers: list[str], rows: list[list[str]],
               col_widths: list[int], hdr_attr=0, row_attr=0) -> int:
    """Draw a formatted table. Returns next y position."""
    max_y, _ = win.getmaxyx()

    # Header
    line = ""
    for i, h in enumerate(headers):
        line += h.ljust(col_widths[i]) if i < len(col_widths) else h
        if i < len(headers) - 1:
            line += "  "
    y = safe_addstr(win, y, x, line, hdr_attr)

    # Separator
    sep = ""
    for i, w in enumerate(col_widths):
        sep += "─" * w
        if i < len(col_widths) - 1:
            sep += "──"
    y = safe_addstr(win, y, x, sep, curses.color_pair(C_DIM))

    # Rows
    for row in rows:
        if y >= max_y - 1:
            break
        line = ""
        for i, cell in enumerate(row):
            w = col_widths[i] if i < len(col_widths) else len(cell)
            if len(cell) > w:
                cell = cell[: w - 1] + "…"
            line += cell.ljust(w) if i < len(row) - 1 else cell
            if i < len(row) - 1:
                line += "  "
        y = safe_addstr(win, y, x, line, row_attr)
    return y


def draw_bar(win, y: int, x: int, label: str, value: float, width: int = 20,
             color_pair: int = C_OK) -> int:
    """Draw a progress bar: [████░░░░░░] 45%"""
    max_y, max_x = win.getmaxyx()
    if y >= max_y - 1:
        return y
    filled = int(value / 100.0 * width)
    filled = max(0, min(filled, width))
    bar = "█" * filled + "░" * (width - filled)
    pct_str = f" {value:5.1f}%"

    safe_addstr(win, y, x, label, curses.color_pair(C_DIM))
    lbl_end = x + len(label)
    safe_addstr(win, y, lbl_end, "[", curses.color_pair(C_DIM))

    # Color based on value
    if value > 90:
        bar_color = curses.color_pair(C_ERR) | curses.A_BOLD
    elif value > 70:
        bar_color = curses.color_pair(C_WARN)
    else:
        bar_color = curses.color_pair(color_pair)

    safe_addstr(win, y, lbl_end + 1, bar, bar_color)
    safe_addstr(win, y, lbl_end + 1 + width, "]", curses.color_pair(C_DIM))
    safe_addstr(win, y, lbl_end + 2 + width, pct_str, curses.color_pair(C_DIM))
    return y + 1


def bar_end_x(x: int, label: str, width: int) -> int:
    """Rightmost column a draw_bar() occupies (bracket + ' 100.0%')."""
    return x + len(label) + width + 2 + 7


def section_header(win, y: int, label: str) -> int:
    max_y, max_x = win.getmaxyx()
    w = max_x - 4
    pad = max(0, w - len(label))
    sep = "─" * (pad // 2) + label + "─" * (pad - pad // 2)
    return safe_addstr(win, y, 2, sep, curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)


# ── Main sections ─────────────────────────────────────────────────────────────

def render_header(win, y: int, snap: dict, stale: bool) -> int:
    """Draw the top banner with a full-width frame.

    All geometry is anchored to ``inner_right = max_x - 2`` — the rightmost
    column ``addstr`` can write without raising (the very last cell, max_x-1,
    is unwritable via addstr). Top/bottom borders and the right ``║`` all land
    on that column so the box stays square at any width, and the timestamp is
    right-aligned *to the frame* instead of a hardcoded floor.
    """
    max_y, max_x = win.getmaxyx()
    if max_x < 16:                       # too narrow to frame anything sane
        return y
    inner_right = max_x - 2              # column of ╗ ╝ and the right ║
    fill = inner_right - 2              # ═ count between the corners
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname = os.uname().nodename
    status = snap.get("status", "?")
    uptime = snap.get("uptime", "")
    container = snap.get("container", "?")

    # Top border: ╔═══ mtop v0.2.0 — Ollama Model Monitor ═══╗
    title = f" mtop v{__version__} — Ollama Model Monitor "
    pad_total = max(0, fill - len(title))
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    top_line = "╔" + "═" * pad_left + title + "═" * pad_right + "╗"
    y = safe_addstr(win, y, 1, top_line, curses.color_pair(C_HEADER) | curses.A_BOLD)

    # Status line: ║ host: ... container: ... up: ...            time ║
    if status == "running":
        status_icon = "● "
        status_attr = curses.color_pair(C_OK) | curses.A_BOLD
    elif status in ("not found", "starting"):
        status_icon = "✗ " if status == "not found" else "… "
        status_attr = curses.color_pair(C_ERR) | curses.A_BOLD
    elif status == "api-only":
        status_icon = "◌ "
        status_attr = curses.color_pair(C_DIM)
    else:
        status_icon = "○ "
        status_attr = curses.color_pair(C_WARN)

    # Both vertical borders on the same column the corners use.
    safe_addstr(win, y, 1, "║", curses.color_pair(C_HEADER))
    safe_addstr(win, y, inner_right, "║", curses.color_pair(C_HEADER))

    col2 = 32
    col3 = 64
    safe_addstr(win, y, 3, "host: ", curses.color_pair(C_DIM))
    # Clamp hostname so it can never bleed into the container field.
    host_room = max(1, col2 - 9 - 1)
    host_show = hostname if len(hostname) <= host_room else hostname[: host_room - 1] + "…"
    safe_addstr(win, y, 9, host_show, curses.color_pair(C_ACCENT))
    if status == "api-only":
        safe_addstr(win, y, col2, "api: ", curses.color_pair(C_DIM))
        safe_addstr(win, y, col2 + 5, status_icon + snap.get("api_url", ""), status_attr)
    else:
        safe_addstr(win, y, col2, "container: ", curses.color_pair(C_DIM))
        safe_addstr(win, y, col2 + 11, status_icon + container, status_attr)
        if uptime:
            safe_addstr(win, y, col3, f"up: {uptime}", curses.color_pair(C_DIM))

    # Right-align timestamp (or STALE flag) to the inner frame.
    right_str = f"STALE {now}" if stale else now
    right_attr = (curses.color_pair(C_ERR) | curses.A_BOLD) if stale \
        else curses.color_pair(C_DIM)
    time_x = inner_right - len(right_str) - 1
    uptime_end = col3 + (len(f"up: {uptime}") if uptime else 0)
    if time_x > uptime_end + 1:
        safe_addstr(win, y, time_x, right_str, right_attr)
    y += 1

    # Bottom border: same length as the top so corners align on inner_right.
    bottom_line = "╚" + "═" * fill + "╝"
    y = safe_addstr(win, y, 1, bottom_line, curses.color_pair(C_HEADER))
    return y


def render_docker_stats(win, y: int, snap: dict) -> int:
    """Show container CPU/MEM with progress bars from the snapshot."""
    stats = snap.get("docker_stats")
    if not stats:
        return y

    y = section_header(win, y, " CONTAINER RESOURCES ")

    # Docker reports CPUPerc per-core (500% = 5 cores); normalize against
    # the container's effective limit (--cpus / quota) or host core count.
    cpu_raw = to_float(str(stats["cpu"]).rstrip("%")) or 0.0
    ncpu = snap.get("cpu_limit") or float(os.cpu_count() or 1)
    cpu_normalized = min(cpu_raw / ncpu, 100.0)
    if ncpu == int(ncpu):
        cpu_detail = f"{cpu_raw:.0f}% / {int(ncpu)} cores"
    else:
        cpu_detail = f"{cpu_raw:.0f}% / {ncpu:.1f} cores"
    y = draw_bar(win, y, 3, "CPU  ", cpu_normalized, 30, C_ACCENT)
    draw_detail_right(win, y - 1, bar_end_x(3, "CPU  ", 30), cpu_detail,
                      curses.color_pair(C_DIM))

    mem_val = to_float(str(stats["mem_pct"]).rstrip("%")) or 0.0
    y = draw_bar(win, y, 3, "MEM  ", mem_val, 30, C_OK)
    draw_detail_right(win, y - 1, bar_end_x(3, "MEM  ", 30), stats["mem_usage"],
                      curses.color_pair(C_DIM))

    y += 1
    return y


def render_gpu_stats(win, y: int, snap: dict) -> int:
    """Render GPU info section."""
    gpus = snap.get("gpus")
    y = section_header(win, y, " GPU ")

    if gpus is None:
        y = safe_addstr(win, y, 3, "GPU monitoring unavailable",
                        curses.color_pair(C_DIM) | curses.A_DIM)
        y += 1
        return y

    for i, gpu in enumerate(gpus):
        prefix = f"[{i}] {gpu['name']}"
        temp_val = to_float(gpu["temp"])
        temp_str = f"  {gpu['temp']}°C" if temp_val is not None else ""
        if gpu.get("unified"):
            temp_str += "  (unified memory)"
        y = safe_addstr(win, y, 3, prefix + temp_str, curses.color_pair(C_GPU))

        # GPU utilization bar
        util_val = to_float(gpu["util"])
        if util_val is not None:
            y = draw_bar(win, y, 5, "UTIL ", util_val, 25, C_GPU)

        # VRAM bar
        mem_used = to_float(gpu["mem_used"])
        mem_total = to_float(gpu["mem_total"])
        if mem_used is not None and mem_total and mem_total > 0:
            mem_pct = mem_used / mem_total * 100
            y = draw_bar(win, y, 5, "VRAM ", mem_pct, 25, C_GPU)
            vram_str = f"{mem_used:.0f} / {mem_total:.0f} MiB"
            draw_detail_right(win, y - 1, bar_end_x(5, "VRAM ", 25), vram_str,
                              curses.color_pair(C_DIM))

    y += 1
    return y


def render_models(win, y: int, snap: dict) -> int:
    """Display loaded models from the snapshot (/api/ps)."""
    y = section_header(win, y, " LOADED MODELS ")

    if not snap.get("models_ok", False):
        y = safe_addstr(win, y, 3, f"API error: {snap.get('models_err', '?')}",
                        curses.color_pair(C_ERR))
        y += 1
        return y

    models = snap.get("models", [])
    if not models:
        y = safe_addstr(win, y, 3, "No models currently loaded",
                        curses.color_pair(C_WARN) | curses.A_DIM)
        y += 1
        return y

    headers = ["MODEL", "VRAM", "RAM", "CTX", "PROCESSOR", "EXPIRES"]
    col_widths = [36, 10, 10, 8, 14, 14]
    rows = []
    for m in models:
        name = m.get("name", "?")
        size_vram = m.get("size_vram", 0)
        size_total = m.get("size", 0)
        size_ram = max(0, size_total - size_vram)
        ctx = str(m.get("context_length", 0))
        # Try to get processor info from details
        details = m.get("details", {})
        processor = "GPU" if size_ram == 0 else "CPU+GPU" if size_vram > 0 else "CPU"
        if isinstance(details, dict):
            processor = details.get("processor", processor)
        expires = relative_time(m.get("expires_at", ""))
        rows.append([
            name,
            bytes_to_gib(size_vram) + " G",
            bytes_to_gib(size_ram) + " G",
            ctx,
            processor,
            expires,
        ])

    y = draw_table(win, y, 3, headers, rows, col_widths,
                   hdr_attr=curses.color_pair(C_TABLE_HDR) | curses.A_BOLD,
                   row_attr=curses.color_pair(C_OK))
    y += 1
    return y


def render_ollama_ps(win, y: int, snap: dict) -> int:
    """Show raw ollama ps output from the snapshot (toggle: 'o')."""
    y = section_header(win, y, " OLLAMA PS ")

    if not snap.get("raw_ps_ok", False):
        y = safe_addstr(win, y, 3, f"ollama ps failed: {snap.get('raw_ps', '')[:80]}",
                        curses.color_pair(C_ERR))
        y += 1
        return y

    for line in snap.get("raw_ps", "").split("\n"):
        if not line.strip():
            continue
        # Header line in dim, data lines in normal
        attr = curses.color_pair(C_DIM) if line.startswith("NAME") else curses.color_pair(C_OK)
        y = safe_addstr(win, y, 3, line, attr)

    y += 1
    return y


def render_footer(win, interval: float, raw_ps: bool, use_docker: bool):
    max_y, max_x = win.getmaxyx()
    footer_y = max_y - 1
    parts = ["q: quit", f"+/-: interval ({interval:.1f}s)"]
    if use_docker:
        parts.append(f"o: raw ps [{'on' if raw_ps else 'off'}]")
    parts.append(f"mtop v{__version__}")
    footer = " " + " │ ".join(parts) + " "
    footer = footer[: max_x - 1].ljust(max_x - 1)
    try:
        win.addstr(footer_y, 0, footer, curses.color_pair(C_DIM) | curses.A_REVERSE)
    except curses.error:
        pass


# ── Main loop ─────────────────────────────────────────────────────────────────

def curses_main(stdscr, args):
    init_colors()
    curses.curs_set(0)  # hide cursor
    stdscr.timeout(UI_POLL_MS)  # fixed fast poll — data cadence lives in Collector

    collector = Collector(
        container=args.container,
        api_url=args.api_url,
        interval=args.interval,
        show_gpu=not args.no_gpu,
        use_docker=not args.no_docker,
    )
    collector.start()

    try:
        while True:
            try:
                key = stdscr.getch()
                if key in (ord("q"), ord("Q"), 27):  # q, Q, ESC
                    break
                elif key == ord("+"):
                    collector.interval = max(0.5, collector.interval - 0.5)
                elif key == ord("-"):
                    collector.interval = min(30.0, collector.interval + 0.5)
                elif key == ord("o") and not args.no_docker:
                    collector.show_raw_ps = not collector.show_raw_ps
                elif key == curses.KEY_RESIZE:
                    stdscr.erase()
            except curses.error:
                pass

            snap = collector.snapshot()
            age = time.monotonic() - snap.get("ts", 0.0)
            stale = snap.get("ts", 0.0) > 0 and age > collector.interval * STALE_FACTOR

            stdscr.erase()
            y = render_header(stdscr, 0, snap, stale)

            status = snap.get("status", "starting")
            if status == "starting":
                y = safe_addstr(stdscr, y + 1, 3, "Collecting first snapshot…",
                                curses.color_pair(C_DIM))
            elif status not in ("running", "api-only"):
                y = safe_addstr(stdscr, y + 1, 3,
                                f"Container '{args.container}' is {status}. Waiting...",
                                curses.color_pair(C_ERR) | curses.A_BOLD)
                y = safe_addstr(stdscr, y + 1, 3,
                                "Will retry automatically.",
                                curses.color_pair(C_DIM))
            else:
                y = render_docker_stats(stdscr, y, snap)
                if not args.no_gpu:
                    y = render_gpu_stats(stdscr, y, snap)
                y = render_models(stdscr, y, snap)
                if collector.show_raw_ps and "raw_ps" in snap:
                    y = render_ollama_ps(stdscr, y, snap)

            render_footer(stdscr, collector.interval, collector.show_raw_ps,
                          not args.no_docker)
            stdscr.refresh()
    finally:
        collector.stop()


# ── One-shot JSON mode ────────────────────────────────────────────────────────

def json_main(args) -> int:
    """--json: run one collection pass, dump JSON to stdout.

    Exit code 0 when the container is running (or --no-docker) and the API
    answered; 1 otherwise. Suitable for cron, Prometheus textfile collectors
    (post-processed), or Ansible facts.
    """
    collector = Collector(
        container=args.container,
        api_url=args.api_url,
        interval=args.interval,
        show_gpu=not args.no_gpu,
        use_docker=not args.no_docker,
    )
    snap = collector.collect(time.monotonic())
    snap.pop("ts", None)  # monotonic value is meaningless outside the process
    print(json.dumps(snap, indent=2))
    healthy = snap.get("status") in ("running", "api-only") and snap.get("models_ok", False)
    return 0 if healthy else 1


def main():
    parser = argparse.ArgumentParser(
        description="mtop — Ollama model monitor for Docker containers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q=quit, +=faster, -=slower, o=toggle raw ollama ps\n\n"
               "https://github.com/Quaerendir/mtop",
    )
    parser.add_argument("-c", "--container", default=DEFAULT_CONTAINER,
                        help=f"Docker container name (default: {DEFAULT_CONTAINER})")
    parser.add_argument("-i", "--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("-u", "--api-url", default=DEFAULT_API_BASE,
                        help="Ollama API base URL (default: $OLLAMA_HOST or "
                             f"{DEFAULT_API_BASE})")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU stats section")
    parser.add_argument("--no-docker", action="store_true",
                        help="API-only mode: skip all docker calls "
                             "(for remote Ollama instances)")
    parser.add_argument("--json", action="store_true",
                        help="One-shot: print a single snapshot as JSON and exit "
                             "(exit code 1 on unhealthy)")
    parser.add_argument("-V", "--version", action="version",
                        version=f"mtop {__version__}")
    args = parser.parse_args()
    args.api_url = normalize_api_url(args.api_url)

    if args.json:
        sys.exit(json_main(args))

    try:
        curses.wrapper(curses_main, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
