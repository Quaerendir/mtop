#!/usr/bin/env python3
"""
mtop — Ollama model monitor for Docker containers.
curses-based TUI with zero flicker, color-coded status, GPU/container stats.

Usage:
    mtop [-c CONTAINER] [-i INTERVAL] [-u URL] [--no-gpu] [-h]
"""

import argparse
import curses
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Optional

__version__ = "0.1.0"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONTAINER = "ollama"
DEFAULT_INTERVAL = 1.0
DEFAULT_API_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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


def get_cpu_count() -> int:
    """Get number of CPU cores available."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def parse_docker_stats(container: str) -> Optional[dict]:
    """Get CPU%, MEM usage/limit from docker stats."""
    fmt = "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    ok, out = run_cmd(["docker", "stats", "--no-stream", "--format", fmt, container])
    if not ok or not out:
        return None
    parts = out.split("\t")
    if len(parts) < 3:
        return None
    return {"cpu": parts[0].strip(), "mem_usage": parts[1].strip(), "mem_pct": parts[2].strip()}


def get_container_status(container: str) -> tuple[str, str]:
    """Return (status, uptime) for a Docker container."""
    ok, status = run_cmd(
        ["docker", "inspect", "--format", "{{.State.Status}}", container]
    )
    if not ok:
        return "not found", ""
    uptime = ""
    if status == "running":
        ok2, started = run_cmd(
            ["docker", "inspect", "--format", "{{.State.StartedAt}}", container]
        )
        if ok2 and started:
            uptime = relative_time(started).replace(" ago", "")
    return status, uptime


def get_gpu_stats() -> Optional[list[dict]]:
    """Query nvidia-smi for GPU stats. Returns list of GPU dicts or None."""
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    # Try host first, then common container paths
    for cmd_prefix in [[], ["docker", "exec", "ollama"]]:
        cmd = cmd_prefix + [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        ok, out = run_cmd(cmd, timeout=3)
        if ok and out:
            gpus = []
            for line in out.strip().split("\n"):
                fields = [f.strip() for f in line.split(",")]
                if len(fields) >= 5:
                    gpus.append(
                        {
                            "name": fields[0],
                            "util": fields[1],
                            "mem_used": fields[2],
                            "mem_total": fields[3],
                            "temp": fields[4],
                        }
                    )
            if gpus:
                return gpus
    # Try tegra/jetson unified memory fallback
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip().replace("\x00", "")
        if any(k in model.lower() for k in ("jetson", "tegra", "spark", "orin")):
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        meminfo[key] = int(val)
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used_kb = total_kb - avail_kb
            return [
                {
                    "name": model,
                    "util": "N/A",
                    "mem_used": str(used_kb // 1024),
                    "mem_total": str(total_kb // 1024),
                    "temp": "N/A",
                }
            ]
    except (FileNotFoundError, PermissionError, KeyError):
        pass
    return None


# ── Curses drawing helpers ────────────────────────────────────────────────────

def safe_addstr(win, y: int, x: int, text: str, attr=0) -> int:
    """Write string to window, clipping to window bounds. Returns next y."""
    max_y, max_x = win.getmaxyx()
    if y >= max_y - 1:
        return y
    text = text[: max_x - x - 1] if x + len(text) >= max_x else text
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass
    return y + 1


def draw_hline(win, y: int, x: int, char: str, width: int, attr=0) -> int:
    max_y, max_x = win.getmaxyx()
    if y >= max_y - 1:
        return y
    w = min(width, max_x - x - 1)
    try:
        win.addstr(y, x, char * w, attr)
    except curses.error:
        pass
    return y + 1


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


# ── Main sections ─────────────────────────────────────────────────────────────

def render_header(win, y: int, container: str, status: str, uptime: str) -> int:
    """Draw the top banner."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname = os.uname().nodename

    title = f" mtop v{__version__} — Ollama Model Monitor "
    y = safe_addstr(win, y, 1, f"─── {title}───", curses.color_pair(C_HEADER) | curses.A_BOLD)

    # Status line
    if status == "running":
        status_icon = "● "
        status_attr = curses.color_pair(C_OK) | curses.A_BOLD
    elif status == "not found":
        status_icon = "✗ "
        status_attr = curses.color_pair(C_ERR) | curses.A_BOLD
    else:
        status_icon = "○ "
        status_attr = curses.color_pair(C_WARN)

    safe_addstr(win, y, 1, "host: ", curses.color_pair(C_DIM))
    safe_addstr(win, y, 7, hostname, curses.color_pair(C_ACCENT))
    col2 = 30
    safe_addstr(win, y, col2, "container: ", curses.color_pair(C_DIM))
    safe_addstr(win, y, col2 + 11, status_icon + container, status_attr)
    col3 = 62
    if uptime:
        safe_addstr(win, y, col3, f"up: {uptime}", curses.color_pair(C_DIM))
    col4_x = max(col3 + 20, 82)
    safe_addstr(win, y, col4_x, now, curses.color_pair(C_DIM))
    y += 1

    y = draw_hline(win, y, 1, "─", 100, curses.color_pair(C_DIM))
    return y


def render_docker_stats(win, y: int, container: str) -> int:
    """Show container CPU/MEM with progress bars."""
    stats = parse_docker_stats(container)
    if not stats:
        return y

    y = safe_addstr(win, y, 1, "CONTAINER RESOURCES", curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)

    # Parse CPU percentage (Docker reports per-core, e.g. 500% = 5 cores)
    try:
        cpu_raw = float(stats["cpu"].rstrip("%"))
    except (ValueError, TypeError):
        cpu_raw = 0.0
    ncpu = get_cpu_count()
    cpu_normalized = min(cpu_raw / ncpu, 100.0)
    cpu_detail = f"{cpu_raw:.0f}% / {ncpu} cores"
    y = draw_bar(win, y, 3, "CPU  ", cpu_normalized, 30, C_ACCENT)
    safe_addstr(win, y - 1, 62, cpu_detail, curses.color_pair(C_DIM))

    # Parse MEM percentage
    try:
        mem_val = float(stats["mem_pct"].rstrip("%"))
    except (ValueError, TypeError):
        mem_val = 0.0
    y = draw_bar(win, y, 3, "MEM  ", mem_val, 30, C_OK)
    safe_addstr(win, y - 1, 62, stats["mem_usage"], curses.color_pair(C_DIM))

    y += 1
    return y


def render_gpu_stats(win, y: int, gpus: Optional[list[dict]]) -> int:
    """Render GPU info section."""
    y = safe_addstr(win, y, 1, "GPU", curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)

    if gpus is None:
        y = safe_addstr(win, y, 3, "GPU monitoring unavailable", curses.color_pair(C_DIM) | curses.A_DIM)
        y += 1
        return y

    for i, gpu in enumerate(gpus):
        prefix = f"[{i}] {gpu['name']}"
        temp_str = f"  {gpu['temp']}°C" if gpu["temp"] != "N/A" else ""
        y = safe_addstr(win, y, 3, prefix + temp_str, curses.color_pair(C_GPU))

        # GPU utilization bar
        if gpu["util"] != "N/A":
            try:
                util_val = float(gpu["util"])
            except ValueError:
                util_val = 0.0
            y = draw_bar(win, y, 5, "UTIL ", util_val, 25, C_GPU)

        # VRAM bar
        try:
            mem_used = float(gpu["mem_used"])
            mem_total = float(gpu["mem_total"])
            if mem_total > 0:
                mem_pct = mem_used / mem_total * 100
                y = draw_bar(win, y, 5, "VRAM ", mem_pct, 25, C_GPU)
                vram_str = f"{mem_used:.0f} / {mem_total:.0f} MiB"
                safe_addstr(win, y - 1, 60, vram_str, curses.color_pair(C_DIM))
        except (ValueError, TypeError):
            pass

    y += 1
    return y


def render_models(win, y: int, api_url: str) -> int:
    """Fetch and display loaded models from /api/ps."""
    y = safe_addstr(win, y, 1, "LOADED MODELS", curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)

    ok, data = http_get_json(f"{api_url}/api/ps")
    if not ok:
        y = safe_addstr(win, y, 3, f"API error: {data}", curses.color_pair(C_ERR))
        y += 1
        return y

    models = data.get("models", [])
    if not models:
        y = safe_addstr(win, y, 3, "No models currently loaded", curses.color_pair(C_WARN) | curses.A_DIM)
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


def render_ollama_ps(win, y: int, container: str) -> int:
    """Show raw ollama ps output from container (supplementary)."""
    y = safe_addstr(win, y, 1, "OLLAMA PS (raw)", curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)

    ok, out = run_cmd(["docker", "exec", container, "ollama", "ps"])
    if not ok:
        y = safe_addstr(win, y, 3, f"ollama ps failed: {out[:80]}", curses.color_pair(C_ERR))
        y += 1
        return y

    for line in out.split("\n"):
        if not line.strip():
            continue
        # Header line in dim, data lines in normal
        attr = curses.color_pair(C_DIM) if line.startswith("NAME") else curses.color_pair(C_OK)
        y = safe_addstr(win, y, 3, line, attr)

    y += 1
    return y


def render_footer(win, y: int, interval: float) -> int:
    max_y, max_x = win.getmaxyx()
    footer_y = max_y - 1
    footer = f" q: quit │ +/-: interval ({interval:.1f}s) │ mtop v{__version__} "
    try:
        win.addstr(footer_y, 0, footer.ljust(max_x - 1), curses.color_pair(C_DIM) | curses.A_REVERSE)
    except curses.error:
        pass
    return y


# ── Main loop ─────────────────────────────────────────────────────────────────

def curses_main(stdscr, args):
    init_colors()
    curses.curs_set(0)  # hide cursor
    stdscr.nodelay(True)  # non-blocking getch
    stdscr.timeout(int(args.interval * 1000))

    container = args.container
    api_url = args.api_url.rstrip("/")
    show_gpu = not args.no_gpu

    # Cache GPU stats with separate refresh (nvidia-smi is slow)
    gpu_cache: Optional[list[dict]] = None
    gpu_last_fetch: float = 0
    GPU_REFRESH = max(2.0, args.interval)

    while True:
        # Handle input
        try:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):  # q, Q, ESC
                break
            elif key == ord("+"):
                args.interval = max(0.5, args.interval - 0.5)
                stdscr.timeout(int(args.interval * 1000))
            elif key == ord("-"):
                args.interval = min(30.0, args.interval + 0.5)
                stdscr.timeout(int(args.interval * 1000))
        except curses.error:
            pass

        stdscr.erase()
        y = 0

        # Container health check first
        status, uptime = get_container_status(container)
        y = render_header(stdscr, y, container, status, uptime)

        if status != "running":
            y = safe_addstr(stdscr, y + 1, 3,
                            f"Container '{container}' is {status}. Waiting...",
                            curses.color_pair(C_ERR) | curses.A_BOLD)
            y = safe_addstr(stdscr, y + 1, 3,
                            "Will retry automatically.",
                            curses.color_pair(C_DIM))
            render_footer(stdscr, y, args.interval)
            stdscr.refresh()
            continue

        # Docker stats
        y = render_docker_stats(stdscr, y, container)

        # GPU stats (cached, refreshed less frequently)
        if show_gpu:
            now = time.monotonic()
            if now - gpu_last_fetch >= GPU_REFRESH:
                gpu_cache = get_gpu_stats()
                gpu_last_fetch = now
            y = render_gpu_stats(stdscr, y, gpu_cache)

        # Models from API
        y = render_models(stdscr, y, api_url)

        # Raw ollama ps
        y = render_ollama_ps(stdscr, y, container)

        # Footer
        render_footer(stdscr, y, args.interval)

        stdscr.refresh()


def main():
    parser = argparse.ArgumentParser(
        description="mtop — Ollama model monitor for Docker containers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q=quit, +=faster, -=slower\n\nhttps://github.com/Quaerendir/mtop",
    )
    parser.add_argument("-c", "--container", default=DEFAULT_CONTAINER,
                        help=f"Docker container name (default: {DEFAULT_CONTAINER})")
    parser.add_argument("-i", "--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("-u", "--api-url", default=DEFAULT_API_BASE,
                        help=f"Ollama API base URL (default: $OLLAMA_HOST or {DEFAULT_API_BASE})")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU stats section")
    parser.add_argument("-V", "--version", action="version",
                        version=f"mtop {__version__}")
    args = parser.parse_args()

    try:
        curses.wrapper(curses_main, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
