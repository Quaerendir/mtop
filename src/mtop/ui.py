"""
mtop.ui — curses rendering and the interactive loop.

Only this module imports curses, so the headless modes (--json,
--prometheus) run where curses does not exist (Windows).
"""

from __future__ import annotations

import curses
import os
import threading
import time
import urllib.parse
from datetime import datetime

from ._version import __version__
from .collector import STALE_FACTOR, Collector
from .procfs import host_cpu_count
from .runner import processor_label
from .util import bytes_to_gib, fmt_duration, relative_time, to_float

UI_POLL_MS = 100          # curses getch timeout — UI responsiveness, not data rate
UNLOAD_TIMEOUT = 30       # s — a busy server finishes in-flight requests before unloading
LOAD_TIMEOUT = 900        # s — a 100B+ model from a cold disk takes minutes
# `t` on a loaded model: key -> (label, keep_alive value Ollama accepts)
TTL_CHOICES = (("30m", "30m"), ("2h", "2h"), ("24h", "24h"), ("forever", -1))
NOTICE_HOLD = 5.0         # s a model-action result stays in the footer
C_HEADER = 1
C_OK = 2
C_WARN = 3
C_ERR = 4
C_DIM = 5
C_ACCENT = 6
C_TABLE_HDR = 7
C_GPU = 8
C_AMD = 9
C_INTEL = 10


def init_colors():
    """Set up color pairs, or do nothing on a terminal without colors.

    ``start_color()`` raises on TERM=dumb / vt100-style terminals, which is
    what a minimal SSH jump host or a CI log often is. Without pairs every
    ``color_pair(n)`` is a plain attribute and the layout still renders.
    """
    if not curses.has_colors():
        return
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    curses.init_pair(C_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(C_OK, curses.COLOR_GREEN, -1)
    curses.init_pair(C_WARN, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_ERR, curses.COLOR_RED, -1)
    curses.init_pair(C_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(C_ACCENT, curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_TABLE_HDR, curses.COLOR_WHITE, -1)
    curses.init_pair(C_GPU, curses.COLOR_GREEN, -1)
    curses.init_pair(C_AMD, curses.COLOR_RED, -1)
    curses.init_pair(C_INTEL, curses.COLOR_BLUE, -1)


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


def put(win, y: int, x: int, text: str, attr=0, limit: int | None = None) -> int:
    """Write text at (y, x) and return the x just past it.

    `limit` is an exclusive right edge (e.g. a frame border column) the text
    must not run into; the window edge always applies via safe_addstr.
    """
    if limit is not None:
        text = text[: max(0, limit - x)]
    safe_addstr(win, y, x, text, attr)
    return x + len(text)


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
               col_widths: list[int], hdr_attr=0, row_attr=0,
               selected: int | None = None) -> int:
    """Draw a formatted table. Returns next y position.

    The rule under the header is sized to the widest *visible* line rather than
    to the sum of col_widths. A trailing column with an empty header (the
    runner flags) padded its heading with spaces — invisible — while the rule
    got the full column width in dashes and overhung the content, leaving two
    tables on screen with rules of different lengths.
    """
    max_y, _ = win.getmaxyx()

    def render(cells: list[str]) -> str:
        line = ""
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else len(cell)
            if len(cell) > w:
                cell = cell[: w - 1] + "…"
            line += cell.ljust(w) if i < len(cells) - 1 else cell
            if i < len(cells) - 1:
                line += "  "
        return line

    header_line = render(headers)
    body = [render(r) for r in rows]
    rule = "─" * max(len(t.rstrip()) for t in [header_line, *body])

    y = safe_addstr(win, y, x, header_line, hdr_attr)
    y = safe_addstr(win, y, x, rule, curses.color_pair(C_DIM))
    for i, line in enumerate(body):
        if y >= max_y - 1:
            break
        y = safe_addstr(win, y, x, line,
                        row_attr | curses.A_REVERSE if i == selected else row_attr)
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


SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int, vmax: float = 100.0) -> str:
    """Last `width` values as block characters, 8 levels, 0..vmax."""
    vals = list(values)[-width:]
    out = []
    for v in vals:
        level = 0 if v <= 0 else min(7, int(v / vmax * 8))
        out.append(SPARK_CHARS[level])
    return "".join(out)


def draw_spark(win, y: int, x: int, right: int, values: list[float] | None,
               attr=0, min_width: int = 8) -> None:
    """Sparkline between x and `right` (exclusive), or nothing if too narrow.

    Sits between the bar and the right-aligned detail text, so on a narrow
    terminal it is the first thing to go — the bar and the number stay.
    """
    if not values:
        return
    width = right - x
    if width < min_width:
        return
    safe_addstr(win, y, x, sparkline(values, width), attr)


def bar_end_x(x: int, label: str, width: int) -> int:
    """Rightmost column a draw_bar() occupies (bracket + ' 100.0%')."""
    return x + len(label) + width + 2 + 7


def section_header(win, y: int, label: str) -> int:
    max_y, max_x = win.getmaxyx()
    w = max_x - 4
    pad = max(0, w - len(label))
    sep = "─" * (pad // 2) + label + "─" * (pad - pad // 2)
    return safe_addstr(win, y, 2, sep, curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)


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
    mode = snap.get("mode", "docker")
    pid = snap.get("pid")

    # Top border: ╔═══ mtop v0.2.0 — Ollama Model Monitor ═══╗
    server = "llama-swap" if snap.get("backend") == "llama-swap" else "Ollama"
    title = f" mtop v{__version__} — {server} Model Monitor "
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

    # Fields flow left to right with a fixed gap instead of sitting on
    # hardcoded columns (32/64), which overlapped on anything under ~90 cols.
    gap = 3
    x = 3
    lim = inner_right - 1                # keep one blank before the right ║

    def field(label: str, value: str, value_attr) -> None:
        nonlocal x
        x = put(win, y, x, label, curses.color_pair(C_DIM), limit=lim)
        x = put(win, y, x, value, value_attr, limit=lim) + gap

    host_show = hostname if len(hostname) <= 22 else hostname[:21] + "…"
    field("host: ", host_show, curses.color_pair(C_ACCENT))
    if mode == "api" or status == "api-only":
        field("api: ", status_icon + snap.get("api_url", ""), status_attr)
    elif mode == "local":
        # "+2r" = two model runner subprocesses rolled into the stats below.
        procs = (snap.get("res_stats") or {}).get("procs") or 1
        runners = f" +{procs - 1}r" if procs > 1 else ""
        backend = snap.get("backend", "ollama")
        label, verb = (("llama-swap: ", "") if backend == "llama-swap" else ("ollama: ", "serve"))
        field(label, status_icon + (f"{verb} · pid {pid}{runners}".strip(" ·")
                                    if pid else (verb or "running")),
              status_attr)
    else:
        field("container: ", status_icon + container, status_attr)
    if uptime and mode != "api" and status != "api-only":
        field("up: ", uptime, curses.color_pair(C_DIM))
    version = (snap.get("server") or {}).get("version")
    if version:
        vlabel = "llama-swap " if snap.get("backend") == "llama-swap" else "ollama "
        field(vlabel, version, curses.color_pair(C_DIM))
    extra = len(snap.get("endpoints") or []) - 1
    if extra > 0:
        field("+", f"{extra} endpoint{'s' if extra > 1 else ''}", curses.color_pair(C_DIM))

    # Right-align timestamp (or STALE flag) to the inner frame; drop it when
    # the left-hand fields already reach that far.
    right_str = f"STALE {now}" if stale else now
    right_attr = (curses.color_pair(C_ERR) | curses.A_BOLD) if stale \
        else curses.color_pair(C_DIM)
    time_x = inner_right - len(right_str) - 1
    if time_x >= x:
        safe_addstr(win, y, time_x, right_str, right_attr)
    y += 1

    # Bottom border: same length as the top so corners align on inner_right.
    bottom_line = "╚" + "═" * fill + "╝"
    y = safe_addstr(win, y, 1, bottom_line, curses.color_pair(C_HEADER))
    return y


def render_resources(win, y: int, snap: dict) -> int:
    """Show CPU/MEM with progress bars — container or bare-metal process."""
    stats = snap.get("res_stats")
    if not stats:
        return y

    label = " PROCESS RESOURCES " if snap.get("mode") == "local" \
        else " CONTAINER RESOURCES "
    y = section_header(win, y, label)

    # CPU% arrives summed-across-cores (docker {{.CPUPerc}} or /proc ticks
    # delta); normalize against the effective core budget.
    cpu_raw = to_float(str(stats["cpu"]).rstrip("%")) or 0.0
    ncpu = snap.get("cpu_limit") or host_cpu_count()
    cpu_normalized = min(cpu_raw / ncpu, 100.0)
    if ncpu == int(ncpu):
        cpu_detail = f"{cpu_raw:.0f}% / {int(ncpu)} cores"
    else:
        cpu_detail = f"{cpu_raw:.0f}% / {ncpu:.1f} cores"
    hist = snap.get("history") or {}
    _, max_x = win.getmaxyx()
    y = draw_bar(win, y, 3, "CPU  ", cpu_normalized, 30, C_ACCENT)
    draw_detail_right(win, y - 1, bar_end_x(3, "CPU  ", 30), cpu_detail,
                      curses.color_pair(C_DIM))
    draw_spark(win, y - 1, bar_end_x(3, "CPU  ", 30) + 2,
               max_x - len(cpu_detail) - 4, hist.get("cpu"), curses.color_pair(C_ACCENT))

    mem_val = to_float(str(stats["mem_pct"]).rstrip("%")) or 0.0
    y = draw_bar(win, y, 3, "MEM  ", mem_val, 30, C_OK)
    mem_detail = stats["mem_usage"]
    kind = stats.get("mem_kind")
    if kind:
        # PSS and RSS differ by a lot once several runners share a GGUF; say
        # which one the bar is showing rather than making the user guess.
        mem_detail += f" ({kind})"
    draw_detail_right(win, y - 1, bar_end_x(3, "MEM  ", 30), mem_detail,
                      curses.color_pair(C_DIM))
    draw_spark(win, y - 1, bar_end_x(3, "MEM  ", 30) + 2,
               max_x - len(mem_detail) - 4, hist.get("mem"), curses.color_pair(C_OK))

    y += 1
    return y


def render_gpu_stats(win, y: int, snap: dict) -> int:
    """Render GPU info section."""
    gpus = snap.get("gpus")
    hist = snap.get("history") or {}
    _, max_x = win.getmaxyx()
    y = section_header(win, y, " GPU ")

    if gpus is None:
        y = safe_addstr(win, y, 3, "GPU monitoring unavailable",
                        curses.color_pair(C_DIM) | curses.A_DIM)
        y += 1
        return y

    for i, gpu in enumerate(gpus):
        vendor = gpu.get("vendor", "nvidia")
        color = {"amd": C_AMD, "intel": C_INTEL}.get(vendor, C_GPU)
        # Index is the provider's own (nvidia-smi index / PCI order), which is
        # not the position in this list once two vendors are present.
        prefix = f"[{vendor}:{gpu.get('index', i)}] {gpu['name']}"
        temp_val = to_float(gpu["temp"])
        temp_str = f"  {gpu['temp']}°C" if temp_val is not None else ""
        if gpu.get("power") and to_float(gpu["power"]) is not None:
            temp_str += f"  {to_float(gpu['power']):.0f}W"
        if gpu.get("freq_mhz") is not None:
            temp_str += f"  {gpu['freq_mhz']} MHz"
        if gpu.get("unified"):
            temp_str += "  (unified memory)"
        y = safe_addstr(win, y, 3, prefix + temp_str, curses.color_pair(color))

        hkey = f"gpu:{vendor}:{gpu.get('index', i)}"
        # GPU utilization bar
        util_val = to_float(gpu["util"])
        if util_val is not None:
            y = draw_bar(win, y, 5, "UTIL ", util_val, 25, color)
            draw_spark(win, y - 1, bar_end_x(5, "UTIL ", 25) + 2, max_x - 2,
                       hist.get(hkey + ":util"), curses.color_pair(color))

        # VRAM bar
        mem_used = to_float(gpu["mem_used"])
        mem_total = to_float(gpu["mem_total"])
        if mem_used is None and mem_total:
            # Intel discrete via sysfs: the total is known, usage is not.
            y = safe_addstr(win, y, 5, f"VRAM  {mem_total:.0f} MiB total (usage not exposed)",
                            curses.color_pair(C_DIM))
        if mem_used is not None and mem_total and mem_total > 0:
            mem_pct = mem_used / mem_total * 100
            label = "MEM  " if gpu.get("unified") else "VRAM "
            y = draw_bar(win, y, 5, label, mem_pct, 25, color)
            vram_str = f"{mem_used:.0f} / {mem_total:.0f} MiB"
            draw_detail_right(win, y - 1, bar_end_x(5, label, 25), vram_str,
                              curses.color_pair(C_DIM))
            draw_spark(win, y - 1, bar_end_x(5, label, 25) + 2,
                       max_x - len(vram_str) - 4, hist.get(hkey + ":mem"),
                       curses.color_pair(color))

        # Compute processes on the card (NVML): runner names when we know them.
        procs = gpu.get("procs") or []
        if procs:
            bits = []
            for pr in procs:
                who = pr.get("model") or f"pid {pr.get('pid')}"
                mem = pr.get("mem_mib")
                bits.append(f"{who} ({mem / 1024:.1f}G)" if mem else who)
            y = safe_addstr(win, y, 5, "procs: " + ", ".join(bits),
                            curses.color_pair(C_DIM))

        # GTT is a second pool on dGPUs (host memory the card can pull from);
        # only worth a line when something actually lives there.
        gtt_used = to_float(gpu.get("gtt_used", ""))
        gtt_total = to_float(gpu.get("gtt_total", ""))
        if gtt_used and gtt_total and gtt_used / gtt_total > 0.01:
            y = draw_bar(win, y, 5, "GTT  ", gtt_used / gtt_total * 100, 25, C_DIM)
            draw_detail_right(win, y - 1, bar_end_x(5, "GTT  ", 25),
                              f"{gtt_used:.0f} / {gtt_total:.0f} MiB",
                              curses.color_pair(C_DIM))

    y += 1
    return y


def render_models(win, y: int, snap: dict,
                  selected: tuple[int, str] | None = None) -> int:
    """Models from /api/ps (Ollama) or /v1/models+/running (llama-swap) —
    one table per endpoint when there are several.

    `selected` is (endpoint index, model name) of the --control cursor.
    """
    backend = snap.get("backend", "ollama")
    endpoints = snap.get("endpoints") or [snap]

    def table(y: int, i: int, ep: dict, title: str) -> int:
        name = selected[1] if selected and selected[0] == i else None
        if backend == "llama-swap":
            return render_llama_swap_models_table(win, y, ep, title, name)
        return render_models_table(win, y, ep, title, name)

    title = " MODELS " if backend == "llama-swap" else " LOADED MODELS "
    if len(endpoints) == 1:
        return table(y, 0, endpoints[0], title)
    for i, ep in enumerate(endpoints):
        ver = f" · {backend} {ep['version']}" if ep.get("version") else ""
        y = table(y, i, ep, f"{title}· {ep.get('label', '?')}{ver} ")
    return y


def render_llama_swap_models_table(win, y: int, ep: dict, title: str,
                                   selected_name: str | None = None) -> int:
    """llama-swap's catalog: every *configured* model, loaded or not — unlike
    Ollama's /api/ps, which only ever lists what is currently loaded. There is
    no VRAM/RAM split or context length here; that lives in the runner argv
    (RUNNERS section) once a model actually starts.
    """
    y = section_header(win, y, title)

    if not ep.get("models_ok", False):
        y = safe_addstr(win, y, 3, f"API error: {ep.get('models_err', '?')}",
                        curses.color_pair(C_ERR))
        y += 1
        return y

    models = ep.get("models", [])
    if not models:
        y = safe_addstr(win, y, 3, "No models configured",
                        curses.color_pair(C_WARN) | curses.A_DIM)
        y += 1
        return y

    headers = ["MODEL", "STATE", "PORT", "TTL", "DESCRIPTION"]
    col_widths = [24, 10, 7, 10, 45]
    rows = []
    for m in models:
        state = m.get("state") or "unknown"
        ttl = m.get("ttl")
        ttl_str = "—" if ttl in (None, 0) else fmt_duration(ttl)
        rows.append([
            m.get("name", "?"),
            state,
            str(m.get("port") or "—"),
            ttl_str,
            m.get("description", "") or "",
        ])

    # draw_table's row_attr is a single style for the whole table (see its
    # docstring) — no per-row coloring, so "starting" and "ready" share the
    # same OK color rather than adding that capability for one table.
    names = [m.get("name", "?") for m in models]
    y = draw_table(win, y, 3, headers, rows, col_widths,
                   hdr_attr=curses.color_pair(C_TABLE_HDR) | curses.A_BOLD,
                   row_attr=curses.color_pair(C_OK),
                   selected=names.index(selected_name) if selected_name in names else None)
    y += 1
    return y


def render_models_table(win, y: int, ep: dict, title: str,
                        selected_name: str | None = None) -> int:
    y = section_header(win, y, title)

    if not ep.get("models_ok", False):
        y = safe_addstr(win, y, 3, f"API error: {ep.get('models_err', '?')}",
                        curses.color_pair(C_ERR))
        y += 1
        return y

    models = ep.get("models", [])
    if not models:
        y = safe_addstr(win, y, 3, "No models currently loaded",
                        curses.color_pair(C_WARN) | curses.A_DIM)
        y += 1
        return y

    headers = ["MODEL", "VRAM", "RAM", "CTX", "PROCESSOR", "EXPIRES"]
    col_widths = [36, 10, 10, 8, 16, 14]
    rows = []
    for m in models:
        name = m.get("name", "?")
        size_vram = m.get("size_vram") or 0
        size_total = m.get("size") or 0
        size_ram = max(0, size_total - size_vram)
        ctx = str(m.get("context_length", 0))
        processor = processor_label(size_total, size_vram)
        expires = relative_time(m.get("expires_at", ""))
        rows.append([
            name,
            bytes_to_gib(size_vram) + " G",
            bytes_to_gib(size_ram) + " G",
            ctx,
            processor,
            expires,
        ])

    names = [m.get("name", "?") for m in models]
    y = draw_table(win, y, 3, headers, rows, col_widths,
                   hdr_attr=curses.color_pair(C_TABLE_HDR) | curses.A_BOLD,
                   row_attr=curses.color_pair(C_OK),
                   selected=names.index(selected_name) if selected_name in names else None)
    y += 1
    return y


def render_ollama_ps(win, y: int, snap: dict) -> int:
    """Show raw ollama ps output from the snapshot (toggle: 'o')."""
    y = section_header(win, y, " OLLAMA PS ")

    if snap.get("backend") != "ollama":
        y = safe_addstr(win, y, 3, "no `ollama ps` equivalent for this backend",
                        curses.color_pair(C_WARN) | curses.A_DIM)
        y += 1
        return y

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


def render_runners(win, y: int, snap: dict) -> int:
    """Effective inference config, one row per model runner process.

    Everything here comes from the runner's argv, which is the only place the
    negotiated settings are observable: `/api/ps` reports the context length and
    stops. `FA` reading `on` vs `auto` is the difference between
    OLLAMA_FLASH_ATTENTION having reached the server and the backend deciding
    for itself.

    VRAM and HOST are deliberately separate columns measuring different things.
    VRAM is Ollama's own `size_vram` for the matched model; HOST is the runner
    process's resident set. On CPU inference they converge. On an accelerator
    they do not and should not: weights allocated through CUDA/ROCm/Metal are
    not charged to the process, so a GB10 Spark holding an 82 GB model reports
    ~7.5 GiB of host RSS. The gap between the two columns *is* the device-memory
    footprint.
    """
    runners = snap.get("runners")
    if not runners:
        return y
    y = section_header(win, y, " RUNNERS ")

    rows = []
    for r in runners:
        kv_k, kv_v = r.get("kv_k", ""), r.get("kv_v", "")
        kv = kv_k if kv_k == kv_v else "/".join(x for x in (kv_k, kv_v) if x)
        rss = r.get("rss")
        vram = r.get("vram")
        # Ollama blobs have no name but a digest; a llama-server under
        # llama-swap that matched nothing still has its GGUF file name.
        name = (r.get("model_name") or r.get("digest", "")[:12]
                or os.path.basename(str(r.get("model") or "")) or "—")
        extras = []
        if r.get("engine") == "ollama":
            extras.append("ollama-engine")
        if r.get("ngl"):
            extras.append(f"ngl:{r['ngl']}")
        if r.get("threads"):
            extras.append(f"thr:{r['threads']}")
        if r.get("mmproj"):
            extras.append("mmproj")
        if r.get("multiuser_cache"):
            extras.append("multiuser")
        if r.get("direct_io") or r.get("load_mode") == "dio":
            extras.append("O_DIRECT")
        elif r.get("load_mode"):
            extras.append(f"load:{r['load_mode']}")
        if r.get("engine") == "vllm":
            if r.get("gpu_mem_util"):
                extras.append(f"gpu-util:{r['gpu_mem_util']}")
            if r.get("quantization"):
                extras.append(f"quant:{r['quantization']}")
            if r.get("tp") and r["tp"] != "1":
                extras.append(f"tp:{r['tp']}")
            if r.get("trust_remote_code"):
                extras.append("trust-remote-code")
        if r.get("state"):                  # llama-swap's view, any engine
            extras.append(f"state:{r['state']}")
        # Card indices from the NVML pid join; "—" when nothing linked (AMD,
        # unified parts, or runners seen only through the container exec).
        gpu_col = ",".join(t.split(":", 1)[1] for t in r.get("gpu") or []) or "—"
        rows.append([
            str(r.get("pid", "—")),
            name,
            str(r.get("ctx", "—")),
            str(r.get("batch", "—")),
            str(r.get("flash_attn", "—")),
            kv or "—",
            f"{vram / 1024**3:.1f} G" if vram else "—",
            f"{rss / 1024**3:.1f} G" if rss else "—",
            gpu_col,
            ",".join(extras) or "",
        ])
    y = draw_table(win, y, 3,
                   ["PID", "MODEL", "CTX", "BATCH", "FA", "KV", "VRAM", "HOST", "GPU", ""],
                   rows, [8, 32, 7, 6, 5, 9, 8, 8, 5, 48],
                   hdr_attr=curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)
    y += 1
    return y


def render_server_config(win, y: int, snap: dict) -> int:
    """The environment Ollama was started with (toggle: 'e').

    Pairs flow left to right and wrap at the frame, so a dozen variables take
    two or three lines instead of a dozen. The source is named because it
    decides what can be trusted: `container` and `process` are the real
    environment, `systemd` is only what the unit and its drop-ins declare.
    """
    server = snap.get("server") or {}
    if snap.get("mode") == "api":
        return y
    y = section_header(win, y, " SERVER CONFIG ")
    env = server.get("env") or {}
    source = server.get("env_source")
    if not source:
        if snap.get("mode") == "local":
            msg = "environment not readable (run mtop as the ollama user or root)"
        else:
            msg = "environment unavailable"
        y = safe_addstr(win, y, 3, msg, curses.color_pair(C_DIM) | curses.A_DIM)
        return y + 1
    if not env:
        if snap.get("backend") == "llama-swap":
            # The model servers' VLLM_* / HF_* usually come from the launcher
            # script in the llama-swap config, not from llama-swap itself.
            msg = (f"no VLLM_* / HF_* / CUDA_* variables in llama-swap's environment "
                   f"({source}) — per-model launchers may set their own")
        else:
            msg = f"no OLLAMA_* variables set — defaults ({source})"
        y = safe_addstr(win, y, 3, msg,
                        curses.color_pair(C_DIM))
        return y + 1
    _, max_x = win.getmaxyx()
    limit = max_x - 3
    x = 3
    for k, v in env.items():
        pair = f"{k}={v}"
        if x > 3 and x + len(pair) > limit:
            y += 1
            x = 3
        safe_addstr(win, y, x, k, curses.color_pair(C_DIM))
        safe_addstr(win, y, x + len(k), "=" + v, curses.color_pair(C_OK))
        x += len(pair) + 3
    y += 1
    y = safe_addstr(win, y, 3, f"source: {source}", curses.color_pair(C_DIM) | curses.A_DIM)
    return y + 1


def format_request_stats(req: dict | None) -> str:
    """'12 req/60s · 11×2xx 1×5xx · p50 1.2s' — or '' when nothing to say."""
    if not req:
        return ""
    if not req.get("total"):
        return f"no requests in last {int(req.get('window_sec', 60))}s"
    parts = [f"{req['total']} req/{int(req.get('window_sec', 60))}s"]
    classes = " ".join(f"{n}×{cls}" for cls, n in sorted(req.get("by_status", {}).items()))
    if classes:
        parts.append(classes)
    if req.get("latency_p50") is not None:
        parts.append(f"p50 {req['latency_p50']:.2g}s")
    return " · ".join(parts)


def render_logs(win, y: int, snap: dict) -> int:
    """Tail of the server log with request stats in the section title (toggle: 'l')."""
    logs = snap.get("logs")
    if logs is None:
        return y
    src = logs.get("source")
    stats = format_request_stats(logs.get("requests"))
    title = " LOGS" + (f" · {src}" if src else "") + (f" · {stats}" if stats else "") + " "
    y = section_header(win, y, title)
    if not logs.get("ok"):
        y = safe_addstr(win, y, 3, f"logs unavailable: {logs.get('error', '?')}",
                        curses.color_pair(C_DIM) | curses.A_DIM)
        return y + 1
    lines = logs.get("lines") or []
    if not lines:
        y = safe_addstr(win, y, 3, "(empty)", curses.color_pair(C_DIM) | curses.A_DIM)
        return y + 1
    attrs = {"error": curses.color_pair(C_ERR), "warn": curses.color_pair(C_WARN)}
    for ln in lines:
        ts = ln.get("ts")
        stamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S ") if ts else ""
        safe_addstr(win, y, 3, stamp, curses.color_pair(C_DIM))
        y = safe_addstr(win, y, 3 + len(stamp), ln.get("text", ""),
                        attrs.get(ln.get("level"), curses.color_pair(C_DIM)))
    return y + 1


def render_footer(win, interval: float, raw_ps: bool, can_raw_ps: bool,
                  runners: bool = True, runtime: str | None = None,
                  env: bool = True, logs: bool = False, control: bool | str = False,
                  notice: tuple[str, int] | None = None):
    """Key hints, or — while a model action asks or reports — `notice` (text, attr)."""
    max_y, max_x = win.getmaxyx()
    footer_y = max_y - 1
    if notice:
        text, attr = notice
        try:
            win.addstr(footer_y, 0, (" " + text + " ")[: max_x - 1].ljust(max_x - 1),
                       attr | curses.A_REVERSE)
        except curses.error:
            pass
        return
    parts = ["q: quit", f"+/-: interval ({interval:.1f}s)"]
    if control:
        parts.append("↑/↓: select │ s: stop │ L: load model" if control == "llama-swap"
                     else "↑/↓: select │ s: stop │ t: keep loaded │ L: load model")
    if can_raw_ps:
        parts.append(f"o: raw ps [{'on' if raw_ps else 'off'}]")
        parts.append(f"r: runners [{'on' if runners else 'off'}]")
        parts.append(f"e: env [{'on' if env else 'off'}]")
        parts.append(f"l: logs [{'on' if logs else 'off'}]")
    if runtime:
        parts.append(f"via {runtime}")
    parts.append(f"mtop v{__version__}")
    footer = " " + " │ ".join(parts) + " "
    footer = footer[: max_x - 1].ljust(max_x - 1)
    try:
        win.addstr(footer_y, 0, footer, curses.color_pair(C_DIM) | curses.A_REVERSE)
    except curses.error:
        pass


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

    def notice(self, now: float) -> tuple[str, int] | None:
        """Footer override: the picker keys, the y/n question, the running
        action, or its result."""
        if self.picker:
            tab = " │ Tab: next endpoint" if len(self.endpoints) > 1 else ""
            full = self.evictions(self.picker["ep"])
            if full:
                return (f"↑/↓ PgUp/PgDn: choose │ Enter: load │ Esc: cancel{tab} │ "
                        f"{len(full[0])}/{full[1]} loaded: a new model evicts one",
                        curses.color_pair(C_WARN))
            return (f"↑/↓ PgUp/PgDn: choose │ Enter: load │ Esc: cancel{tab}",
                    curses.color_pair(C_ACCENT))
        if self.load_confirm:
            ep, name, loaded, limit, source = self.load_confirm
            # The question first: long model names get cut at the right edge.
            return (f"Load {self.label((ep, name))}? [y/N] — {len(loaded)}/{limit} loaded "
                    f"({source}), Ollama will unload one of: {', '.join(loaded)}",
                    curses.color_pair(C_WARN) | curses.A_BOLD)
        if self.swap_confirm:
            ep, name, others = self.swap_confirm
            return (f"Load {self.label((ep, name))}? [y/N] — llama-swap will stop "
                    f"{', '.join(others)} unless its config groups them",
                    curses.color_pair(C_WARN) | curses.A_BOLD)
        if self.ttl_target:
            keys = " │ ".join(f"{n}: {label}" for n, (label, _) in enumerate(TTL_CHOICES, 1))
            return (f"Keep {self.label(self.ttl_target)} loaded for — {keys} │ other key: cancel",
                    curses.color_pair(C_WARN) | curses.A_BOLD)
        if self.confirm:
            return (f"Stop {self.label(self.confirm)}? [y/N]",
                    curses.color_pair(C_WARN) | curses.A_BOLD)
        busy = self.busy
        if busy:
            verb, i, name, started = busy
            return (f"{verb} {self.label((i, name))}… {now - started:.0f}s",
                    curses.color_pair(C_ACCENT))
        with self._lock:
            result = self._result
        if result and now < result[2]:
            return result[0], curses.color_pair(C_OK if result[1] else C_ERR)
        return None


def render_picker(win, control: ModelControl, snap: dict) -> None:
    """The load overlay: a framed, scrolling list of /api/tags, centred.

    Models already in /api/ps of that endpoint are marked, so loading one
    (which only resets its keep-alive) is not a surprise.
    """
    pk = control.picker
    if pk is None:
        return
    max_y, max_x = win.getmaxyx()
    with control._lock:
        items = list(pk["items"]) if pk["items"] is not None else None
        err = pk["err"]
    width = min(max_x - 4, 96)
    # Shrink to the list, never past the footer; 2 rows of frame.
    height = max(3, min(len(items or [None]) + 2, max_y - 5))
    if width < 30:
        return
    x0, y0 = (max_x - width) // 2, 3             # below the header frame
    endpoints = snap.get("endpoints") or [snap]
    ep_snap = endpoints[pk["ep"]] if pk["ep"] < len(endpoints) else {}
    loaded = {m.get("name") for m in ep_snap.get("models", [])}
    title = " LOAD MODEL "
    if len(control.endpoints) > 1:
        title = f" LOAD MODEL · {control.endpoints[pk['ep']].label} "

    if items is None:
        lines, attrs = ["Fetching /api/tags…"], [curses.color_pair(C_DIM)]
    elif err:
        lines, attrs = [f"/api/tags failed: {err}"], [curses.color_pair(C_ERR)]
    elif not items:
        lines, attrs = ["No models on this server"], [curses.color_pair(C_WARN)]
    else:
        body = height - 2
        idx = pk["index"]
        if idx < pk["top"]:
            pk["top"] = idx
        elif idx >= pk["top"] + body:
            pk["top"] = idx - body + 1
        name_w = width - 2 - 2 - 30
        lines, attrs = [], []
        for n, m in enumerate(items[pk["top"]:pk["top"] + body], start=pk["top"]):
            det = m.get("details") or {}
            name = m.get("name", "?")
            if len(name) > name_w:
                name = name[: name_w - 1] + "…"
            mark = "●" if m.get("name") in loaded else " "
            size = bytes_to_gib(m.get("size") or 0) + " G"
            lines.append(f"{mark} {name.ljust(name_w)} {size:>9}  "
                         f"{str(det.get('parameter_size', '')):>7}  "
                         f"{str(det.get('quantization_level', ''))[:8]:<8}")
            attr = curses.color_pair(C_OK)
            attrs.append(attr | curses.A_REVERSE if n == idx else attr)

    frame = curses.color_pair(C_ACCENT)
    inner = width - 2
    safe_addstr(win, y0, x0, "┌" + title.center(inner, "─")[:inner] + "┐", frame)
    for r in range(height - 2):
        safe_addstr(win, y0 + 1 + r, x0, "│", frame)
        text = lines[r] if r < len(lines) else ""
        safe_addstr(win, y0 + 1 + r, x0 + 1, (" " + text)[:inner].ljust(inner),
                    attrs[r] if r < len(attrs) else 0)
        safe_addstr(win, y0 + 1 + r, x0 + width - 1, "│", frame)
    count = f" {pk['index'] + 1}/{len(items)} " if items else ""
    safe_addstr(win, y0 + height - 1, x0, "└" + count.rjust(inner, "─") + "┘", frame)


def curses_main(stdscr, args):
    init_colors()
    try:
        curses.curs_set(0)  # hide cursor
    except curses.error:
        pass                # terminal can't; harmless
    stdscr.timeout(UI_POLL_MS)  # fixed fast poll — data cadence lives in Collector

    collector = Collector(
        container=args.container,
        api_url=args.api_url,
        interval=args.interval,
        show_gpu=not args.no_gpu,
        mode=args.mode,
        show_runners=not args.no_runners,
        runtime=args.runtime,
        show_env=not args.no_env,
        endpoints=args.endpoints,
        show_logs=args.logs,
        log_lines=args.log_lines,
        backend=getattr(args, "backend", "ollama"),
    )
    collector.start()
    backend = getattr(args, "backend", "ollama")
    control = (ModelControl(collector.endpoints, backend)
               if getattr(args, "control", False) else None)
    snap = collector.snapshot()

    try:
        while True:
            try:
                key = stdscr.getch()
                if control and control.picker and key not in (-1, curses.KEY_RESIZE):
                    control.picker_key(key)
                elif control and control.ttl_target and key not in (-1, curses.KEY_RESIZE):
                    control.ttl_key(key)
                elif control and control.asking and key not in (-1, curses.KEY_RESIZE):
                    control.answer(key in (ord("y"), ord("Y")))   # anything else: no
                elif key in (ord("q"), ord("Q"), 27):  # q, Q, ESC
                    break
                elif key == ord("+"):
                    collector.interval = max(0.5, collector.interval - 0.5)
                elif key == ord("-"):
                    collector.interval = min(30.0, collector.interval + 0.5)
                elif key == ord("o"):
                    collector.show_raw_ps = not collector.show_raw_ps
                elif key == ord("r"):
                    collector.show_runners = not collector.show_runners
                elif key == ord("e"):
                    collector.show_env = not collector.show_env
                elif key == ord("l"):
                    collector.show_logs = not collector.show_logs
                    collector.reset_log_source()
                    collector.force_slow()      # fetch on the next cycle, not in 2 s
                elif control and key == curses.KEY_UP:
                    control.move(-1, snap)
                elif control and key == curses.KEY_DOWN:
                    control.move(1, snap)
                elif control and key == ord("s"):
                    control.request_stop()
                elif control and key == ord("t"):
                    control.request_ttl()
                elif control and key == ord("L"):
                    control.request_load()
                elif key == curses.KEY_RESIZE:
                    stdscr.erase()
            except curses.error:
                pass

            snap = collector.snapshot()
            if control:
                control.sync(snap)
            age = time.monotonic() - snap.get("ts", 0.0)
            stale = snap.get("ts", 0.0) > 0 and age > collector.interval * STALE_FACTOR
            mode = snap.get("mode", "")

            stdscr.erase()
            y = render_header(stdscr, 0, snap, stale)

            status = snap.get("status", "starting")
            models_ok = snap.get("models_ok", False)
            source_up = status in ("running", "api-only")
            if status == "starting":
                y = safe_addstr(stdscr, y + 1, 3, "Collecting first snapshot…",
                                curses.color_pair(C_DIM))
            elif not source_up and not models_ok:
                # source (container/process) is down AND the API isn't answering
                subj = (f"Container '{args.container}'" if mode == "docker"
                        else "Ollama process" if mode == "local"
                        else "Ollama")
                y = safe_addstr(stdscr, y + 1, 3,
                                f"{subj} is {status}. Waiting...",
                                curses.color_pair(C_ERR) | curses.A_BOLD)
                y = safe_addstr(stdscr, y + 1, 3,
                                "Will retry automatically.",
                                curses.color_pair(C_DIM))
            else:
                # source up, or source down but API still answering (render models)
                y = render_resources(stdscr, y, snap)
                if not args.no_gpu:
                    y = render_gpu_stats(stdscr, y, snap)
                y = render_models(stdscr, y, snap, control.selected if control else None)
                if collector.show_runners:
                    y = render_runners(stdscr, y, snap)
                if collector.show_env:
                    y = render_server_config(stdscr, y, snap)
                if collector.show_raw_ps and "raw_ps" in snap:
                    y = render_ollama_ps(stdscr, y, snap)
                if collector.show_logs and "logs" in snap:
                    y = render_logs(stdscr, y, snap)

            if control and control.picker:
                render_picker(stdscr, control, snap)
            render_footer(stdscr, collector.interval, collector.show_raw_ps,
                          mode in ("docker", "local"), collector.show_runners,
                          snap.get("runtime"), collector.show_env, collector.show_logs,
                          backend if control else False,
                          control.notice(time.monotonic()) if control else None)
            stdscr.refresh()
    finally:
        collector.stop()
