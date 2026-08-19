#!/usr/bin/env python3
"""
mtop — Ollama model monitor for Docker containers.
curses-based TUI with zero flicker, color-coded status, GPU/container stats.

Architecture (v0.4.0): a background collector thread gathers all data
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

from .gpu import (AmdSysfsProvider, GpuMonitor, GpuProvider, NvidiaSmiProvider,
                  RocmSmiProvider, TegraUnifiedProvider)

__version__ = "0.4.0"

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
C_AMD = 9


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
    curses.init_pair(C_AMD, curses.COLOR_RED, -1)


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


# ── Bare-metal / local Ollama helpers ─────────────────────────────────────────
#
# When Ollama runs outside Docker (the official install.sh systemd service, a
# manual `ollama serve` in tmux, or the macOS app) there is no container to
# inspect. We monitor the server process directly. Numbers come from /proc on
# Linux (world-readable — no root needed, works regardless of launch method)
# and from `ps`/`sysctl` on macOS. systemd is used only for discovery/status,
# not for the numbers, to sidestep the "MemoryAccounting is off" and locale-
# dependent timestamp headaches.

IS_LINUX = sys.platform.startswith("linux")
IS_DARWIN = sys.platform == "darwin"
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def total_ram_bytes() -> int:
    """System RAM in bytes (for the process MEM% denominator)."""
    if IS_DARWIN:
        ok, out = run_cmd(["sysctl", "-n", "hw.memsize"], timeout=2)
        if ok:
            v = to_float(out)
            if v:
                return int(v)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass
    return 0


def find_ollama_pid() -> int | None:
    """Find the Ollama *server* PID (the one running `serve`).

    Linux: scan /proc for a process whose argv[0] basename is 'ollama' and
    which has 'serve' among its args — this excludes `ollama run`/`ollama ps`
    clients. macOS: pgrep. Returns None if not found.

    Note: on a Docker host the containerized `ollama serve` is *also* visible
    in host /proc, so callers must probe Docker before falling back here (auto
    mode does exactly that).
    """
    if IS_DARWIN:
        ok, out = run_cmd(["pgrep", "-f", "ollama serve"], timeout=2)
        if ok and out.strip():
            first = out.split()[0]
            return int(first) if first.isdigit() else None
        ok, out = run_cmd(["pgrep", "-x", "ollama"], timeout=2)
        if ok and out.strip():
            first = out.split()[0]
            return int(first) if first.isdigit() else None
        return None

    if not os.path.isdir("/proc"):
        return None
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                raw = f.read()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        args = [a for a in raw.decode("utf-8", "replace").split("\x00") if a]
        if not args:
            continue
        if os.path.basename(args[0]) == "ollama" and any(a == "serve" for a in args[1:]):
            return int(entry)
    return None


def systemd_ollama() -> tuple[str, int] | None:
    """Query the ollama systemd unit. Returns (status, main_pid) or None.

    status is normalized: running / starting / failed / not found.
    Only used for discovery + status; the numbers come from /proc.
    """
    ok, out = run_cmd(
        ["systemctl", "show", "ollama.service",
         "--property=ActiveState,SubState,MainPID"],
        timeout=3,
    )
    if not ok:
        return None
    props: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            props[k] = v
    active = props.get("ActiveState", "")
    if not active or active == "inactive":
        return None
    try:
        pid = int(props.get("MainPID", "0"))
    except ValueError:
        pid = 0
    if active == "active" and pid > 0:
        return "running", pid
    if active == "activating":
        return "starting", pid
    if active == "failed":
        return "failed", pid
    return None


def read_proc_cpu_ticks(pid: int) -> int | None:
    """Cumulative CPU time (utime+stime) of a PID in clock ticks, via /proc/<pid>/stat.

    Splits on the last ')' so a comm containing spaces/parens can't shift the
    field offsets (the classic /proc/<pid>/stat parsing trap).
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    try:
        fields = data.rsplit(")", 1)[1].split()
        # after the split, index 0 = state (field 3); utime=field14→idx11,
        # stime=field15→idx12
        return int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None


def read_proc_rss_bytes(pid: int) -> int | None:
    """Resident set size of a PID in bytes, via /proc/<pid>/status VmRSS.

    Caveat: Ollama mmaps its GGUF model files, so VmRSS includes resident
    mmapped model pages that also live in the kernel page cache — the process
    footprint can look ≈ model size and appears to "double count" against
    buffers/cache. This is the honest footprint, just worth knowing.
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    return None


def read_proc_pss_bytes(pid: int) -> int | None:
    """Proportional set size via /proc/<pid>/smaps_rollup, or None.

    Preferred over VmRSS when rolling up a process tree: every runner maps the
    same GGUF and the same CUDA/ROCm libraries, so summing VmRSS across the
    tree double-counts the shared pages. PSS divides each shared page by the
    number of mappers, which is exactly the accounting we want.

    smaps_rollup needs PTRACE_MODE_READ on the target, so it works for our own
    processes but not for another user's (the systemd unit runs as `ollama`).
    Callers must fall back to VmRSS.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def read_proc_ppid(pid: int) -> int | None:
    """Parent PID from /proc/<pid>/stat field 4 (idx 1 after the comm split)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        return int(data.rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def proc_children_map() -> dict[int, list[int]]:
    """One /proc scan -> {ppid: [pid, ...]}."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return children
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        ppid = read_proc_ppid(pid)
        if ppid is not None:
            children.setdefault(ppid, []).append(pid)
    return children


def process_tree(root: int, children: dict[int, list[int]] | None = None) -> list[int]:
    """[root] + all descendants.

    Ollama runs the model in a child process, so the weights live below the
    process we discovered via `serve` in argv. Reporting only the root makes a
    multi-GB model look like a ~64 MiB server.

    Matching is on ppid and never on process name: that child has been called
    `ollama_llama_server`, then `ollama runner`, and is `llama-server` in
    current builds. Name matching would break on the next rename — and `pgrep
    -C ollama` never sees it at all.
    """
    if children is None:
        children = proc_children_map()
    out = [root]
    stack = list(children.get(root, []))
    seen = {root}
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def read_proc_cmdline(pid: int) -> list[str] | None:
    """argv of a PID as a list, or None."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except (OSError, ValueError):
        return None
    args = [a for a in raw.decode("utf-8", "replace").split("\x00") if a]
    return args or None


# Every name this process has shipped under. Kept as a hint for the *display*
# only — tree membership is decided by ppid, never by these (see process_tree).
RUNNER_BASENAMES = {"llama-server", "ollama_llama_server", "ollama-runner"}

# argv flag -> (key, takes_value). Long and short spellings both appear
# depending on the llama.cpp vintage Ollama vendored.
_RUNNER_FLAGS: dict[str, tuple[str, bool]] = {
    "--model": ("model", True), "-m": ("model", True),
    "--ctx-size": ("ctx", True), "-c": ("ctx", True),
    "--batch-size": ("batch", True), "-b": ("batch", True),
    "--ubatch-size": ("ubatch", True), "-ub": ("ubatch", True),
    "--parallel": ("parallel", True), "-np": ("parallel", True),
    "--n-gpu-layers": ("ngl", True), "--gpu-layers": ("ngl", True),
    "-ngl": ("ngl", True),
    "--cache-type-k": ("kv_k", True), "-ctk": ("kv_k", True),
    "--cache-type-v": ("kv_v", True), "-ctv": ("kv_v", True),
    "--mmproj": ("mmproj", True),
    "--port": ("port", True),
    "--tensor-split": ("tensor_split", True), "-ts": ("tensor_split", True),
    "--main-gpu": ("main_gpu", True), "-mg": ("main_gpu", True),
    "--direct-io": ("direct_io", False),
    "--no-mmap": ("no_mmap", False),
    "--context-shift": ("context_shift", False),
}


def parse_runner_argv(args: list[str]) -> dict | None:
    """Extract the effective inference config from a runner's argv.

    This is the only place the *negotiated* settings are visible. `/api/ps`
    reports the context length and nothing else; everything Ollama worked out
    between the Modelfile, the environment and its own heuristics — flash
    attention, KV cache dtype, batch sizes, layer split — exists only here.
    Notably `OLLAMA_FLASH_ATTENTION=1` shows up as `--flash-attn on` while an
    unset environment yields `--flash-attn auto`, which is how you tell whether
    the variable actually reached the server.
    """
    if not args:
        return None
    base = os.path.basename(args[0])
    if base not in RUNNER_BASENAMES and not (
            base == "ollama" and "runner" in args[1:]):
        return None

    out: dict[str, Any] = {}
    i = 1
    while i < len(args):
        spec = _RUNNER_FLAGS.get(args[i])
        if spec is None:
            # --flash-attn takes an optional value: 'on'/'off'/'auto' in recent
            # builds, bare (implying on) in older ones.
            if args[i] in ("--flash-attn", "-fa"):
                nxt = args[i + 1] if i + 1 < len(args) else None
                if nxt and not nxt.startswith("-"):
                    out["flash_attn"] = nxt
                    i += 2
                    continue
                out["flash_attn"] = "on"
            i += 1
            continue
        key, takes_value = spec
        if not takes_value:
            out[key] = True
            i += 1
            continue
        if i + 1 < len(args):
            out[key] = args[i + 1]
        i += 2

    model = out.get("model", "")
    digest = ""
    if model:
        stem = os.path.basename(str(model))
        if stem.startswith("sha256-"):
            digest = stem[len("sha256-"):]
    out["digest"] = digest
    return out


def _claim(runner: dict, model: dict) -> None:
    """Copy the API's view of a matched model onto its runner.

    `size_vram` matters most: on accelerator-backed hosts the weights are device
    allocations that never appear in the runner's VmRSS. On a GB10 Spark an 82 GB
    model shows a 7.5 GiB RSS — the bytes come out of the same unified pool
    (/proc/meminfo sees them) but are not charged to the process.
    """
    runner["model_name"] = model.get("name", "")
    runner["vram"] = model.get("size_vram")
    runner["model_size"] = model.get("size")


def match_runners_to_models(runners: list[dict], models: list[dict]) -> None:
    """Best-effort blob -> tag mapping, mutating `runners` in place.

    The `--model` path is a blob digest; `/api/tags` and `/api/ps` expose the
    *manifest* digest, so there is no direct join. Two honest heuristics, in
    order, and a shortened digest when neither is conclusive — a wrong tag on a
    monitoring screen is worse than no tag:

    1. context length, when it uniquely identifies one loaded model;
    2. one unmatched runner left facing one unmatched model.
    """
    unclaimed = list(models)
    for r in runners:
        ctx = to_float(r.get("ctx"))
        if ctx is None:
            continue
        hits = [m for m in unclaimed if to_float(m.get("context_length")) == ctx]
        if len(hits) == 1:
            _claim(r, hits[0])
            unclaimed.remove(hits[0])
    rest = [r for r in runners if not r.get("model_name")]
    if len(rest) == 1 and len(unclaimed) == 1:
        _claim(rest[0], unclaimed[0])


def proc_uptime_sec(pid: int) -> float | None:
    """Seconds since a PID started: system uptime minus the process starttime."""
    try:
        with open("/proc/uptime") as f:
            sys_up = float(f.read().split()[0])
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        fields = data.rsplit(")", 1)[1].split()
        starttime_ticks = int(fields[19])  # field 22 → idx 19
        return max(0.0, sys_up - starttime_ticks / CLK_TCK)
    except (FileNotFoundError, PermissionError, ProcessLookupError,
            ValueError, IndexError):
        return None


def fmt_duration(sec: float) -> str:
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60}m {sec % 60}s"
    if sec < 86400:
        return f"{sec // 3600}h {(sec % 3600) // 60}m"
    return f"{sec // 86400}d {(sec % 86400) // 3600}h"


# ── Collector (background thread) ─────────────────────────────────────────────

class Collector(threading.Thread):
    """Gathers container/GPU/API data in the background.

    Publishes point-in-time snapshot dicts; the UI thread reads the latest
    one under a lock. All blocking I/O (subprocess, HTTP with up to 5 s
    timeouts) lives here so the curses loop stays responsive.

    ``interval`` and ``show_raw_ps`` are mutated from the UI thread; both
    are single-reference reads/writes so the GIL makes them safe without
    additional locking.

    ``mode`` selects the data source:
      docker — inspect/stats/exec a container (the v0.2.0 behavior)
      local  — monitor a bare-metal `ollama serve` process (systemd/proc/ps)
      api    — API only, no host resource stats (former --no-docker)
      auto   — probe docker first, then a local process, else api; the
               resolved mode is cached once a concrete source is found
    """

    def __init__(self, container: str, api_url: str, interval: float,
                 show_gpu: bool, mode: str = "auto", show_raw_ps: bool = False,
                 show_runners: bool = True):
        super().__init__(daemon=True, name="mtop-collector")
        self.container = container
        self.api_url = api_url
        self.interval = interval
        self.show_gpu = show_gpu
        self.mode = mode                    # requested: auto|docker|local|api
        self._resolved: str | None = None   # concrete mode once known
        self.show_raw_ps = show_raw_ps
        self.show_runners = show_runners

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot: dict = {"ts": 0.0, "status": "starting", "uptime": ""}

        # Slow-path caches (container/process stats + nvidia-smi), refreshed at
        # max(SLOW_FLOOR, self.interval) — recomputed each cycle so runtime
        # +/- interval changes take effect (fixes v0.1.0 frozen-cap bug).
        self._slow_ts: float = 0.0
        self._res_stats: dict | None = None
        self._runners: list[dict] | None = None
        self._gpu_cache: list[dict] | None = None

        # Multi-vendor GPU registry, built lazily so `use_docker` is already
        # resolved when the nvidia provider asks for its argv prefixes.
        self._gpu_monitor: GpuMonitor | None = None

        # Previous CPU sample for the local process tree: (root_pid,
        # {pid: ticks}, monotonic). Keyed by root pid so a server restart
        # resets the baseline, and per-pid inside so a runner spawning or
        # exiting does not register as a CPU spike.
        self._cpu_prev: tuple[int, dict[int, int], float] | None = None

    @property
    def use_docker(self) -> bool:
        """True once auto/explicit resolution has settled on the docker source.

        Gates the container-exec GPU probe path in _gpu_read().
        """
        return self._resolved == "docker"

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

    def _resolve_mode(self) -> str:
        """Pick a concrete data source. Docker wins over a bare-metal process
        because a containerized `ollama serve` is also visible in host /proc;
        probing Docker first avoids mistaking the container for a local install.
        """
        if self.mode != "auto":
            return self.mode
        if self._detect_docker():
            return "docker"
        if self._detect_local():
            return "local"
        return "api"

    def _detect_docker(self) -> bool:
        ok, out = run_cmd(
            ["docker", "inspect", "--format", "{{.State.Status}}", self.container]
        )
        return ok and out.strip() == "running"

    def _detect_local(self) -> bool:
        if IS_LINUX and systemd_ollama() is not None:
            return True
        return find_ollama_pid() is not None

    def collect(self, now: float) -> dict:
        """One full collection pass. Also used synchronously by --json."""
        # Resolve the source. Once locked to docker/local we stop probing;
        # while still on the soft 'api' fallback we keep trying to upgrade,
        # so starting mtop before Ollama is up still self-heals.
        if self.mode == "auto":
            if self._resolved in (None, "api"):
                self._resolved = self._resolve_mode()
        elif self._resolved is None:
            self._resolved = self.mode
        mode = self._resolved

        snap: dict = {
            "ts": time.monotonic(),
            "wallclock": datetime.now().astimezone().isoformat(timespec="seconds"),
            "container": self.container,
            "api_url": self.api_url,
            "mode": mode,
        }

        if mode == "docker":
            status, uptime, cpu_limit, init_pid = self._inspect_container()
            snap.update(status=status, uptime=uptime, cpu_limit=cpu_limit,
                        pid=init_pid)
            if status != "running":
                return snap
        elif mode == "local":
            status, pid, uptime = self._local_status()
            snap.update(status=status, uptime=uptime, cpu_limit=float(os.cpu_count() or 1),
                        pid=pid)
            if status != "running":
                # process gone but API might still answer (remote -u, race) —
                # fall through so models still render
                snap["cpu_limit"] = None
        else:  # api
            snap.update(status="api-only", uptime="", cpu_limit=None, pid=None)

        slow_due = now - self._slow_ts >= max(SLOW_FLOOR, self.interval)
        if slow_due:
            if mode == "docker":
                self._res_stats = self._docker_stats_read()
            elif mode == "local" and snap.get("pid"):
                self._res_stats = self._local_stats(snap["pid"])
            elif mode == "api":
                self._res_stats = None
            if self.show_gpu:
                self._gpu_cache = self._gpu_read()
            self._slow_ts = now

        if slow_due and mode == "docker":
            self._runners = self._docker_runners(snap.get("pid"))

        snap["res_stats"] = self._res_stats if mode != "api" else None
        snap["runners"] = (self._runners if mode == "docker"
                           else (self._res_stats or {}).get("runners"))
        snap["gpus"] = self._gpu_cache if self.show_gpu else None

        ok, data = http_get_json(f"{self.api_url}/api/ps")
        snap["models_ok"] = ok
        snap["models"] = data.get("models", []) if ok else []
        snap["models_err"] = "" if ok else str(data)
        if snap.get("runners") and snap["models"]:
            match_runners_to_models(snap["runners"], snap["models"])

        if self.show_raw_ps and mode in ("docker", "local"):
            if mode == "docker":
                cmd = ["docker", "exec", self.container, "ollama", "ps"]
            else:
                cmd = ["ollama", "ps"]
            ok2, out2 = run_cmd(cmd)
            snap["raw_ps_ok"] = ok2
            snap["raw_ps"] = out2

        return snap

    # -- local (bare-metal) process source -------------------------------------

    def _local_status(self) -> tuple[str, int | None, str]:
        """(status, pid, uptime_str) for a bare-metal ollama server.

        Prefers systemd (gives a real activating/failed distinction and the
        MainPID) and falls back to a /proc or pgrep scan for manual
        `ollama serve` launches.
        """
        pid: int | None = None
        status = "not found"
        if IS_LINUX:
            sd = systemd_ollama()
            if sd is not None:
                status, pid = sd
        if pid is None:
            pid = find_ollama_pid()
            if pid is not None:
                status = "running"
        uptime = ""
        if pid and IS_LINUX:
            up = proc_uptime_sec(pid)
            if up is not None:
                uptime = fmt_duration(up)
        return status, pid, uptime

    def _local_stats(self, pid: int) -> dict | None:
        """CPU/MEM for the local server process, in the same shape docker uses.

        CPU is reported summed-across-cores (docker's {{.CPUPerc}} convention),
        so the shared renderer's divide-by-cpu_limit produces a correct %.
        """
        total = total_ram_bytes()
        if IS_DARWIN:
            return self._local_stats_macos(pid, total)
        return self._local_stats_linux(pid, total)

    def _local_stats_linux(self, pid: int, total: int) -> dict | None:
        """CPU/MEM rolled up over the server process and its model runners.

        CPU deltas are computed per-pid and summed only over pids present in
        *both* samples. Summing the tree totals instead would report a huge
        spike the cycle a runner spawns (its accumulated ticks appear at once)
        and a clamped-to-zero dip the cycle one exits.
        """
        tree = process_tree(pid)
        ticks_by_pid: dict[int, int] = {}
        rss = 0
        pss = 0
        pss_complete = True
        runners: list[dict] = []
        for p in tree:
            t = read_proc_cpu_ticks(p)
            if t is not None:
                ticks_by_pid[p] = t
            r = read_proc_rss_bytes(p)
            if r is not None:
                rss += r
            q = read_proc_pss_bytes(p)
            if q is None:
                pss_complete = False
            else:
                pss += q
            if p != pid:
                # cmdline is one extra read on a /proc entry we already opened,
                # and it is the only source for the effective inference config.
                info = parse_runner_argv(read_proc_cmdline(p) or [])
                if info:
                    info["pid"] = p
                    info["rss"] = r
                    runners.append(info)
        if not ticks_by_pid:
            return None

        now = time.monotonic()
        cpu_pct = 0.0
        prev = self._cpu_prev
        if prev is not None and prev[0] == pid:
            dt = now - prev[2]
            if dt > 0:
                delta = sum(ticks_by_pid[p] - prev[1][p]
                            for p in ticks_by_pid if p in prev[1])
                cpu_pct = delta / CLK_TCK / dt * 100.0
        self._cpu_prev = (pid, ticks_by_pid, now)

        # PSS when every process in the tree was readable, else RSS. Mixed
        # accounting would be worse than either: silently omitting a runner's
        # share is a bigger error than double-counting shared library pages.
        mem = pss if (pss_complete and pss) else rss
        mem_pct = (mem / total * 100.0) if total else 0.0
        return {
            "cpu": f"{max(0.0, cpu_pct):.2f}%",
            "mem_usage": f"{mem / 1024**3:.1f}GiB / {total / 1024**3:.1f}GiB",
            "mem_pct": f"{mem_pct:.1f}%",
            "mem_kind": "pss" if (pss_complete and pss) else "rss",
            "procs": len(tree),
            "runners": runners,
        }

    def _local_stats_macos(self, pid: int, total: int) -> dict | None:
        """Same tree rollup as Linux, but from one `ps -ax` snapshot.

        No /proc, and no PSS equivalent that does not need root, so this is
        RSS-based and will over-count pages shared between the runners.
        """
        ok, out = run_cmd(["ps", "-axo", "pid=,ppid=,%cpu=,rss=,args="], timeout=3)
        if not ok or not out.strip():
            return None
        procs: dict[int, tuple[int, float, float]] = {}
        argv: dict[int, list[str]] = {}
        children: dict[int, list[int]] = {}
        for line in out.splitlines():
            f = line.split()
            if len(f) < 4 or not f[0].isdigit() or not f[1].isdigit():
                continue
            p, pp = int(f[0]), int(f[1])
            procs[p] = (pp, to_float(f[2]) or 0.0, (to_float(f[3]) or 0.0) * 1024)
            argv[p] = f[4:]
            children.setdefault(pp, []).append(p)
        if pid not in procs:
            return None

        tree = process_tree(pid, children)
        cpu_pct = sum(procs[p][1] for p in tree if p in procs)
        rss = sum(procs[p][2] for p in tree if p in procs)
        runners = []
        for p in tree:
            if p == pid:
                continue
            info = parse_runner_argv(argv.get(p, []))
            if info:
                info["pid"] = p
                info["rss"] = procs[p][2] if p in procs else None
                runners.append(info)
        mem_pct = (rss / total * 100.0) if total else 0.0
        return {
            "cpu": f"{cpu_pct:.2f}%",
            "mem_usage": f"{rss / 1024**3:.1f}GiB / {total / 1024**3:.1f}GiB",
            "mem_pct": f"{mem_pct:.1f}%",
            "mem_kind": "rss",
            "procs": len(tree),
            "runners": runners,
        }

    def _inspect_container(self) -> tuple[str, str, float | None, int | None]:
        """(status, uptime, effective_cpu_limit, init_pid) in a single inspect call.

        CPU limit comes from HostConfig (NanoCpus for --cpus, quota/period
        for --cpu-quota); falls back to host core count. This makes the CPU
        bar normalize against what the container can actually use instead
        of the host total (a --cpus=4 container saturating on a 32-core
        host previously showed 12.5%).
        """
        fmt = ("{{.State.Status}}\t{{.State.StartedAt}}\t"
               "{{.HostConfig.NanoCpus}}\t{{.HostConfig.CpuQuota}}\t"
               "{{.HostConfig.CpuPeriod}}\t{{.State.Pid}}")
        ok, out = run_cmd(["docker", "inspect", "--format", fmt, self.container])
        if not ok:
            return "not found", "", None, None
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
        init_pid = None
        try:
            if len(parts) > 5 and parts[5].strip():
                init_pid = int(parts[5]) or None
        except ValueError:
            pass
        return status, uptime, cpu_limit, init_pid

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

    def _docker_runners(self, init_pid: int | None) -> list[dict] | None:
        """Runner argv for a containerized Ollama.

        Preferred path: walk the container init PID's tree on the *host* /proc.
        Same kernel, so the runners are visible there, and it costs no exec.
        Falls back to one `docker exec` reading every cmdline in the container's
        PID namespace — needed when mtop itself runs containerized without the
        host PID namespace, at the price of no per-runner RSS.
        """
        runners: list[dict] = []
        if init_pid and IS_LINUX:
            for p in process_tree(init_pid):
                info = parse_runner_argv(read_proc_cmdline(p) or [])
                if info:
                    info["pid"] = p
                    info["rss"] = read_proc_rss_bytes(p)
                    runners.append(info)
            if runners:
                return runners

        # `head -c` on a glob emits '==> /proc/N/cmdline <==' separators, which
        # is the cheapest way to get every argv out in a single exec.
        ok, out = run_cmd(
            ["docker", "exec", self.container, "sh", "-c",
             "head -c 4096 /proc/[0-9]*/cmdline 2>/dev/null"], timeout=5)
        if not ok or not out:
            return runners or None
        pid_now: int | None = None
        for chunk in out.split("==> "):
            if not chunk.strip():
                continue
            head, _, body = chunk.partition(" <==")
            try:
                pid_now = int(head.split("/")[2])
            except (IndexError, ValueError):
                continue
            args = [a for a in body.replace("\x00", "\0").split("\0") if a.strip()]
            if len(args) < 2:
                args = [a for a in body.split() if a]
            info = parse_runner_argv(args)
            if info:
                info["pid"] = pid_now
                info["rss"] = None
                runners.append(info)
        return runners or None

    # -- GPU -------------------------------------------------------------------

    def _build_gpu_providers(self) -> list[GpuProvider]:
        """Every provider that could plausibly answer on this host.

        The registry runs all of them and concatenates the results, so a box
        with an NVIDIA card *and* a Radeon shows both. v0.3.0 memoized a single
        winning strategy and structurally could not.
        """
        def nvidia_prefixes() -> list[list[str]]:
            prefixes: list[list[str]] = [[]]          # host first
            if self.use_docker:
                prefixes.append(["docker", "exec", self.container])
            return prefixes

        providers: list[GpuProvider] = [
            NvidiaSmiProvider(run_cmd, nvidia_prefixes),
        ]
        if IS_LINUX:
            providers += [
                AmdSysfsProvider(),
                RocmSmiProvider(run_cmd),
                TegraUnifiedProvider(read_unified_memory),
            ]
        return providers

    def _gpu_read(self) -> list[dict] | None:
        if self._gpu_monitor is None:
            self._gpu_monitor = GpuMonitor(self._build_gpu_providers(),
                                           time.monotonic)
        return self._gpu_monitor.collect()


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
    mode = snap.get("mode", "docker")
    pid = snap.get("pid")

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
    if mode == "api" or status == "api-only":
        safe_addstr(win, y, col2, "api: ", curses.color_pair(C_DIM))
        safe_addstr(win, y, col2 + 5, status_icon + snap.get("api_url", ""), status_attr)
    elif mode == "local":
        label = "ollama: "
        # "+2r" = two model runner subprocesses rolled into the stats below.
        procs = (snap.get("res_stats") or {}).get("procs") or 1
        runners = f" +{procs - 1}r" if procs > 1 else ""
        val = status_icon + (f"serve · pid {pid}{runners}" if pid else "serve")
        safe_addstr(win, y, col2, label, curses.color_pair(C_DIM))
        safe_addstr(win, y, col2 + len(label), val, status_attr)
        if uptime:
            safe_addstr(win, y, col3, f"up: {uptime}", curses.color_pair(C_DIM))
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
    mem_detail = stats["mem_usage"]
    kind = stats.get("mem_kind")
    if kind:
        # PSS and RSS differ by a lot once several runners share a GGUF; say
        # which one the bar is showing rather than making the user guess.
        mem_detail += f" ({kind})"
    draw_detail_right(win, y - 1, bar_end_x(3, "MEM  ", 30), mem_detail,
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
        vendor = gpu.get("vendor", "nvidia")
        color = C_AMD if vendor == "amd" else C_GPU
        # Index is the provider's own (nvidia-smi index / PCI order), which is
        # not the position in this list once two vendors are present.
        prefix = f"[{vendor}:{gpu.get('index', i)}] {gpu['name']}"
        temp_val = to_float(gpu["temp"])
        temp_str = f"  {gpu['temp']}°C" if temp_val is not None else ""
        if gpu.get("power") and to_float(gpu["power"]) is not None:
            temp_str += f"  {to_float(gpu['power']):.0f}W"
        if gpu.get("unified"):
            temp_str += "  (unified memory)"
        y = safe_addstr(win, y, 3, prefix + temp_str, curses.color_pair(color))

        # GPU utilization bar
        util_val = to_float(gpu["util"])
        if util_val is not None:
            y = draw_bar(win, y, 5, "UTIL ", util_val, 25, color)

        # VRAM bar
        mem_used = to_float(gpu["mem_used"])
        mem_total = to_float(gpu["mem_total"])
        if mem_used is not None and mem_total and mem_total > 0:
            mem_pct = mem_used / mem_total * 100
            label = "MEM  " if gpu.get("unified") else "VRAM "
            y = draw_bar(win, y, 5, label, mem_pct, 25, color)
            vram_str = f"{mem_used:.0f} / {mem_total:.0f} MiB"
            draw_detail_right(win, y - 1, bar_end_x(5, label, 25), vram_str,
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
        name = r.get("model_name") or (r.get("digest", "")[:12] or "—")
        extras = []
        if r.get("ngl"):
            extras.append(f"ngl:{r['ngl']}")
        if r.get("mmproj"):
            extras.append("mmproj")
        if r.get("direct_io"):
            extras.append("O_DIRECT")
        rows.append([
            str(r.get("pid", "—")),
            name,
            str(r.get("ctx", "—")),
            str(r.get("batch", "—")),
            str(r.get("flash_attn", "—")),
            kv or "—",
            f"{vram / 1024**3:.1f} G" if vram else "—",
            f"{rss / 1024**3:.1f} G" if rss else "—",
            ",".join(extras) or "",
        ])
    y = draw_table(win, y, 3,
                   ["PID", "MODEL", "CTX", "BATCH", "FA", "KV", "VRAM", "HOST", ""],
                   rows, [8, 32, 7, 6, 5, 9, 8, 8, 20],
                   hdr_attr=curses.color_pair(C_TABLE_HDR) | curses.A_BOLD)
    y += 1
    return y


def render_footer(win, interval: float, raw_ps: bool, can_raw_ps: bool,
                  runners: bool = True):
    max_y, max_x = win.getmaxyx()
    footer_y = max_y - 1
    parts = ["q: quit", f"+/-: interval ({interval:.1f}s)"]
    if can_raw_ps:
        parts.append(f"o: raw ps [{'on' if raw_ps else 'off'}]")
        parts.append(f"r: runners [{'on' if runners else 'off'}]")
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
        mode=args.mode,
        show_runners=not args.no_runners,
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
                elif key == ord("o"):
                    collector.show_raw_ps = not collector.show_raw_ps
                elif key == ord("r"):
                    collector.show_runners = not collector.show_runners
                elif key == curses.KEY_RESIZE:
                    stdscr.erase()
            except curses.error:
                pass

            snap = collector.snapshot()
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
                y = render_models(stdscr, y, snap)
                if collector.show_runners:
                    y = render_runners(stdscr, y, snap)
                if collector.show_raw_ps and "raw_ps" in snap:
                    y = render_ollama_ps(stdscr, y, snap)

            render_footer(stdscr, collector.interval, collector.show_raw_ps,
                          mode in ("docker", "local"), collector.show_runners)
            stdscr.refresh()
    finally:
        collector.stop()


# ── One-shot JSON mode ────────────────────────────────────────────────────────

def json_main(args) -> int:
    """--json: run one collection pass, dump JSON to stdout.

    Exit code 0 when the source (container/process) is up (or api mode) and
    the API answered; 1 otherwise. Suitable for cron, Prometheus textfile
    collectors (post-processed), or Ansible facts.
    """
    collector = Collector(
        container=args.container,
        api_url=args.api_url,
        interval=args.interval,
        show_gpu=not args.no_gpu,
        mode=args.mode,
        show_runners=not args.no_runners,
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
        epilog="Keys: q=quit, +=faster, -=slower, o=toggle raw ollama ps, "
               "r=toggle runners\n\n"
               "https://github.com/Quaerendir/mtop",
    )
    parser.add_argument("-c", "--container", default=DEFAULT_CONTAINER,
                        help=f"Docker container name (default: {DEFAULT_CONTAINER})")
    parser.add_argument("-i", "--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("-u", "--api-url", default=DEFAULT_API_BASE,
                        help="Ollama API base URL (default: $OLLAMA_HOST or "
                             f"{DEFAULT_API_BASE})")
    parser.add_argument("-m", "--mode", choices=["auto", "docker", "local", "api"],
                        default="auto",
                        help="Data source (default: auto — probe docker, then a "
                             "bare-metal ollama process, else api). "
                             "local: monitor a systemd/manual `ollama serve`. "
                             "api: models only, no host resource stats.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU stats section")
    parser.add_argument("--no-runners", action="store_true",
                        help="Hide the RUNNERS section (effective inference "
                             "config parsed from each runner process argv)")
    parser.add_argument("--no-docker", action="store_true",
                        help="Alias for --mode api (kept for compatibility)")
    parser.add_argument("--json", action="store_true",
                        help="One-shot: print a single snapshot as JSON and exit "
                             "(exit code 1 on unhealthy)")
    parser.add_argument("-V", "--version", action="version",
                        version=f"mtop {__version__}")
    args = parser.parse_args()
    args.api_url = normalize_api_url(args.api_url)
    # --no-docker is the v0.2.0 spelling of "api only"; let it win only when the
    # user didn't pass an explicit --mode, so `--mode local --no-docker` errors
    # toward the explicit choice rather than silently overriding it.
    if args.no_docker and args.mode == "auto":
        args.mode = "api"

    if args.json:
        sys.exit(json_main(args))

    try:
        curses.wrapper(curses_main, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
