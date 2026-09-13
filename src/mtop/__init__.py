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
    mtop [-c CONTAINER] [-i INTERVAL] [-u URL ...] [-H HEADER] [--insecure]
         [-m MODE] [--runtime RT] [--no-gpu] [--no-runners] [--json] [-h]
"""

import argparse
import base64
import concurrent.futures
import curses
import json
import locale
import os
import re
import shlex
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .container import ContainerRuntime, detect_runtime
from .export import parse_iso, parse_size, prometheus_text
from .gpu import (AmdSysfsProvider, GpuMonitor, GpuProvider, NvidiaSmiProvider,
                  NvmlProvider, RocmSmiProvider, TegraUnifiedProvider)

__version__ = "0.8.0"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONTAINER = "ollama"
DEFAULT_INTERVAL = 1.0
DEFAULT_API_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
API_KEY_ENV = "OLLAMA_API_KEY"    # same variable the ollama CLI uses for Bearer auth

UI_POLL_MS = 100          # curses getch timeout — UI responsiveness, not data rate
SLOW_FLOOR = 2.0          # minimum cadence for docker stats / nvidia-smi
STALE_FACTOR = 3.0        # snapshot older than interval*factor => flagged stale
JSON_CPU_WINDOW = 0.5     # --json: seconds between the two /proc CPU samples
FOREVER_AFTER_SEC = 10 * 365 * 86400   # expires_at this far out == keep_alive -1

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


def run_cmd(cmd: list[str], timeout: int = 5,
            env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Run a command, return (success, stdout_or_stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, r.stderr.strip() or r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, f"command not found: {cmd[0]}"
    except Exception as e:
        return False, str(e)


# urllib honors $http_proxy for *every* host, including 127.0.0.1, so on a
# box with a corporate proxy the local Ollama call comes back as a 502 from the
# proxy. Go's ProxyFromEnvironment — and therefore Ollama's own client — skips
# loopback. Mirror that: env proxies for remote hosts, none for loopback.
_PROXY_OPENER = urllib.request.build_opener()
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}


def is_loopback_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def api_port(url: str) -> int | None:
    """Port of the API URL, defaulting to Ollama's 11434 when unspecified."""
    try:
        parts = urllib.parse.urlsplit(url)
        return parts.port or (443 if parts.scheme == "https" else 11434)
    except ValueError:
        return None


def http_get_json(url: str, timeout: int = 5, headers: dict[str, str] | None = None,
                  context: ssl.SSLContext | None = None) -> tuple[bool, Any]:
    """GET JSON from URL, return (success, data_or_error_string).

    `headers` carry auth (Bearer / Basic / anything a reverse proxy wants);
    `context` a custom TLS setup (--insecure, --cacert). HTTP errors surface
    as "HTTP 401 Unauthorized"-style strings so the screen says *why*.
    """
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   **(headers or {})})
        if context is not None:
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=context),
                *([urllib.request.ProxyHandler({})] if is_loopback_url(url) else []))
        else:
            opener = _DIRECT_OPENER if is_loopback_url(url) else _PROXY_OPENER
        with opener.open(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def make_ssl_context(insecure: bool = False,
                     cacert: str | None = None) -> ssl.SSLContext | None:
    """TLS context for --insecure / --cacert; None means urllib's default."""
    if not insecure and not cacert:
        return None
    ctx = ssl.create_default_context(cafile=cacert) if cacert else ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def split_userinfo(url: str) -> tuple[str, dict[str, str]]:
    """`https://user:pw@host` -> (`https://host`, {"Authorization": "Basic ..."}).

    urllib does not turn URL credentials into a header on its own, and a
    reverse proxy in front of Ollama is most often protected with basic auth.
    """
    parts = urllib.parse.urlsplit(url)
    if not parts.username and not parts.password:
        return url, {}
    user = urllib.parse.unquote(parts.username or "")
    pw = urllib.parse.unquote(parts.password or "")
    cred = f"{user}:{pw}"
    host = parts.hostname or ""
    if ":" in host:                      # IPv6 literal
        host = f"[{host}]"
    netloc = host + (f":{parts.port}" if parts.port else "")
    clean = urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    token = base64.b64encode(cred.encode()).decode()
    return clean, {"Authorization": f"Basic {token}"}


def parse_header_arg(value: str) -> tuple[str, str]:
    """`Name: value` -> ("Name", "value"); raises ValueError otherwise."""
    name, sep, val = value.partition(":")
    if not sep or not name.strip():
        raise ValueError(f"expected 'Name: value', got {value!r}")
    return name.strip(), val.strip()


_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_endpoint_arg(value: str) -> tuple[str | None, str]:
    """`label=URL` or bare `URL` -> (label_or_None, url).

    A label is a plain word before the first '='; anything with a '/' or ':'
    before the '=' is part of a URL (query strings and the like).
    """
    head, sep, rest = value.partition("=")
    if sep and _LABEL_RE.match(head) and rest:
        return head, rest
    return None, value


class Endpoint:
    """One Ollama API base URL plus how to talk to it (auth, TLS, label)."""

    def __init__(self, spec: str, headers: dict[str, str] | None = None,
                 insecure: bool = False, cacert: str | None = None):
        label, url = parse_endpoint_arg(spec)
        url = normalize_api_url(url)
        url, basic = split_userinfo(url)
        self.url = url
        self.headers = {**(headers or {}), **basic}
        self.context = make_ssl_context(insecure, cacert)
        self.label = label or (urllib.parse.urlsplit(url).netloc or url)

    def get_json(self, path: str, timeout: int = 5) -> tuple[bool, Any]:
        return http_get_json(self.url + path, timeout, self.headers or None, self.context)

    def describe(self) -> dict:
        return {"label": self.label, "url": self.url,
                "auth": "Authorization" in self.headers,
                "tls": ("insecure" if self.context is not None
                        and self.context.verify_mode == ssl.CERT_NONE
                        else "custom-ca" if self.context is not None else "default")}


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
    target = parse_iso(iso_str)
    if target is None:
        return iso_str[:19]
    try:
        if target.year <= 1:
            return "never"          # Go zero time: no expiry scheduled
        now = datetime.now(timezone.utc)
        delta = target - now
        total_sec = int(delta.total_seconds())
        if total_sec >= FOREVER_AFTER_SEC:
            # keep_alive -1: Ollama schedules expiry ~292 years out and
            # `ollama ps` prints "Forever". "106394d left" is not helpful.
            return "forever"
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


def host_cpu_count() -> float:
    """CPUs this process may actually run on (cgroup cpuset / taskset aware)."""
    try:
        return float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return float(os.cpu_count() or 1)


def parse_systemd_cpu_quota(value: str) -> float | None:
    """`CPUQuotaPerSecUSec` -> effective core budget, or None for unlimited.

    systemd prints it as a human duration: 'infinity', '2s', '500ms',
    '1.5s', '1min 30s'. One CPU-second per second is one core.
    """
    v = value.strip().lower()
    if not v or v == "infinity":
        return None
    units = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0}
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(us|ms|min|s|h)", v):
        total += float(num) * units[unit]
    return total or None


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


def listening_inodes(port: int) -> set[str]:
    """Socket inodes in LISTEN state on `port`, from /proc/net/tcp{,6}."""
    inodes: set[str] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                next(f, None)                          # header
                for line in f:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != "0A":
                        continue
                    if int(fields[1].rsplit(":", 1)[1], 16) == port:
                        inodes.add(fields[9])
        except (OSError, ValueError, IndexError):
            continue
    return inodes


def pid_owns_socket(pid: int, inodes: set[str]) -> bool:
    """True if any fd of `pid` is one of the given socket inodes.

    Needs read access to /proc/<pid>/fd, i.e. same user or root; a
    PermissionError simply means "don't know".
    """
    if not inodes:
        return False
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return True
    except OSError:
        return False
    return False


def proc_starttime_ticks(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return sys.maxsize


def find_ollama_pids() -> list[int]:
    """Every `ollama serve` PID visible in /proc (Linux only), unordered."""
    pids: list[int] = []
    if not os.path.isdir("/proc"):
        return pids
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        args = read_proc_cmdline(int(entry))
        if not args:
            continue
        if os.path.basename(args[0]) == "ollama" and any(a == "serve" for a in args[1:]):
            pids.append(int(entry))
    return pids


def find_ollama_pid(port: int | None = None) -> int | None:
    """Find the Ollama *server* PID (the one running `serve`).

    Linux: scan /proc for a process whose argv[0] basename is 'ollama' and
    which has 'serve' among its args — this excludes `ollama run`/`ollama ps`
    clients. macOS: pgrep. Returns None if not found.

    With several servers (a user's manual `ollama serve` next to the systemd
    one, or a second instance for a ROCm card) the /proc directory order is
    effectively random. Prefer the process that owns the listening socket on
    the API port when /proc/<pid>/fd is readable, else the oldest one — the
    same answer every cycle either way.

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

    pids = find_ollama_pids()
    if not pids:
        return None
    if len(pids) == 1:
        return pids[0]
    if port:
        inodes = listening_inodes(port)
        for pid in sorted(pids):
            if pid_owns_socket(pid, inodes):
                return pid
    return min(pids, key=proc_starttime_ticks)


def parse_systemd_show(out: str) -> tuple[str, int, float | None] | None:
    """Normalize `systemctl show` output to (status, main_pid, cpu_limit).

    status: running / starting / failed. cpu_limit is the unit's CPUQuota=
    as a core count, or None when unlimited.
    """
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
    quota = parse_systemd_cpu_quota(props.get("CPUQuotaPerSecUSec", ""))
    if active == "active" and pid > 0:
        return "running", pid, quota
    if active == "activating":
        return "starting", pid, quota
    if active == "failed":
        return "failed", pid, quota
    return None


def systemd_ollama() -> tuple[str, int, float | None] | None:
    """Query the ollama systemd unit. Returns (status, main_pid, cpu_limit)
    or None when the unit is absent/inactive.

    Only used for discovery, status and the CPU budget; the numbers come
    from /proc.
    """
    ok, out = run_cmd(
        ["systemctl", "show", "ollama.service",
         "--property=ActiveState,SubState,MainPID,CPUQuotaPerSecUSec"],
        timeout=3,
    )
    if not ok:
        return None
    return parse_systemd_show(out)


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


# ── Effective server configuration ────────────────────────────────────────────
#
# Ollama is configured almost entirely through environment variables, and the
# RUNNERS section shows what it *negotiated* from them. This is the other half:
# what was actually set on the server process. Three sources, by launch method:
#   container  — Config.Env from inspect (always readable)
#   process    — /proc/<pid>/environ (same user or root only)
#   systemd    — `systemctl show -p Environment` (unit + drop-ins; not
#                EnvironmentFile= contents, which systemd does not expose)

ENV_PREFIXES = ("OLLAMA_", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "HIP_", "HSA_", "ROCR_", "GPU_DEVICE_ORDINAL", "GGML_", "LLAMA_")
ENV_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS")


def inference_env(pairs: Iterable[str]) -> dict[str, str]:
    """Filter KEY=VALUE strings down to the inference-relevant ones, masked."""
    out: dict[str, str] = {}
    for kv in pairs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if not k.startswith(ENV_PREFIXES):
            continue
        if any(m in k.upper() for m in ENV_SECRET_MARKERS) and v:
            v = "••••"
        out[k] = v
    return dict(sorted(out.items()))


def read_proc_environ(pid: int) -> list[str] | None:
    """KEY=VALUE list from /proc/<pid>/environ, or None when unreadable."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    return [a for a in raw.decode("utf-8", "replace").split("\0") if a]


def parse_systemd_environment(line: str) -> list[str]:
    """`Environment=A=1 "B=x y"` -> ["A=1", "B=x y"]."""
    _, _, val = line.partition("=")
    try:
        return shlex.split(val)
    except ValueError:
        return val.split()


def systemd_environment(unit: str = "ollama.service") -> list[str] | None:
    ok, out = run_cmd(["systemctl", "show", unit, "--property=Environment"], timeout=3)
    if not ok:
        return None
    for line in out.splitlines():
        if line.startswith("Environment="):
            return parse_systemd_environment(line)
    return []


def link_runners_to_gpus(runners: list[dict] | None, gpus: list[dict] | None) -> None:
    """Join runner processes to cards by PID, both ways, in place.

    Each runner gains ``gpu`` (["nvidia:0", ...]) and ``gpu_mem_mib``; each
    GPU's ``procs`` entries gain ``model`` when the PID is a known runner.
    PIDs are host-namespace on both sides (NVML reports host PIDs; runner PIDs
    come from the host /proc walk), so a runner discovered through the
    in-container exec fallback — container-namespace PIDs — does not link.
    """
    if not runners or not gpus:
        return
    by_pid = {r["pid"]: r for r in runners if r.get("pid")}
    for r in by_pid.values():
        r.pop("gpu", None)
        r.pop("gpu_mem_mib", None)
    for g in gpus:
        for proc in g.get("procs") or []:
            r = by_pid.get(proc.get("pid"))
            if r is None:
                proc.pop("model", None)
                continue
            proc["model"] = r.get("model_name") or r.get("digest", "")[:12]
            tag = f"{g.get('vendor', '?')}:{g.get('index', '?')}"
            r.setdefault("gpu", []).append(tag)
            if proc.get("mem_mib"):
                r["gpu_mem_mib"] = r.get("gpu_mem_mib", 0) + proc["mem_mib"]


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
    "--kv-cache-type": ("kv", True),      # new engine: one flag for K and V
    "--threads": ("threads", True), "-t": ("threads", True),
    "--ollama-engine": ("ollama_engine", False),
    "--multiuser-cache": ("multiuser_cache", False),
    "--mmproj": ("mmproj", True),
    "--port": ("port", True),
    "--tensor-split": ("tensor_split", True), "-ts": ("tensor_split", True),
    "--main-gpu": ("main_gpu", True), "-mg": ("main_gpu", True),
    "--direct-io": ("direct_io", False),   # pre-0.33 spelling
    "--load-mode": ("load_mode", True),    # 0.33+: mmap | dio | ...
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

    # The Ollama engine spells the KV cache dtype as one flag; fold it into
    # the K/V pair the renderer already understands.
    kv = out.pop("kv", None)
    if kv is not None:
        out.setdefault("kv_k", kv)
        out.setdefault("kv_v", kv)
    out["engine"] = "ollama" if out.pop("ollama_engine", False) else "llama"

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


def processor_label(size: int | float | None, size_vram: int | float | None) -> str:
    """The PROCESSOR column exactly as `ollama ps` computes it (cmd/cmd.go)."""
    size = size or 0
    size_vram = size_vram or 0
    if size_vram == 0:
        return "100% CPU"
    if size_vram == size:
        return "100% GPU"
    if size_vram > size or size == 0:
        return "Unknown"
    cpu = round((size - size_vram) / size * 100)
    return f"{cpu}%/{100 - cpu}% CPU/GPU"


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
      docker — inspect/stats/exec a container, through the Engine API on a
               socket when one answers, else the docker/podman CLI
      local  — monitor a bare-metal `ollama serve` process (systemd/proc/ps)
      api    — API only, no host resource stats (former --no-docker)
      auto   — probe docker first, then a local process, else api; the
               resolved mode is cached once a concrete source is found
    """

    def __init__(self, container: str, api_url: str, interval: float,
                 show_gpu: bool, mode: str = "auto", show_raw_ps: bool = False,
                 show_runners: bool = True, runtime: str = "auto",
                 container_runtime: ContainerRuntime | None = None,
                 show_env: bool = True, endpoints: list[Endpoint] | None = None):
        super().__init__(daemon=True, name="mtop-collector")
        self.show_env = show_env
        self.container = container
        self.runtime_pref = runtime         # auto | api | cli
        self._runtime: ContainerRuntime | None = container_runtime
        self._runtime_probed = container_runtime is not None
        # The first endpoint is the *primary*: it is the one the docker/local
        # source, the runner match and the raw `ollama ps` refer to. Any
        # further endpoints are API-only — models and version, nothing about
        # the host they run on.
        self.endpoints = endpoints or [Endpoint(api_url)]
        self.primary = self.endpoints[0]
        self.api_url = self.primary.url
        self._versions: dict[str, str | None] = {}
        # Used to pick the right `ollama serve` when several are running, and
        # only meaningful for a loopback URL — a remote API can't be a local pid.
        self.api_port = api_port(api_url) if is_loopback_url(api_url) else None
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
        self._last_inspect: dict | None = None
        self._uptime_sec: float | None = None     # numeric twin of snap["uptime"]
        # Slow-path too: the version never changes and the environment only
        # changes with a restart.
        self._server: dict = {"version": None, "env": {}, "env_source": None}
        self._pool = (concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.endpoints), thread_name_prefix="mtop-api")
            if len(self.endpoints) > 1 else None)

        # Multi-vendor GPU registry, built lazily so `use_docker` is already
        # resolved when the nvidia provider asks for its argv prefixes.
        self._gpu_monitor: GpuMonitor | None = None

        # Previous CPU sample for the local process tree: (root_pid,
        # {pid: ticks}, monotonic). Keyed by root pid so a server restart
        # resets the baseline, and per-pid inside so a runner spawning or
        # exiting does not register as a CPU spike.
        self._cpu_prev: tuple[int, dict[int, int], float] | None = None

    @property
    def runtime(self) -> ContainerRuntime | None:
        """The container runtime, resolved on first use; None when the host has
        neither an Engine API socket nor a docker/podman CLI."""
        if not self._runtime_probed:
            self._runtime = detect_runtime(run_cmd, self.runtime_pref)
            self._runtime_probed = True
        return self._runtime

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
        rt = self.runtime
        if rt is None:
            return False
        info = rt.inspect(self.container)
        return bool(info) and info["status"] == "running"

    def needs_second_sample(self, snap: dict) -> bool:
        """True when CPU% in this snapshot is a delta we have no baseline for.

        local mode always; docker mode when the Engine API is in use (one-shot
        stats carry no precpu). The CLI path samples internally.
        """
        if not snap.get("pid"):
            return False
        if snap.get("mode") == "local":
            return True
        return snap.get("mode") == "docker" and str(snap.get("runtime", "")).endswith("-api")

    def _detect_local(self) -> bool:
        if IS_LINUX and systemd_ollama() is not None:
            return True
        return find_ollama_pid(self.api_port) is not None

    def force_slow(self) -> None:
        """Make the next collect() refresh the slow-path caches."""
        self._slow_ts = 0.0

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
            snap["runtime"] = self.runtime.name if self.runtime else None

        if mode == "docker":
            status, uptime, cpu_limit, init_pid = self._inspect_container()
            snap.update(status=status, uptime=uptime, cpu_limit=cpu_limit,
                        pid=init_pid, uptime_sec=self._uptime_sec)
            if status != "running":
                return snap
        elif mode == "local":
            status, pid, uptime, cpu_limit = self._local_status()
            snap.update(status=status, uptime=uptime,
                        cpu_limit=cpu_limit or host_cpu_count(), pid=pid,
                        uptime_sec=self._uptime_sec)
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
            self._server = self._server_info(mode, snap.get("pid"))
            self._slow_ts = now

        if slow_due and mode == "docker":
            self._runners = self._docker_runners(snap.get("pid"))

        snap["res_stats"] = self._res_stats if mode != "api" else None
        snap["runners"] = (self._runners if mode == "docker"
                           else (self._res_stats or {}).get("runners"))
        snap["gpus"] = self._gpu_cache if self.show_gpu else None
        snap["server"] = self._server

        results = self._fetch_models()
        primary = results[0]
        snap["models_ok"] = primary["models_ok"]
        snap["models"] = primary["models"]
        snap["models_err"] = primary["models_err"]
        snap["endpoints"] = results
        if snap.get("runners") and snap["models"]:
            match_runners_to_models(snap["runners"], snap["models"])
        link_runners_to_gpus(snap.get("runners"), snap.get("gpus"))

        if self.show_raw_ps and mode in ("docker", "local"):
            if mode == "docker" and self.runtime is not None:
                ok2, out2 = self.runtime.exec(self.container, ["ollama", "ps"])
            elif mode == "docker":
                ok2, out2 = False, "no container runtime"
            else:
                # The CLI reads $OLLAMA_HOST; without this a `-u` pointing at
                # another instance would list the wrong server's models.
                ok2, out2 = run_cmd(["ollama", "ps"],
                                    env={**os.environ, "OLLAMA_HOST": self.api_url})
            snap["raw_ps_ok"] = ok2
            snap["raw_ps"] = out2

        return snap

    def _fetch_models(self) -> list[dict]:
        """/api/ps from every endpoint, in parallel so one dead remote does not
        stall the cycle by its full timeout. Order matches self.endpoints."""
        def one(ep: Endpoint) -> dict:
            ok, data = ep.get_json("/api/ps")
            return {
                "label": ep.label, "url": ep.url,
                "models_ok": ok,
                "models": (data.get("models", []) if ok and isinstance(data, dict) else []),
                "models_err": "" if ok else str(data),
                "version": self._versions.get(ep.label),
            }
        if self._pool is None:
            return [one(self.endpoints[0])]
        return list(self._pool.map(one, self.endpoints))

    def _fetch_versions(self) -> None:
        def one(ep: Endpoint) -> tuple[str, str | None]:
            ok, ver = ep.get_json("/api/version", timeout=3)
            v = str(ver.get("version") or "") if ok and isinstance(ver, dict) else ""
            return ep.label, (v or None)
        pairs = (self._pool.map(one, self.endpoints) if self._pool is not None
                 else [one(self.endpoints[0])])
        self._versions.update(dict(pairs))

    def _server_info(self, mode: str, pid: int | None) -> dict:
        """Ollama version + the inference-relevant environment it was started with."""
        info: dict = {"version": None, "env": {}, "env_source": None}
        self._fetch_versions()
        info["version"] = self._versions.get(self.primary.label)
        if mode == "docker" and self._last_inspect:
            info["env"] = inference_env(self._last_inspect.get("env") or [])
            info["env_source"] = "container"
        elif mode == "local" and pid and IS_LINUX:
            pairs = read_proc_environ(pid)
            if pairs is not None:
                info["env"], info["env_source"] = inference_env(pairs), "process"
            else:
                sd = systemd_environment()
                if sd is not None:
                    info["env"], info["env_source"] = inference_env(sd), "systemd"
        return info

    # -- local (bare-metal) process source -------------------------------------

    def _local_status(self) -> tuple[str, int | None, str, float | None]:
        """(status, pid, uptime_str, cpu_limit) for a bare-metal ollama server.

        Prefers systemd (gives a real activating/failed distinction, the
        MainPID and any CPUQuota=) and falls back to a /proc or pgrep scan for
        manual `ollama serve` launches.
        """
        pid: int | None = None
        status = "not found"
        cpu_limit: float | None = None
        if IS_LINUX:
            sd = systemd_ollama()
            if sd is not None:
                status, pid, cpu_limit = sd
                pid = pid or None
        if pid is None:
            pid = find_ollama_pid(self.api_port)
            if pid is not None:
                status = "running"
        uptime = ""
        self._uptime_sec = None
        if pid and IS_LINUX:
            up = proc_uptime_sec(pid)
            if up is not None:
                uptime = fmt_duration(up)
                self._uptime_sec = up
        return status, pid, uptime, cpu_limit

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
        cpu_pct = max(0.0, cpu_pct)
        return {
            "cpu": f"{cpu_pct:.2f}%",
            "mem_usage": f"{mem / 1024**3:.1f}GiB / {total / 1024**3:.1f}GiB",
            "mem_pct": f"{mem_pct:.1f}%",
            "mem_kind": "pss" if (pss_complete and pss) else "rss",
            "procs": len(tree),
            "runners": runners,
            "cpu_pct": round(cpu_pct, 2),
            "mem_used_bytes": int(mem),
            "mem_limit_bytes": int(total),
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
            "cpu_pct": round(cpu_pct, 2),
            "mem_used_bytes": int(rss),
            "mem_limit_bytes": int(total),
        }

    def _inspect_container(self) -> tuple[str, str, float | None, int | None]:
        """(status, uptime, effective_cpu_limit, init_pid) from one inspect.

        CPU limit comes from HostConfig (NanoCpus for --cpus, quota/period
        for --cpu-quota); falls back to host core count. This makes the CPU
        bar normalize against what the container can actually use instead
        of the host total (a --cpus=4 container saturating on a 32-core
        host previously showed 12.5%).
        """
        rt = self.runtime
        info = rt.inspect(self.container) if rt is not None else None
        self._last_inspect = info
        if not info:
            return "not found", "", None, None
        status = info["status"]
        uptime = ""
        self._uptime_sec = None
        if status == "running" and info["started_at"]:
            uptime = relative_time(info["started_at"]).replace(" ago", "")
            started = parse_iso(info["started_at"])
            if started is not None:
                self._uptime_sec = max(
                    0.0, (datetime.now(timezone.utc) - started).total_seconds())

        cpu_limit: float | None = None
        nano, quota = info["nano_cpus"], info["cpu_quota"]
        period = info["cpu_period"] or 100000
        if nano > 0:
            cpu_limit = nano / 1e9
        elif quota > 0 and period > 0:
            cpu_limit = quota / period
        if not cpu_limit or cpu_limit <= 0:
            cpu_limit = host_cpu_count()
        return status, uptime, cpu_limit, info["pid"]

    def _docker_stats_read(self) -> dict | None:
        rt = self.runtime
        stats = rt.stats(self.container) if rt is not None else None
        if stats and "mem_used_bytes" not in stats:
            # CLI path: only the docker-formatted strings; derive the numbers
            # the exporters want.
            used, _, limit = str(stats.get("mem_usage", "")).partition("/")
            stats["mem_used_bytes"] = parse_size(used)
            stats["mem_limit_bytes"] = parse_size(limit)
            stats["cpu_pct"] = to_float(str(stats.get("cpu", "")).rstrip("%"))
        return stats

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
        rt = self.runtime
        if rt is None:
            return runners or None
        ok, out = rt.exec(self.container,
                          ["sh", "-c", "head -c 4096 /proc/[0-9]*/cmdline 2>/dev/null"],
                          timeout=5)
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
            args = [a for a in body.split("\0") if a.strip()]
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
        def nvidia_attempts() -> list[tuple[str, Any]]:
            attempts: list[tuple[str, Any]] = [("host", run_cmd)]   # host first
            rt = self.runtime if self.use_docker else None
            if rt is not None:
                def in_container(cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
                    return rt.exec(self.container, cmd, timeout)
                attempts.append(("container", in_container))
            return attempts

        providers: list[GpuProvider] = [
            NvmlProvider(),                     # driver library: no fork, per-pid VRAM
            NvidiaSmiProvider(nvidia_attempts),  # superseded when NVML answers
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
               col_widths: list[int], hdr_attr=0, row_attr=0) -> int:
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
    for line in body:
        if y >= max_y - 1:
            break
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
        field("ollama: ", status_icon + (f"serve · pid {pid}{runners}" if pid else "serve"),
              status_attr)
    else:
        field("container: ", status_icon + container, status_attr)
    if uptime and mode != "api" and status != "api-only":
        field("up: ", uptime, curses.color_pair(C_DIM))
    version = (snap.get("server") or {}).get("version")
    if version:
        field("ollama ", version, curses.color_pair(C_DIM))
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


def render_models(win, y: int, snap: dict) -> int:
    """Loaded models from /api/ps — one table per endpoint when there are several."""
    endpoints = snap.get("endpoints") or [snap]
    if len(endpoints) == 1:
        return render_models_table(win, y, endpoints[0], " LOADED MODELS ")
    for ep in endpoints:
        ver = f" · ollama {ep['version']}" if ep.get("version") else ""
        y = render_models_table(win, y, ep, f" LOADED MODELS · {ep.get('label', '?')}{ver} ")
    return y


def render_models_table(win, y: int, ep: dict, title: str) -> int:
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
        y = safe_addstr(win, y, 3, f"no OLLAMA_* variables set — defaults ({source})",
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


def render_footer(win, interval: float, raw_ps: bool, can_raw_ps: bool,
                  runners: bool = True, runtime: str | None = None,
                  env: bool = True):
    max_y, max_x = win.getmaxyx()
    footer_y = max_y - 1
    parts = ["q: quit", f"+/-: interval ({interval:.1f}s)"]
    if can_raw_ps:
        parts.append(f"o: raw ps [{'on' if raw_ps else 'off'}]")
        parts.append(f"r: runners [{'on' if runners else 'off'}]")
        parts.append(f"e: env [{'on' if env else 'off'}]")
    if runtime:
        parts.append(f"via {runtime}")
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
                elif key == ord("e"):
                    collector.show_env = not collector.show_env
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
                if collector.show_env:
                    y = render_server_config(stdscr, y, snap)
                if collector.show_raw_ps and "raw_ps" in snap:
                    y = render_ollama_ps(stdscr, y, snap)

            render_footer(stdscr, collector.interval, collector.show_raw_ps,
                          mode in ("docker", "local"), collector.show_runners,
                          snap.get("runtime"), collector.show_env)
            stdscr.refresh()
    finally:
        collector.stop()


# ── Headless modes: --json / --prometheus, once or --watch ────────────────────

def snapshot_healthy(snap: dict) -> bool:
    return snap.get("status") in ("running", "api-only") and bool(snap.get("models_ok", False))


def write_output(text: str, path: str | None, append: bool = False) -> None:
    """stdout, or a file: appended (NDJSON) or replaced atomically (tmp + rename).

    Atomic replacement is what the node_exporter textfile collector wants —
    it must never read a half-written .prom file.
    """
    if not path:
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    if append:
        with open(path, "a") as f:
            f.write(text)
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def headless_main(args) -> int:
    """--json / --prometheus: one snapshot, or a stream with --watch.

    --json           one indented JSON document (the pre-0.8 --json output)
    --json --watch   one compact JSON object per line (NDJSON), every interval
    --prometheus     text exposition format for a textfile collector or a pipe
    --prometheus --watch   re-rendered every interval; with -o the file is
                     replaced atomically each time, so `mtop --prometheus
                     --watch -o /var/lib/node_exporter/textfile/mtop.prom`
                     is a complete exporter setup with no cron and no port.

    Exit code (one-shot): 0 when the source is up and the API answered, 1
    otherwise. --watch runs until Ctrl-C and exits 0.
    """
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
    )
    fmt = "prometheus" if args.prometheus else "json"
    watch = bool(args.watch)

    def render(snap: dict) -> str:
        snap = dict(snap)
        snap.pop("ts", None)  # monotonic value is meaningless outside the process
        if fmt == "prometheus":
            return prometheus_text(snap, __version__)
        if watch:
            return json.dumps(snap, separators=(",", ":")) + "\n"
        return json.dumps(snap, indent=2) + "\n"

    snap = collector.collect(time.monotonic())
    if collector.needs_second_sample(snap):
        # CPU% is a delta between two samples (/proc ticks, or one-shot Engine
        # API stats), so a single pass can only ever say 0.00%. Take a second
        # one. The docker CLI path samples internally and would pay another
        # ~2 s for nothing, so it is excluded.
        time.sleep(JSON_CPU_WINDOW)
        collector.force_slow()
        snap = collector.collect(time.monotonic())
    write_output(render(snap), args.output, append=(watch and fmt == "json"))
    if not watch:
        return 0 if snapshot_healthy(snap) else 1

    try:
        while True:
            time.sleep(args.interval)
            snap = collector.collect(time.monotonic())
            write_output(render(snap), args.output, append=(fmt == "json"))
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        # `mtop --json --watch | head -3`: the reader is gone. Detach stdout
        # so the interpreter's exit-time flush does not print a traceback.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0


def json_main(args) -> int:
    """Kept for callers of the pre-0.8 name."""
    for attr, default in (("prometheus", False), ("watch", False), ("output", None)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    return headless_main(args)


def main():
    parser = argparse.ArgumentParser(
        description="mtop — Ollama model monitor for Docker containers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q=quit, +=faster, -=slower, o=toggle raw ollama ps, "
               "r=toggle runners, e=toggle server config\n\n"
               "https://github.com/Quaerendir/mtop",
    )
    parser.add_argument("-c", "--container", default=DEFAULT_CONTAINER,
                        help=f"Docker container name (default: {DEFAULT_CONTAINER})")
    parser.add_argument("-i", "--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("-u", "--api-url", action="append", dest="api_urls", metavar="URL",
                        help="Ollama API base URL (default: $OLLAMA_HOST or "
                             f"{DEFAULT_API_BASE}). Repeat for several instances; the "
                             "first is the primary (host stats, runners), the rest are "
                             "API-only. Optional label: -u rig=http://gpu-rig:11434. "
                             "Credentials in the URL become basic auth.")
    parser.add_argument("-H", "--header", action="append", default=[], metavar="'Name: value'",
                        help="Extra HTTP header for every API request (repeatable), e.g. "
                             "'Authorization: Bearer ...' for an instance behind a proxy")
    parser.add_argument("--insecure", action="store_true",
                        help="Skip TLS certificate verification for https:// endpoints")
    parser.add_argument("--cacert", metavar="FILE",
                        help="CA bundle (PEM) to verify https:// endpoints against")
    parser.add_argument("-m", "--mode", choices=["auto", "docker", "local", "api"],
                        default="auto",
                        help="Data source (default: auto — probe docker, then a "
                             "bare-metal ollama process, else api). "
                             "local: monitor a systemd/manual `ollama serve`. "
                             "api: models only, no host resource stats.")
    parser.add_argument("--runtime", choices=["auto", "api", "cli"], default="auto",
                        help="How to reach the container runtime (default: auto — "
                             "Engine API on $DOCKER_HOST or a docker/podman socket, "
                             "else the docker/podman CLI). api: socket only. "
                             "cli: subprocesses only.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU stats section")
    parser.add_argument("--no-runners", action="store_true",
                        help="Hide the RUNNERS section (effective inference "
                             "config parsed from each runner process argv)")
    parser.add_argument("--no-env", action="store_true",
                        help="Hide the SERVER CONFIG section (OLLAMA_* environment "
                             "the server was started with)")
    parser.add_argument("--no-docker", action="store_true",
                        help="Alias for --mode api (kept for compatibility)")
    parser.add_argument("--json", action="store_true",
                        help="One-shot: print a single snapshot as JSON and exit "
                             "(exit code 1 on unhealthy)")
    parser.add_argument("--prometheus", action="store_true",
                        help="One-shot: print the snapshot in Prometheus text exposition "
                             "format and exit (exit code 1 on unhealthy)")
    parser.add_argument("--watch", action="store_true",
                        help="With --json/--prometheus: keep emitting every INTERVAL "
                             "seconds until Ctrl-C. --json --watch prints NDJSON (one "
                             "compact object per line)")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Write to FILE instead of stdout. Prometheus output "
                             "replaces the file atomically (tmp + rename) — point it at "
                             "the node_exporter textfile directory; NDJSON is appended")
    parser.add_argument("-V", "--version", action="version",
                        version=f"mtop {__version__}")
    args = parser.parse_args()
    headers: dict[str, str] = {}
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for h in args.header:
        try:
            name, val = parse_header_arg(h)
        except ValueError as e:
            parser.error(str(e))
        headers[name] = val
    if args.cacert and not os.path.exists(args.cacert):
        parser.error(f"--cacert: no such file: {args.cacert}")
    try:
        args.endpoints = [Endpoint(u, headers, args.insecure, args.cacert)
                          for u in (args.api_urls or [DEFAULT_API_BASE])]
    except ValueError as e:
        parser.error(str(e))
    args.api_url = args.endpoints[0].url
    # --no-docker is the v0.2.0 spelling of "api only"; let it win only when the
    # user didn't pass an explicit --mode, so `--mode local --no-docker` errors
    # toward the explicit choice rather than silently overriding it.
    if args.no_docker and args.mode == "auto":
        args.mode = "api"

    if args.watch and not (args.json or args.prometheus):
        parser.error("--watch needs --json or --prometheus")
    if args.json and args.prometheus:
        parser.error("--json and --prometheus are mutually exclusive")
    if args.json or args.prometheus:
        sys.exit(headless_main(args))

    # curses encodes with the C library's locale; without this, a LANG=C shell
    # (minimal containers, some SSH jump hosts) turns every █ ░ ─ ║ into '?'.
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass
    try:
        curses.wrapper(curses_main, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
