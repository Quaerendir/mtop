"""
mtop.procfs — bare-metal Ollama discovery and process accounting.

When Ollama runs outside Docker (the official install.sh systemd service, a
manual `ollama serve` in tmux, or the macOS app) there is no container to
inspect. We monitor the server process directly. Numbers come from /proc on
Linux (world-readable — no root needed, works regardless of launch method)
and from `ps`/`sysctl` on macOS. systemd is used only for discovery/status,
not for the numbers, to sidestep the "MemoryAccounting is off" and locale-
dependent timestamp headaches.
"""

from __future__ import annotations

import os
import re
import shlex
import sys

from .util import CLK_TCK, IS_DARWIN, run_cmd, to_float

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
