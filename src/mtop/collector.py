"""
mtop.collector — the background thread that gathers everything and publishes
immutable snapshots.

All blocking I/O (subprocess, HTTP with up to 5 s timeouts, Engine API
sockets) lives here so the curses loop only ever draws the latest dict.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .container import ContainerRuntime, detect_runtime
from .export import parse_iso, parse_size
from .gpu import (AmdSysfsProvider, AppleGpuProvider, GpuMonitor, GpuProvider,
                  IntelSysfsProvider, NvidiaSmiProvider, NvmlProvider, RocmSmiProvider,
                  TegraUnifiedProvider)
from .logs import ContainerLogs, JournalLogs, LogSource, line_level, request_stats
from .procfs import (find_ollama_pid, host_cpu_count, process_tree, proc_uptime_sec,
                     read_proc_cmdline, read_proc_cpu_ticks, read_proc_environ,
                     read_proc_pss_bytes, read_proc_rss_bytes, read_unified_memory,
                     systemd_environment, systemd_ollama, total_ram_bytes)
from .runner import inference_env, link_runners_to_gpus, match_runners_to_models, parse_runner_argv
from .util import (CLK_TCK, IS_DARWIN, IS_LINUX, Endpoint, api_port, fmt_duration,
                   is_loopback_url, relative_time, run_cmd, to_float)

SLOW_FLOOR = 2.0          # minimum cadence for docker stats / nvidia-smi


STALE_FACTOR = 3.0        # snapshot older than interval*factor => flagged stale


JSON_CPU_WINDOW = 0.5     # --json: seconds between the two /proc CPU samples


HISTORY_LEN = 240         # sparkline samples kept (one per slow cycle, >= 2 s each)


LOG_FETCH = 200           # log lines fetched per slow cycle (request stats window)


DEFAULT_LOG_LINES = 8     # log lines shown on screen


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
                 show_env: bool = True, endpoints: list[Endpoint] | None = None,
                 show_logs: bool = False, log_lines: int = DEFAULT_LOG_LINES):
        super().__init__(daemon=True, name="mtop-collector")
        self.show_env = show_env
        self.show_logs = show_logs
        self.log_lines = log_lines
        self._log_source: LogSource | None = None
        self._log_mode: str | None = None       # mode the source was built for
        self._logs: dict | None = None
        # Sparkline history, one sample per slow cycle. Keys: "cpu", "mem",
        # and "gpu:<vendor>:<index>:<util|mem>".
        self._history: dict[str, deque] = {}
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
            self._logs = self._read_logs(mode) if self.show_logs else None
            self._slow_ts = now

        if slow_due and mode == "docker":
            self._runners = self._docker_runners(snap.get("pid"))

        snap["res_stats"] = self._res_stats if mode != "api" else None
        snap["runners"] = (self._runners if mode == "docker"
                           else (self._res_stats or {}).get("runners"))
        snap["gpus"] = self._gpu_cache if self.show_gpu else None
        snap["server"] = self._server
        if slow_due:
            self._record_history(snap)
        snap["history"] = {k: list(v) for k, v in self._history.items()}
        if self.show_logs:
            snap["logs"] = self._logs

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

    def _record_history(self, snap: dict) -> None:
        def push(key: str, value: float | None) -> None:
            if value is None:
                return
            self._history.setdefault(key, deque(maxlen=HISTORY_LEN)).append(round(value, 1))

        stats = snap.get("res_stats") or {}
        if stats:
            cpu_raw = stats.get("cpu_pct")
            if cpu_raw is None:
                cpu_raw = to_float(str(stats.get("cpu", "")).rstrip("%"))
            ncpu = snap.get("cpu_limit") or host_cpu_count()
            push("cpu", min(cpu_raw / ncpu, 100.0) if cpu_raw is not None else None)
            push("mem", to_float(str(stats.get("mem_pct", "")).rstrip("%")))
        for g in snap.get("gpus") or []:
            key = f"gpu:{g.get('vendor')}:{g.get('index')}"
            push(key + ":util", to_float(g.get("util")))
            used, total = to_float(g.get("mem_used")), to_float(g.get("mem_total"))
            push(key + ":mem", used / total * 100.0 if used is not None and total else None)

    def _build_log_source(self, mode: str) -> LogSource | None:
        if mode == "docker" and self.runtime is not None:
            return ContainerLogs(self.runtime, self.container)
        if mode == "local" and IS_LINUX and systemd_ollama() is not None:
            return JournalLogs(run_cmd)
        return None

    def _read_logs(self, mode: str) -> dict:
        """Tail the server log; request stats over the fetched lines."""
        if self._log_source is None or self._log_mode != mode:
            self._log_source = self._build_log_source(mode)
            self._log_mode = mode
        src = self._log_source
        if src is None:
            why = ("a manual `ollama serve` logs to its own terminal" if mode == "local"
                   else "no log source in api mode")
            return {"source": None, "ok": False, "error": why, "lines": [], "requests": None}
        ok, res = src.tail(max(LOG_FETCH, self.log_lines))
        if not ok:
            return {"source": src.name, "ok": False, "error": str(res), "lines": [],
                    "requests": None}
        lines = res
        return {
            "source": src.name, "ok": True, "error": "",
            "lines": [{"ts": ts, "text": text, "level": line_level(text)}
                      for ts, text in lines[-self.log_lines:]],
            "requests": request_stats(lines),
        }

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
                IntelSysfsProvider(),
                TegraUnifiedProvider(read_unified_memory),
            ]
        if IS_DARWIN:
            providers.append(AppleGpuProvider(run_cmd))
        return providers

    def _gpu_read(self) -> list[dict] | None:
        if self._gpu_monitor is None:
            self._gpu_monitor = GpuMonitor(self._build_gpu_providers(),
                                           time.monotonic)
        return self._gpu_monitor.collect()
