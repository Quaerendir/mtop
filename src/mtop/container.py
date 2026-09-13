"""
mtop.container — container runtime access: Docker Engine API over a socket,
with the `docker` / `podman` CLI as the fallback.

Why the API and not just the CLI:

* No `docker` binary needed on the box mtop runs on. A monitoring host with
  the socket bind-mounted, a distro that only ships podman, or a minimal
  image all work.
* Podman serves the same REST surface on its own socket, so one client
  covers both runtimes.
* `docker stats --no-stream` blocks ~2 s because the CLI waits for two
  samples to compute CPU%. The API's `one-shot=true` returns immediately and
  we compute the delta from our own previous sample, exactly as the local
  /proc path does — the collector cycle no longer sits near the STALE line.
* `docker inspect` forks a Go binary per cycle; a request on an already
  open socket is a fraction of that.

Both implementations expose the same three calls and return the same
normalized dicts, so the collector does not know which one it is talking to.

Everything is stdlib. `DOCKER_HOST` (unix:// or plain tcp://) is honoured
the way the CLI honours it.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import struct
import time
import urllib.parse
from collections.abc import Callable
from typing import Any

Runner = Callable[..., tuple[bool, str]]   # run_cmd(cmd, timeout=..., env=...)

# Sockets probed, in order, when DOCKER_HOST is unset.
def default_socket_candidates() -> list[str]:
    xdg = os.environ.get("XDG_RUNTIME_DIR", "")
    cands = ["/var/run/docker.sock"]
    if xdg:
        cands += [os.path.join(xdg, "docker.sock"),
                  os.path.join(xdg, "podman", "podman.sock")]
    cands.append("/run/podman/podman.sock")
    return cands


# ── formatting helpers (docker CLI look-alikes) ───────────────────────────────

def fmt_bytes_docker(n: float) -> str:
    """go-units HumanSizeWithPrecision(n, 4): 384.8MiB, 1.536GiB, 121.7GiB."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.4g}{units[i]}"


# ── normalized shapes ─────────────────────────────────────────────────────────
#
# inspect() -> {
#   status: str, started_at: str, pid: int|None,
#   nano_cpus: int, cpu_quota: int, cpu_period: int,
#   env: list[str], image: str,
# } or None when the container does not exist / the runtime is unreachable.
#
# stats() -> {
#   cpu: "12.34%", mem_usage: "1.5GiB / 62GiB", mem_pct: "2.4%",   # as docker
#   cpu_pct: float, mem_used_bytes: int, mem_limit_bytes: int,     # api only
# } or None.
#
# exec(name, cmd, timeout) -> (ok, stdout_or_stderr)


class ContainerRuntime:
    name = "base"

    def inspect(self, container: str) -> dict | None:
        raise NotImplementedError

    def stats(self, container: str) -> dict | None:
        raise NotImplementedError

    def exec(self, container: str, cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
        raise NotImplementedError

    def logs(self, container: str, tail: int = 100) -> tuple[bool, str]:
        """Last `tail` lines of stdout+stderr, each prefixed with an RFC3339 stamp."""
        raise NotImplementedError


# ── Engine API over unix / tcp socket ─────────────────────────────────────────

class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


def demux_stream(data: bytes) -> tuple[bytes, bytes]:
    """Split a docker raw-stream (8-byte frame headers) into (stdout, stderr).

    Frames are [type, 0, 0, 0, len(4, big-endian)] + payload. When the data
    does not look framed (a Tty exec, or podman quirks) the whole thing is
    treated as stdout.
    """
    out, err = bytearray(), bytearray()
    i, n = 0, len(data)
    if n < 8 or data[0] not in (0, 1, 2) or data[1:4] != b"\0\0\0":
        return bytes(data), b""
    while i + 8 <= n:
        kind = data[i]
        (size,) = struct.unpack(">I", data[i + 4:i + 8])
        chunk = data[i + 8:i + 8 + size]
        (err if kind == 2 else out).extend(chunk)
        i += 8 + size
    return bytes(out), bytes(err)


class DockerApi(ContainerRuntime):
    """Docker Engine / Podman compat REST API on a unix or plain-tcp socket."""

    def __init__(self, host: str = "unix:///var/run/docker.sock",
                 clock: Callable[[], float] = time.monotonic):
        self.host = host
        self.name = "docker-api"
        self._clock = clock
        self._prev_cpu: dict[str, tuple[int, int, float]] = {}  # name -> (total, system, t)
        u = urllib.parse.urlsplit(host)
        if u.scheme in ("unix", ""):
            self._path: str | None = u.path or host
            self._tcp: tuple[str, int] | None = None
        elif u.scheme == "tcp":
            self._path = None
            self._tcp = (u.hostname or "localhost", u.port or 2375)
        else:
            raise ValueError(f"unsupported DOCKER_HOST scheme: {host}")

    # -- transport -------------------------------------------------------------

    def _conn(self, timeout: float) -> http.client.HTTPConnection:
        if self._path is not None:
            return _UnixHTTPConnection(self._path, timeout)
        return http.client.HTTPConnection(*self._tcp, timeout=timeout)   # type: ignore[misc]

    def request(self, method: str, path: str, body: Any = None,
                timeout: float = 5.0) -> tuple[int, bytes]:
        """One request; returns (status, raw_body). Raises OSError on transport failure."""
        conn = self._conn(timeout)
        try:
            headers = {}
            data = None
            if body is not None:
                data = json.dumps(body).encode()
                headers["Content-Type"] = "application/json"
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def _json(self, method: str, path: str, body: Any = None,
              timeout: float = 5.0) -> tuple[int, Any]:
        status, raw = self.request(method, path, body, timeout)
        try:
            return status, json.loads(raw.decode() or "null")
        except (ValueError, UnicodeDecodeError):
            return status, None

    # -- discovery -------------------------------------------------------------

    def probe(self, timeout: float = 1.0) -> bool:
        """True when something speaks the Engine API here. Sets self.name."""
        try:
            status, ver = self._json("GET", "/version", timeout=timeout)
        except (OSError, http.client.HTTPException):
            return False
        if status != 200 or not isinstance(ver, dict):
            return False
        blob = json.dumps(ver).lower()
        self.name = "podman-api" if "podman" in blob or "libpod" in blob else "docker-api"
        return True

    # -- calls -----------------------------------------------------------------

    def inspect(self, container: str) -> dict | None:
        try:
            status, d = self._json("GET", f"/containers/{urllib.parse.quote(container)}/json")
        except (OSError, http.client.HTTPException):
            return None
        if status != 200 or not isinstance(d, dict):
            return None
        state = d.get("State") or {}
        hc = d.get("HostConfig") or {}
        cfg = d.get("Config") or {}
        return {
            "status": str(state.get("Status", "")),
            "started_at": str(state.get("StartedAt", "")),
            "pid": int(state.get("Pid") or 0) or None,
            "nano_cpus": int(hc.get("NanoCpus") or 0),
            "cpu_quota": int(hc.get("CpuQuota") or 0),
            "cpu_period": int(hc.get("CpuPeriod") or 0),
            "env": list(cfg.get("Env") or []),
            "image": str(cfg.get("Image", "")),
        }

    def stats(self, container: str) -> dict | None:
        try:
            status, s = self._json(
                "GET", f"/containers/{urllib.parse.quote(container)}/stats"
                       "?stream=false&one-shot=true")
        except (OSError, http.client.HTTPException):
            return None
        if status != 200 or not isinstance(s, dict):
            return None
        return self.normalize_stats(container, s)

    def normalize_stats(self, container: str, s: dict) -> dict | None:
        """Docker CLI arithmetic on a stats JSON, with our own CPU baseline.

        CPU% is summed-across-cores (the {{.CPUPerc}} convention the renderer
        divides by cpu_limit). With one-shot the daemon sends no precpu_stats,
        so the delta comes from the previous sample *we* took; precpu_stats is
        used only when present and we have no baseline yet (first cycle on a
        daemon that ignores one-shot).
        """
        cpu = s.get("cpu_stats") or {}
        usage = cpu.get("cpu_usage") or {}
        total = usage.get("total_usage")
        system = cpu.get("system_cpu_usage")
        if total is None:
            return None
        online = (cpu.get("online_cpus")
                  or len(usage.get("percpu_usage") or [])
                  or os.cpu_count() or 1)
        now = self._clock()
        cpu_pct = 0.0
        prev = self._prev_cpu.get(container)
        if prev is None:
            pre = s.get("precpu_stats") or {}
            pre_total = (pre.get("cpu_usage") or {}).get("total_usage")
            pre_system = pre.get("system_cpu_usage")
            if pre_total is not None and pre_system:
                prev = (pre_total, pre_system, now)
        if prev is not None and system:
            d_total = total - prev[0]
            d_system = system - prev[1]
            if d_system > 0 and d_total >= 0:
                cpu_pct = d_total / d_system * online * 100.0
        if system:
            self._prev_cpu[container] = (total, system, now)

        mem = s.get("memory_stats") or {}
        m_usage = mem.get("usage")
        m_limit = mem.get("limit")
        if m_usage is None or not m_limit:
            return None
        ms = mem.get("stats") or {}
        # cgroup v2 reports inactive_file, v1 total_inactive_file; the CLI
        # subtracts it so page cache does not count as container memory.
        cache = ms.get("inactive_file", ms.get("total_inactive_file", 0)) or 0
        used = max(0, int(m_usage) - int(cache))
        return {
            "cpu": f"{cpu_pct:.2f}%",
            "mem_usage": f"{fmt_bytes_docker(used)} / {fmt_bytes_docker(m_limit)}",
            "mem_pct": f"{used / m_limit * 100.0:.2f}%",
            "cpu_pct": round(cpu_pct, 2),
            "mem_used_bytes": used,
            "mem_limit_bytes": int(m_limit),
        }

    def logs(self, container: str, tail: int = 100) -> tuple[bool, str]:
        try:
            status, raw = self.request(
                "GET", f"/containers/{urllib.parse.quote(container)}/logs"
                       f"?stdout=1&stderr=1&timestamps=1&tail={int(tail)}", timeout=5)
        except (OSError, http.client.HTTPException) as e:
            return False, str(e) or e.__class__.__name__
        if status != 200:
            return False, f"logs failed: HTTP {status}"
        out, err = demux_stream(raw)
        # Ollama logs to stderr; merge both streams in arrival order is not
        # possible after demux, so stderr (the log) comes last — the tail
        # ordering is what matters and each stream is internally ordered.
        return True, (out + err).decode("utf-8", "replace")

    def exec(self, container: str, cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
        """Create + start an exec, demux its output, read the exit code."""
        try:
            status, created = self._json(
                "POST", f"/containers/{urllib.parse.quote(container)}/exec",
                {"AttachStdout": True, "AttachStderr": True, "Tty": False, "Cmd": cmd},
                timeout=timeout)
            if status not in (200, 201) or not isinstance(created, dict) or not created.get("Id"):
                return False, f"exec create failed: HTTP {status}"
            exec_id = created["Id"]
            status, raw = self.request("POST", f"/exec/{exec_id}/start",
                                       {"Detach": False, "Tty": False}, timeout=timeout)
            if status != 200:
                return False, f"exec start failed: HTTP {status}"
            out, err = demux_stream(raw)
            status, info = self._json("GET", f"/exec/{exec_id}/json", timeout=timeout)
        except (OSError, http.client.HTTPException) as e:
            return False, str(e) or e.__class__.__name__
        code = info.get("ExitCode") if isinstance(info, dict) else None
        text_out = out.decode("utf-8", "replace").strip()
        text_err = err.decode("utf-8", "replace").strip()
        if code in (0, None):
            return True, text_out
        return False, text_err or text_out


# ── CLI fallback ──────────────────────────────────────────────────────────────

class DockerCli(ContainerRuntime):
    """`docker` or `podman` subprocesses (the pre-0.5.0 path)."""

    def __init__(self, runner: Runner, binary: str = "docker"):
        self._run = runner
        self.binary = binary
        self.name = f"{os.path.basename(binary)}-cli"

    def inspect(self, container: str) -> dict | None:
        fmt = ("{{.State.Status}}\t{{.State.StartedAt}}\t"
               "{{.HostConfig.NanoCpus}}\t{{.HostConfig.CpuQuota}}\t"
               "{{.HostConfig.CpuPeriod}}\t{{.State.Pid}}\t{{.Config.Image}}\t"
               "{{json .Config.Env}}")
        ok, out = self._run([self.binary, "inspect", "--format", fmt, container])
        if not ok:
            return None
        parts = out.split("\t")
        if len(parts) < 2:
            return None

        def num(i: int) -> int:
            try:
                return int(parts[i]) if len(parts) > i and parts[i].strip() else 0
            except ValueError:
                return 0

        env: list[str] = []
        if len(parts) > 7:
            try:
                env = list(json.loads(parts[7]) or [])
            except ValueError:
                env = []
        return {
            "status": parts[0].strip(),
            "started_at": parts[1].strip(),
            "pid": num(5) or None,
            "nano_cpus": num(2),
            "cpu_quota": num(3),
            "cpu_period": num(4),
            "env": env,
            "image": parts[6].strip() if len(parts) > 6 else "",
        }

    def stats(self, container: str) -> dict | None:
        fmt = "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        ok, out = self._run([self.binary, "stats", "--no-stream", "--format", fmt, container])
        if not ok or not out:
            return None
        parts = out.split("\t")
        if len(parts) < 3:
            return None
        return {"cpu": parts[0].strip(), "mem_usage": parts[1].strip(),
                "mem_pct": parts[2].strip()}

    def exec(self, container: str, cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
        return self._run([self.binary, "exec", container, *cmd], timeout=timeout)

    def logs(self, container: str, tail: int = 100) -> tuple[bool, str]:
        # The container's stderr (where Ollama logs) comes out on *our*
        # stderr, so the runner must merge the two streams.
        return self._run([self.binary, "logs", "--timestamps", "--tail", str(int(tail)),
                          container], timeout=5, merge_stderr=True)


# ── discovery ─────────────────────────────────────────────────────────────────

def detect_runtime(runner: Runner, prefer: str = "auto",
                   env: dict[str, str] | None = None,
                   candidates: list[str] | None = None) -> ContainerRuntime | None:
    """Pick a runtime: API socket first, then a CLI on PATH; None if neither.

    prefer: auto | api | cli. DOCKER_HOST wins over the candidate list, as it
    does for the docker CLI itself; a TLS tcp endpoint (DOCKER_TLS_VERIFY) is
    left to the CLI, which knows where the certificates are.
    """
    env = os.environ if env is None else env
    if prefer in ("auto", "api"):
        hosts: list[str] = []
        dh = env.get("DOCKER_HOST", "").strip()
        if dh and not env.get("DOCKER_TLS_VERIFY"):
            hosts.append(dh)
        elif not dh:
            hosts += [f"unix://{p}" for p in (candidates or default_socket_candidates())
                      if os.path.exists(p)]
        for h in hosts:
            try:
                api = DockerApi(h)
            except ValueError:
                continue
            if api.probe():
                return api
    if prefer in ("auto", "cli"):
        for binary in ("docker", "podman"):
            if shutil.which(binary):
                return DockerCli(runner, binary)
    return None
