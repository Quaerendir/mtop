"""Container runtimes: the Engine API against a real unix-socket HTTP server
(stdlib http.server bound to AF_UNIX), the CLI fallback, and discovery."""

import http.server
import json
import socket
import socketserver
import struct
import threading

import pytest

from mtop import container as ct

GIB = 1024 ** 3

INSPECT = {
    "State": {"Status": "running", "StartedAt": "2026-09-13T00:00:00.123456789Z", "Pid": 3137},
    "HostConfig": {"NanoCpus": 0, "CpuQuota": 0, "CpuPeriod": 0},
    "Config": {"Image": "ollama/ollama:latest",
               "Env": ["OLLAMA_KEEP_ALIVE=24h", "OLLAMA_FLASH_ATTENTION=1"]},
}


def stats_json(total, system, usage, cache, limit=121 * GIB, online=20, pre=None, v1=False):
    d = {
        "cpu_stats": {"cpu_usage": {"total_usage": total}, "system_cpu_usage": system,
                      "online_cpus": online},
        "precpu_stats": pre or {"cpu_usage": {"total_usage": 0}},
        "memory_stats": {"usage": usage, "limit": limit,
                         "stats": ({"total_inactive_file": cache} if v1
                                   else {"inactive_file": cache})},
    }
    return d


# ── Engine API via a fake daemon on a unix socket ─────────────────────────────

class _UnixHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "unix", 0


def make_handler(routes, log):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):        # AF_UNIX client_address is '' — silence
            pass

        def _serve(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            log.append((self.command, self.path, body))
            key = (self.command, self.path.split("?")[0])
            if key not in routes:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"message":"no such container"}')
                return
            status, payload = routes[key]
            if callable(payload):
                payload = payload(body)
            self.send_response(status)
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload).encode()
                self.send_header("Content-Type", "application/json")
            else:
                self.send_header("Content-Type", "application/vnd.docker.raw-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = _serve
    return Handler


def frame(kind, payload: bytes) -> bytes:
    return bytes([kind, 0, 0, 0]) + struct.pack(">I", len(payload)) + payload


@pytest.fixture
def daemon(sock_dir):
    """Start a fake Engine API on a unix socket; yields (DockerApi, routes, log)."""
    sock = str(sock_dir / "d.sock")
    routes, log = {}, []
    routes[("GET", "/version")] = (200, {"Version": "29.2.1", "ApiVersion": "1.53",
                                         "Platform": {"Name": "Docker Engine - Community"}})
    routes[("GET", "/containers/ollama/json")] = (200, INSPECT)
    srv = _UnixHTTPServer(sock, make_handler(routes, log))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield ct.DockerApi(f"unix://{sock}", clock=lambda: 100.0), routes, log
    finally:
        srv.shutdown()
        srv.server_close()


class TestDockerApi:
    def test_probe_and_name(self, daemon):
        api, routes, _ = daemon
        assert api.probe() is True
        assert api.name == "docker-api"
        routes[("GET", "/version")] = (200, {"Components": [{"Name": "Podman Engine"}]})
        api.probe()
        assert api.name == "podman-api"

    def test_probe_false_on_dead_socket(self, tmp_path):
        api = ct.DockerApi(f"unix://{tmp_path}/nope.sock")
        assert api.probe() is False
        assert api.inspect("ollama") is None
        assert api.stats("ollama") is None
        ok, msg = api.exec("ollama", ["true"])
        assert ok is False and msg

    def test_inspect_normalized(self, daemon):
        api, _, _ = daemon
        info = api.inspect("ollama")
        assert info["status"] == "running" and info["pid"] == 3137
        assert info["started_at"].startswith("2026-09-13")
        assert info["nano_cpus"] == 0 and info["cpu_quota"] == 0
        assert info["image"] == "ollama/ollama:latest"
        assert "OLLAMA_KEEP_ALIVE=24h" in info["env"]
        assert api.inspect("missing") is None

    def test_stats_one_shot_delta_from_own_baseline(self, daemon):
        api, routes, log = daemon
        samples = [stats_json(1_000_000_000, 100_000_000_000, 2 * GIB, GIB // 2),
                   stats_json(3_000_000_000, 110_000_000_000, 2 * GIB, GIB // 2)]
        routes[("GET", "/containers/ollama/stats")] = (200, lambda body: samples.pop(0))
        first = api.stats("ollama")
        assert first["cpu"] == "0.00%"                      # no precpu, no baseline
        assert first["mem_used_bytes"] == 2 * GIB - GIB // 2
        assert first["mem_usage"] == "1.5GiB / 121GiB"
        assert first["mem_pct"] == "1.24%"
        second = api.stats("ollama")
        # 2e9 ns of container CPU over 1e10 ns of system time on 20 cores
        assert second["cpu"] == "400.00%" and second["cpu_pct"] == 400.0
        assert "one-shot=true" in log[-1][1] and "stream=false" in log[-1][1]

    def test_stats_uses_precpu_when_no_baseline(self, daemon):
        api, routes, _ = daemon
        pre = {"cpu_usage": {"total_usage": 500_000_000}, "system_cpu_usage": 90_000_000_000}
        routes[("GET", "/containers/ollama/stats")] = (
            200, stats_json(1_500_000_000, 100_000_000_000, GIB, 0, pre=pre, online=4))
        s = api.stats("ollama")
        assert s["cpu"] == "40.00%"

    def test_stats_cgroup_v1_cache_key(self, daemon):
        api, routes, _ = daemon
        routes[("GET", "/containers/ollama/stats")] = (
            200, stats_json(1, 1, 3 * GIB, GIB, v1=True))
        assert api.stats("ollama")["mem_used_bytes"] == 2 * GIB

    def test_stats_missing_memory_is_none(self, daemon):
        api, routes, _ = daemon
        routes[("GET", "/containers/ollama/stats")] = (200, {"cpu_stats": {"cpu_usage": {}}})
        assert api.stats("ollama") is None

    def test_exec_roundtrip_demuxed(self, daemon):
        api, routes, log = daemon
        routes[("POST", "/containers/ollama/exec")] = (201, {"Id": "abc"})
        routes[("POST", "/exec/abc/start")] = (
            200, frame(1, b"NAME  ID\n") + frame(2, b"warn\n") + frame(1, b"bielik  b669\n"))
        routes[("GET", "/exec/abc/json")] = (200, {"ExitCode": 0})
        ok, out = api.exec("ollama", ["ollama", "ps"])
        assert ok is True
        assert out == "NAME  ID\nbielik  b669"
        create = json.loads(log[-3][2])
        assert create["Cmd"] == ["ollama", "ps"] and create["AttachStderr"] is True

    def test_exec_nonzero_exit_reports_stderr(self, daemon):
        api, routes, _ = daemon
        routes[("POST", "/containers/ollama/exec")] = (201, {"Id": "x"})
        routes[("POST", "/exec/x/start")] = (200, frame(2, b"sh: nvidia-smi: not found\n"))
        routes[("GET", "/exec/x/json")] = (200, {"ExitCode": 127})
        ok, out = api.exec("ollama", ["nvidia-smi"])
        assert ok is False and "not found" in out

    def test_exec_create_failure(self, daemon):
        api, _, _ = daemon
        ok, msg = api.exec("ghost", ["true"])
        assert ok is False and "404" in msg

    def test_tcp_host_parses(self):
        api = ct.DockerApi("tcp://10.0.0.5:2375")
        assert api._tcp == ("10.0.0.5", 2375) and api._path is None
        with pytest.raises(ValueError):
            ct.DockerApi("ssh://box")


def test_demux_unframed_passthrough():
    assert ct.demux_stream(b"plain tty output\n") == (b"plain tty output\n", b"")
    assert ct.demux_stream(b"") == (b"", b"")


@pytest.mark.parametrize("n,s", [
    (0, "0B"), (1023, "1023B"), (384.8 * 1024 ** 2, "384.8MiB"),
    (1.536 * GIB, "1.536GiB"), (121.7 * GIB, "121.7GiB"), (2 * 1024 ** 4, "2TiB"),
])
def test_fmt_bytes_docker(n, s):
    assert ct.fmt_bytes_docker(n) == s


# ── CLI fallback ──────────────────────────────────────────────────────────────

class TestDockerCli:
    def _run(self, table):
        calls = []

        def run(cmd, timeout=5, env=None):
            calls.append(cmd)
            return table.get(cmd[1], (False, "boom"))
        return run, calls

    def test_inspect(self):
        env = json.dumps(["OLLAMA_HOST=0.0.0.0"])
        run, calls = self._run({"inspect": (
            True, f"running\t2026-09-13T00:00:00Z\t4000000000\t0\t100000\t3137\timg:1\t{env}")})
        info = ct.DockerCli(run, "podman").inspect("ollama")
        assert calls[0][:2] == ["podman", "inspect"]
        assert info["status"] == "running" and info["pid"] == 3137
        assert info["nano_cpus"] == 4_000_000_000 and info["cpu_period"] == 100000
        assert info["image"] == "img:1" and info["env"] == ["OLLAMA_HOST=0.0.0.0"]
        assert ct.DockerCli(run).name == "docker-cli"
        assert ct.DockerCli(run, "podman").name == "podman-cli"

    def test_inspect_failure(self):
        run, _ = self._run({})
        assert ct.DockerCli(run).inspect("ollama") is None

    def test_stats_and_exec(self):
        run, calls = self._run({"stats": (True, "0.08%\t1.536GiB / 121.7GiB\t1.26%"),
                                "exec": (True, "NAME")})
        cli = ct.DockerCli(run)
        assert cli.stats("ollama") == {"cpu": "0.08%", "mem_usage": "1.536GiB / 121.7GiB",
                                       "mem_pct": "1.26%"}
        assert cli.exec("ollama", ["ollama", "ps"]) == (True, "NAME")
        assert calls[-1] == ["docker", "exec", "ollama", "ollama", "ps"]


# ── discovery ─────────────────────────────────────────────────────────────────

class TestDetect:
    def test_socket_wins_over_cli(self, daemon, monkeypatch):
        api, _, _ = daemon
        monkeypatch.setattr(ct.shutil, "which", lambda b: "/usr/bin/" + b)
        rt = ct.detect_runtime(lambda *a, **k: (False, ""), "auto", env={},
                               candidates=[api._path])
        assert isinstance(rt, ct.DockerApi) and rt.name == "docker-api"

    def test_docker_host_env_is_honoured(self, daemon):
        api, _, _ = daemon
        rt = ct.detect_runtime(lambda *a, **k: (False, ""), "api",
                               env={"DOCKER_HOST": f"unix://{api._path}"}, candidates=[])
        assert isinstance(rt, ct.DockerApi)

    def test_tls_docker_host_goes_to_cli(self, monkeypatch):
        monkeypatch.setattr(ct.shutil, "which",
                            lambda b: "/usr/bin/docker" if b == "docker" else None)
        rt = ct.detect_runtime(lambda *a, **k: (False, ""), "auto",
                               env={"DOCKER_HOST": "tcp://x:2376", "DOCKER_TLS_VERIFY": "1"},
                               candidates=[])
        assert isinstance(rt, ct.DockerCli) and rt.binary == "docker"

    def test_falls_back_to_podman_cli(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ct.shutil, "which",
                            lambda b: "/usr/bin/podman" if b == "podman" else None)
        rt = ct.detect_runtime(lambda *a, **k: (False, ""), "auto", env={},
                               candidates=[str(tmp_path / "none.sock")])
        assert isinstance(rt, ct.DockerCli) and rt.name == "podman-cli"

    def test_nothing_available(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ct.shutil, "which", lambda b: None)
        assert ct.detect_runtime(lambda *a, **k: (False, ""), "auto", env={},
                                 candidates=[str(tmp_path / "none.sock")]) is None
        assert ct.detect_runtime(lambda *a, **k: (False, ""), "cli", env={}) is None

    def test_default_candidates_include_rootless(self, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        c = ct.default_socket_candidates()
        assert c[0] == "/var/run/docker.sock"
        assert "/run/user/1000/podman/podman.sock" in c
        assert "/run/podman/podman.sock" in c
