from datetime import datetime, timedelta, timezone

import pytest

import mtop


def test_normalize_api_url_adds_scheme_and_strips_slash():
    assert mtop.normalize_api_url("gpu-rig:11434") == "http://gpu-rig:11434"
    assert mtop.normalize_api_url("http://localhost:11434/") == "http://localhost:11434"
    assert mtop.normalize_api_url("  https://x/  ") == "https://x"
    assert mtop.normalize_api_url("") == ""


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:11434", True),
    ("http://127.0.0.1:11434", True),
    ("http://127.1.2.3:11434", True),
    ("http://0.0.0.0:11434", True),
    ("http://[::1]:11434", True),
    ("http://gpu-rig:11434", False),
    ("http://192.168.1.10:11434", False),
])
def test_is_loopback_url(url, expected):
    assert mtop.is_loopback_url(url) is expected


def test_api_port_defaults():
    assert mtop.api_port("http://gpu-rig:12345") == 12345
    assert mtop.api_port("http://gpu-rig") == 11434
    assert mtop.api_port("https://gpu-rig") == 443


class TestRelativeTime:
    def _iso(self, delta: timedelta) -> str:
        return (datetime.now(timezone.utc) + delta).isoformat()

    def test_empty(self):
        assert mtop.relative_time("") == "—"

    def test_left_and_ago(self):
        assert mtop.relative_time(self._iso(timedelta(minutes=4, seconds=32))).endswith(" left")
        assert mtop.relative_time(self._iso(timedelta(hours=-2))).endswith(" ago")

    def test_nanoseconds_and_zulu(self):
        # Ollama emits 9 fractional digits and sometimes a trailing Z.
        s = self._iso(timedelta(hours=3))[:19] + ".123456789+00:00"
        assert mtop.relative_time(s) == "2h 59m left" or mtop.relative_time(s) == "3h 0m left"
        z = self._iso(timedelta(hours=3))[:19] + "Z"
        assert mtop.relative_time(z).endswith(" left")

    def test_keep_alive_minus_one_is_forever(self):
        # keep_alive -1: Ollama schedules expiry ~292 years out; `ollama ps` says Forever.
        assert mtop.relative_time("2318-09-13T00:00:00Z") == "forever"

    def test_go_zero_time_is_never(self):
        assert mtop.relative_time("0001-01-01T00:00:00Z") == "never"

    def test_garbage_falls_back_to_prefix(self):
        assert mtop.relative_time("not-a-date") == "not-a-date"


def test_fmt_duration():
    assert mtop.fmt_duration(65) == "1m 5s"
    assert mtop.fmt_duration(3600 * 3 + 120) == "3h 2m"
    assert mtop.fmt_duration(86400 * 2 + 3600 * 5) == "2d 5h"


def test_to_float_tolerates_na():
    assert mtop.to_float("[N/A]") is None
    assert mtop.to_float(" 42.5 ") == 42.5
    assert mtop.to_float(None) is None


@pytest.mark.parametrize("size,vram,label", [
    (100, 0, "100% CPU"),
    (100, 100, "100% GPU"),
    (100, 45, "55%/45% CPU/GPU"),
    (0, 0, "100% CPU"),
    (10, 20, "Unknown"),
    (None, None, "100% CPU"),
])
def test_processor_label_matches_ollama_ps(size, vram, label):
    assert mtop.processor_label(size, vram) == label


@pytest.mark.parametrize("value,expected", [
    ("infinity", None),
    ("", None),
    ("2s", 2.0),
    ("500ms", 0.5),
    ("1.5s", 1.5),
    ("1min 30s", 90.0),
])
def test_parse_systemd_cpu_quota(value, expected):
    assert mtop.parse_systemd_cpu_quota(value) == expected


def test_parse_systemd_show():
    out = "ActiveState=active\nSubState=running\nMainPID=3035\nCPUQuotaPerSecUSec=4s\n"
    assert mtop.parse_systemd_show(out) == ("running", 3035, 4.0)
    assert mtop.parse_systemd_show("ActiveState=inactive\nMainPID=0\n") is None
    assert mtop.parse_systemd_show("ActiveState=activating\nMainPID=0\n") == ("starting", 0, None)
    assert mtop.parse_systemd_show("ActiveState=failed\nMainPID=0\n") == ("failed", 0, None)
    assert mtop.parse_systemd_show("") is None


def test_host_cpu_count_positive():
    assert mtop.host_cpu_count() >= 1


def test_inference_env_filters_sorts_and_masks():
    env = mtop.inference_env([
        "PATH=/usr/bin", "OLLAMA_KEEP_ALIVE=24h", "OLLAMA_FLASH_ATTENTION=1",
        "CUDA_VISIBLE_DEVICES=0,1", "HSA_OVERRIDE_GFX_VERSION=11.0.0",
        "OLLAMA_API_KEY=sk-abc", "OLLAMA_AUTH_TOKEN=", "HOME=/root", "GGML_CUDA_NO_PINNED=1",
        "garbage-without-equals",
    ])
    assert list(env) == ["CUDA_VISIBLE_DEVICES", "GGML_CUDA_NO_PINNED", "HSA_OVERRIDE_GFX_VERSION",
                         "OLLAMA_API_KEY", "OLLAMA_AUTH_TOKEN", "OLLAMA_FLASH_ATTENTION",
                         "OLLAMA_KEEP_ALIVE"]
    assert env["OLLAMA_API_KEY"] == "••••"
    assert env["OLLAMA_AUTH_TOKEN"] == ""            # empty stays empty
    assert env["OLLAMA_KEEP_ALIVE"] == "24h"


def test_parse_systemd_environment_quotes():
    assert mtop.parse_systemd_environment('Environment=A=1 "B=x y" OLLAMA_HOST=0.0.0.0') == \
        ["A=1", "B=x y", "OLLAMA_HOST=0.0.0.0"]
    assert mtop.parse_systemd_environment("Environment=") == []


def test_link_runners_to_gpus_both_ways():
    runners = [{"pid": 10, "model_name": "a:7b"}, {"pid": 11, "digest": "deadbeefcafe0000"},
               {"pid": 12}]
    gpus = [{"vendor": "nvidia", "index": 0, "procs": [{"pid": 10, "mem_mib": 4000},
                                                        {"pid": 999, "mem_mib": 100}]},
            {"vendor": "nvidia", "index": 1, "procs": [{"pid": 10, "mem_mib": 4000},
                                                        {"pid": 11, "mem_mib": None}]}]
    mtop.link_runners_to_gpus(runners, gpus)
    assert runners[0]["gpu"] == ["nvidia:0", "nvidia:1"] and runners[0]["gpu_mem_mib"] == 8000
    assert runners[1]["gpu"] == ["nvidia:1"] and "gpu_mem_mib" not in runners[1]
    assert "gpu" not in runners[2]
    assert gpus[0]["procs"][0]["model"] == "a:7b"
    assert "model" not in gpus[0]["procs"][1]           # unknown pid
    assert gpus[1]["procs"][1]["model"] == "deadbeefcafe"
    # idempotent across cycles: re-linking does not accumulate
    mtop.link_runners_to_gpus(runners, gpus)
    assert runners[0]["gpu"] == ["nvidia:0", "nvidia:1"] and runners[0]["gpu_mem_mib"] == 8000
    mtop.link_runners_to_gpus(None, gpus)
    mtop.link_runners_to_gpus(runners, None)


# ── endpoints, auth, TLS ─────────────────────────────────────────────────────

import http.server  # noqa: E402
import ssl  # noqa: E402
import threading  # noqa: E402


@pytest.mark.parametrize("value,expected", [
    ("http://gpu-rig:11434", (None, "http://gpu-rig:11434")),
    ("rig=http://gpu-rig:11434", ("rig", "http://gpu-rig:11434")),
    ("gpu-rig:11434", (None, "gpu-rig:11434")),
    ("http://x/?a=b", (None, "http://x/?a=b")),
    ("my.box=https://u:p@x", ("my.box", "https://u:p@x")),
    ("bad label=http://x", (None, "bad label=http://x")),
])
def test_parse_endpoint_arg(value, expected):
    assert mtop.parse_endpoint_arg(value) == expected


def test_split_userinfo():
    url, h = mtop.split_userinfo("https://marek:s3cret%40x@gpu-rig:8443/ollama")
    assert url == "https://gpu-rig:8443/ollama"
    assert h == {"Authorization": "Basic bWFyZWs6czNjcmV0QHg="}      # marek:s3cret@x
    assert mtop.split_userinfo("http://gpu-rig:11434") == ("http://gpu-rig:11434", {})
    url6, h6 = mtop.split_userinfo("http://u:p@[::1]:11434")
    assert url6 == "http://[::1]:11434" and "Authorization" in h6


def test_parse_header_arg():
    assert mtop.parse_header_arg("Authorization: Bearer abc:def") == \
        ("Authorization", "Bearer abc:def")
    assert mtop.parse_header_arg("X-Empty:") == ("X-Empty", "")
    with pytest.raises(ValueError):
        mtop.parse_header_arg("no-colon")
    with pytest.raises(ValueError):
        mtop.parse_header_arg(": value")


def test_make_ssl_context():
    assert mtop.make_ssl_context() is None
    ctx = mtop.make_ssl_context(insecure=True)
    assert ctx.verify_mode == ssl.CERT_NONE and ctx.check_hostname is False
    with pytest.raises(FileNotFoundError):
        mtop.make_ssl_context(cacert="/nonexistent/ca.pem")


def test_endpoint_label_headers_and_describe():
    ep = mtop.Endpoint("rig=https://u:p@gpu-rig:8443/", {"X-Api": "1"}, insecure=True)
    assert ep.url == "https://gpu-rig:8443" and ep.label == "rig"
    assert ep.headers["X-Api"] == "1" and ep.headers["Authorization"].startswith("Basic ")
    assert ep.describe() == {"label": "rig", "url": "https://gpu-rig:8443",
                             "auth": True, "tls": "insecure"}
    plain = mtop.Endpoint("gpu-rig:11434")
    assert plain.label == "gpu-rig:11434" and plain.url == "http://gpu-rig:11434"
    assert plain.describe()["tls"] == "default" and plain.describe()["auth"] is False


@pytest.fixture
def auth_server():
    """A stand-in for Ollama behind a proxy: 401 without the expected header."""
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append(dict(self.headers))
            if self.headers.get("Authorization") != "Bearer sekret":
                self.send_response(401, "Unauthorized")
                self.end_headers()
                return
            body = json.dumps({"models": [{"name": "remote:7b", "size": 1, "size_vram": 1,
                                           "context_length": 1}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    finally:
        srv.shutdown()
        srv.server_close()


import json  # noqa: E402


def test_http_get_json_sends_headers_and_reports_http_errors(auth_server):
    base, seen = auth_server
    ok, err = mtop.http_get_json(base + "/api/ps")
    assert ok is False and err == "HTTP 401 Unauthorized"
    ok, data = mtop.http_get_json(base + "/api/ps", headers={"Authorization": "Bearer sekret"})
    assert ok is True and data["models"][0]["name"] == "remote:7b"
    assert seen[-1]["Accept"] == "application/json"


def test_endpoint_get_json_uses_its_headers(auth_server):
    base, _ = auth_server
    ep = mtop.Endpoint("proxy=" + base, {"Authorization": "Bearer sekret"})
    ok, data = ep.get_json("/api/ps")
    assert ok and data["models"]
    assert mtop.Endpoint(base).get_json("/api/ps") == (False, "HTTP 401 Unauthorized")
