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
