"""Prometheus exposition, NDJSON watch loop, atomic output files."""

import argparse
import json
import math
import re
from datetime import datetime, timedelta, timezone

import pytest

import mtop
from mtop import export

GIB = 1024 ** 3

SAMPLE_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? (-?[0-9.e+-]+|[+-]Inf|NaN)$')


def _lines(text):
    return [ln for ln in text.splitlines() if ln and not ln.startswith("#")]


def _samples(text):
    out = {}
    for ln in _lines(text):
        m = SAMPLE_RE.match(ln)
        assert m, f"malformed sample line: {ln!r}"
        out.setdefault(m.group(1), []).append((m.group(2) or "", m.group(3)))
    return out


def test_parse_iso_variants():
    t = export.parse_iso("2026-09-13T20:53:54.639194927Z")
    assert t.tzinfo is not None and t.microsecond == 639194
    assert export.parse_iso("2026-09-13T20:53:54+02:00").utcoffset() == timedelta(hours=2)
    assert export.parse_iso("2026-09-13T20:53:54.5Z").microsecond == 500000
    assert export.parse_iso("2026-09-13T20:53:54").tzinfo == timezone.utc
    assert export.parse_iso("") is None and export.parse_iso("nope") is None


@pytest.mark.parametrize("text,expected", [
    ("1.536GiB", int(1.536 * GIB)), ("427MiB", 427 << 20), ("121.7GiB", int(121.7 * GIB)),
    ("14 GB", 14 * 10 ** 9), ("512B", 512), ("0B", 0), ("2TiB", 2 << 40), ("3.5G", int(3.5 * GIB)),
    ("garbage", None), ("", None), ("12XB", None),
])
def test_parse_size(text, expected):
    assert export.parse_size(text) == expected


def test_helpers_num_pct():
    assert export.num("[N/A]") is None and export.num("12.5") == 12.5 and export.num("inf") is None
    assert export.pct("12.50%") == 12.5 and export.pct(None) is None


FULL_SNAP = {
    "mode": "docker", "runtime": "docker-api", "container": "ollama", "status": "running",
    "api_url": "http://localhost:11434", "cpu_limit": 20.0, "pid": 3137, "uptime_sec": 72000.5,
    "res_stats": {"cpu": "2.07%", "mem_usage": "426.8MiB / 121.7GiB", "mem_pct": "0.34%",
                  "cpu_pct": 2.07, "mem_used_bytes": 447582208, "mem_limit_bytes": 130658811904},
    "models_ok": True,
    "endpoints": [
        {"label": "localhost:11434", "url": "http://localhost:11434", "models_ok": True,
         "version": "0.33.2",
         "models": [{"name": "bielik-ocr-64k:latest", "size": 14803397507,
                     "size_vram": 14803397507, "context_length": 65536,
                     "expires_at": "2318-01-01T00:00:00Z",
                     "details": {"family": "mistral3", "quantization_level": "Q4_K_M"}},
                    {"name": 'odd"name\\x', "size": 10, "size_vram": 5, "context_length": 1,
                     "expires_at": "SET-IN-TEST"}]},
        {"label": "dead", "url": "http://10.0.0.9:11434", "models_ok": False, "models": [],
         "models_err": "timed out", "version": None},
    ],
    "runners": [{"pid": 1637884, "model_name": "bielik-ocr-64k:latest", "ctx": "65536",
                 "batch": "2048", "flash_attn": "on", "kv_k": "f16", "kv_v": "f16",
                 "engine": "llama", "load_mode": "dio", "rss": 1 * GIB, "vram": 14803397507,
                 "gpu": ["nvidia:0"], "gpu_mem_mib": 15097}],
    "gpus": [{"vendor": "nvidia", "index": 0, "name": "NVIDIA GB10", "util": "0",
              "mem_used": "19517", "mem_total": "124605", "temp": "40", "power": "10.71",
              "unified": True,
              "procs": [{"pid": 1637884, "mem_mib": 15097, "model": "bielik-ocr-64k:latest"}]},
             {"vendor": "amd", "index": 0, "name": "Radeon", "util": "[N/A]", "mem_used": "100",
              "mem_total": "16368", "temp": "N/A", "gtt_used": "50", "gtt_total": "32768"}],
    "server": {"version": "0.33.2", "env": {}, "env_source": "container"},
}


def test_prometheus_text_full_snapshot():
    snap = json.loads(json.dumps(FULL_SNAP))
    snap["endpoints"][0]["models"][1]["expires_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat()
    text = export.prometheus_text(snap, "9.9.9", now=1_700_000_000.0)
    s = _samples(text)
    assert s["mtop_up"] == [("", "1")]
    assert s["mtop_snapshot_timestamp_seconds"] == [("", "1700000000")]
    assert ('{mode="docker",container="ollama",version="9.9.9",runtime="docker-api",'
            'api_url="http://localhost:11434"}', "1") in s["mtop_info"]
    # endpoints
    assert ('{endpoint="localhost:11434",url="http://localhost:11434"}', "1") in s["mtop_api_up"]
    assert ('{endpoint="dead",url="http://10.0.0.9:11434"}', "0") in s["mtop_api_up"]
    assert len(s["mtop_models_loaded"]) == 1              # not emitted for the dead endpoint
    assert s["mtop_models_loaded"][0][1] == "2"
    assert len(s["mtop_ollama_info"]) == 1
    # models: forever -> +Inf, near -> ~90, label escaping
    exp = dict(s["mtop_model_expires_seconds"])
    inf = [v for k, v in exp.items() if "bielik" in k]
    assert inf == ["+Inf"]
    near = [float(v) for k, v in exp.items() if "odd" in k]
    assert 3500 <= near[0] <= 3600
    assert any('model="odd\\"name\\\\x"' in k for k in exp)
    assert any('family="mistral3",quantization="Q4_K_M"' in k
               for k in dict(s["mtop_model_vram_bytes"]))
    # server
    assert s["mtop_server_cpu_percent"] == [('{mode="docker",container="ollama"}', "2.07")]
    assert s["mtop_server_memory_bytes"] == [('{mode="docker",container="ollama",kind="cgroup"}',
                                              "447582208")]
    assert s["mtop_server_memory_limit_bytes"][0][1] == "130658811904"
    assert s["mtop_server_uptime_seconds"][0][1] == "72000.5"
    assert "mtop_server_processes" not in s                # docker stats have no procs
    # runners
    info = s["mtop_runner_info"][0][0]
    for frag in ('pid="1637884"', 'model="bielik-ocr-64k:latest"', 'engine="llama"',
                 'kv_cache="f16"', 'load_mode="dio"', 'gpu="nvidia:0"', 'flash_attn="on"'):
        assert frag in info
    assert s["mtop_runner_gpu_memory_bytes"][0][1] == str(15097 << 20)
    assert s["mtop_runner_rss_bytes"][0][1] == str(GIB)
    # gpus: N/A dropped, MiB -> bytes, unified flag, procs
    assert ('{vendor="nvidia",index="0",name="NVIDIA GB10",unified="1"}', "1") in s["mtop_gpu_info"]
    assert dict(s["mtop_gpu_memory_used_bytes"])['{vendor="nvidia",index="0"}'] == str(19517 << 20)
    assert len(s["mtop_gpu_utilization_percent"]) == 1    # amd util [N/A] skipped
    assert len(s["mtop_gpu_temperature_celsius"]) == 1
    assert dict(s["mtop_gpu_gtt_total_bytes"])['{vendor="amd",index="0"}'] == str(32768 << 20)
    assert s["mtop_gpu_process_memory_bytes"][0][0].endswith('model="bielik-ocr-64k:latest"}')
    # every sample has HELP + TYPE exactly once
    for name in s:
        assert text.count(f"# HELP {name} ") == 1 and text.count(f"# TYPE {name} ") == 1
    assert text.endswith("\n")


def test_prometheus_text_minimal_and_legacy_shape():
    snap = {"mode": "api", "status": "api-only", "api_url": "http://x:1", "models_ok": False,
            "models": [], "models_err": "refused"}
    text = export.prometheus_text(snap, "1")
    s = _samples(text)
    assert s["mtop_up"] == [("", "0")]
    assert s["mtop_api_up"] == [('{endpoint="http://x:1",url="http://x:1"}', "0")]
    assert "mtop_server_cpu_percent" not in s and "mtop_gpu_info" not in s
    assert 'container=' not in text.split("mtop_info")[1].split("\n")[0]


def test_prometheus_cli_stats_strings_are_parsed():
    snap = {"mode": "docker", "status": "running", "container": "c", "models_ok": True,
            "res_stats": {"cpu": "0.08%", "mem_usage": "1.536GiB / 121.7GiB", "mem_pct": "1%"}}
    s = _samples(export.prometheus_text(snap, "1"))
    assert s["mtop_server_cpu_percent"][0][1] == "0.08"
    assert s["mtop_server_memory_bytes"][0][1] == str(int(1.536 * GIB))
    assert s["mtop_server_memory_limit_bytes"][0][1] == str(int(121.7 * GIB))


def test_fmt_values():
    assert export._fmt(math.inf) == "+Inf" and export._fmt(-math.inf) == "-Inf"
    assert export._fmt(3.0) == "3" and export._fmt(2.5) == "2.5" and export._fmt(1e20) == "1e+20"


# ── headless loop ────────────────────────────────────────────────────────────

class _Snaps:
    """Feeds a scripted sequence of snapshots to headless_main."""

    def __init__(self, monkeypatch, snaps, sleeps_before_interrupt):
        self.snaps = list(snaps)
        self.sleeps = 0
        self.limit = sleeps_before_interrupt
        monkeypatch.setattr(mtop.Collector, "collect", lambda c, now: dict(self.snaps.pop(0)))
        monkeypatch.setattr(mtop.Collector, "needs_second_sample", lambda c, s: False)

        def sleep(sec):
            self.sleeps += 1
            if self.sleeps > self.limit:
                raise KeyboardInterrupt
        monkeypatch.setattr(mtop.time, "sleep", sleep)


def _args(**kw):
    base = {"container": "ollama", "api_url": "http://localhost:11434", "interval": 1.0,
            "no_gpu": True, "mode": "api", "no_runners": False, "runtime": "auto",
            "no_env": False, "endpoints": None, "json": False, "prometheus": False,
            "watch": False, "output": None, "logs": False, "log_lines": 8}
    base.update(kw)
    return argparse.Namespace(**base)


UP = {"ts": 1.0, "mode": "api", "status": "api-only", "models_ok": True, "models": [],
      "api_url": "http://localhost:11434"}
DOWN = {**UP, "models_ok": False, "models_err": "refused"}


def test_json_oneshot_indented_and_exit_code(monkeypatch, capsys):
    _Snaps(monkeypatch, [DOWN], 0)
    assert mtop.headless_main(_args(json=True)) == 1
    out = capsys.readouterr().out
    assert out.startswith("{\n  ") and "ts" not in json.loads(out)


def test_json_watch_is_ndjson(monkeypatch, capsys):
    _Snaps(monkeypatch, [UP, DOWN, UP], 2)
    assert mtop.headless_main(_args(json=True, watch=True)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and all(ln.startswith("{") and "\n" not in ln for ln in lines)
    assert [json.loads(ln)["models_ok"] for ln in lines] == [True, False, True]


def test_prometheus_oneshot(monkeypatch, capsys):
    _Snaps(monkeypatch, [UP], 0)
    assert mtop.headless_main(_args(prometheus=True)) == 0
    out = capsys.readouterr().out
    assert "# TYPE mtop_up gauge" in out and "mtop_up 1" in out
    assert f'version="{mtop.__version__}"' in out


def test_prometheus_watch_replaces_file_atomically(monkeypatch, tmp_path):
    _Snaps(monkeypatch, [UP, DOWN], 1)
    target = tmp_path / "mtop.prom"
    seen = []
    real_replace = mtop.os.replace

    def replace(src, dst):
        seen.append((src, dst))
        real_replace(src, dst)
    monkeypatch.setattr(mtop.os, "replace", replace)
    assert mtop.headless_main(_args(prometheus=True, watch=True, output=str(target))) == 0
    assert len(seen) == 2 and all(dst == str(target) for _, dst in seen)
    assert all(src.startswith(str(target) + ".") for src, _ in seen)
    assert "mtop_up 0" in target.read_text()               # last write wins, no tmp left
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mtop.prom"]


def test_json_watch_appends_to_file(monkeypatch, tmp_path):
    _Snaps(monkeypatch, [UP, UP], 1)
    target = tmp_path / "out.ndjson"
    target.write_text('{"old":1}\n')
    assert mtop.headless_main(_args(json=True, watch=True, output=str(target))) == 0
    assert len(target.read_text().splitlines()) == 3


def test_second_sample_taken_once_in_watch(monkeypatch, capsys):
    snaps = _Snaps(monkeypatch, [UP, UP, UP], 1)
    calls = []
    monkeypatch.setattr(mtop.Collector, "needs_second_sample", lambda c, s: True)
    monkeypatch.setattr(mtop.Collector, "force_slow", lambda c: calls.append("slow"))
    assert mtop.headless_main(_args(json=True, watch=True)) == 0
    assert calls == ["slow"]                                 # only for the first snapshot
    assert snaps.sleeps == 2                                 # 1 window + 1 interval


def test_json_main_alias_defaults(monkeypatch, capsys):
    _Snaps(monkeypatch, [UP], 0)
    ns = _args()
    for a in ("prometheus", "watch", "output"):
        delattr(ns, a)
    assert mtop.json_main(ns) == 0
    assert json.loads(capsys.readouterr().out)["models_ok"] is True
