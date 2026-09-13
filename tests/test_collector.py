"""Collector and renderer behaviour with every I/O seam stubbed out."""

import argparse
import json

import pytest
from conftest import OLLAMA_ENGINE_ARGV, FakeWin

import mtop

SERVER_PID, RUNNER_PID = 4242, 4243
GIB = 1024 ** 3

API_PS = {
    "models": [{
        "name": "qwen2.5-coder:32b", "model": "qwen2.5-coder:32b",
        "size": 20 * GIB, "digest": "abc", "expires_at": "2318-01-01T00:00:00Z",
        "size_vram": 20 * GIB, "context_length": 32768,
        "details": {"parameter_size": "32.8B", "quantization_level": "Q4_K_M",
                    "family": "qwen2", "format": "gguf"},
    }]
}


@pytest.fixture
def local_host(monkeypatch):
    """A Linux host running `ollama serve` under systemd with one runner."""
    monkeypatch.setattr(mtop, "IS_LINUX", True)
    monkeypatch.setattr(mtop, "IS_DARWIN", False)
    monkeypatch.setattr(mtop, "systemd_ollama", lambda: ("running", SERVER_PID, None))
    monkeypatch.setattr(mtop, "find_ollama_pid", lambda port=None: SERVER_PID)
    monkeypatch.setattr(mtop, "proc_uptime_sec", lambda pid: 3700.0)
    monkeypatch.setattr(mtop, "total_ram_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(mtop, "process_tree", lambda root, children=None: [SERVER_PID, RUNNER_PID])
    monkeypatch.setattr(mtop, "read_proc_rss_bytes",
                        lambda pid: 64 << 20 if pid == SERVER_PID else 7 * GIB)
    monkeypatch.setattr(mtop, "read_proc_pss_bytes", lambda pid: None)   # other user
    monkeypatch.setattr(
        mtop, "read_proc_cmdline",
        lambda pid: OLLAMA_ENGINE_ARGV if pid == RUNNER_PID else ["ollama", "serve"])
    monkeypatch.setattr(mtop, "http_get_json", lambda url, timeout=5: (True, API_PS))

    ticks = {"n": 0}
    samples = [{SERVER_PID: 100, RUNNER_PID: 1000}, {SERVER_PID: 150, RUNNER_PID: 1200}]

    def cpu_ticks(pid):
        return samples[min(ticks["n"], 1)][pid]
    monkeypatch.setattr(mtop, "read_proc_cpu_ticks", cpu_ticks)

    def advance():
        ticks["n"] += 1
    return advance


def _collector(**kw):
    defaults = {"container": "ollama", "api_url": "http://localhost:11434",
                "interval": 1.0, "show_gpu": False, "mode": "local"}
    defaults.update(kw)
    return mtop.Collector(**defaults)


def test_local_snapshot_rolls_up_tree_and_matches_runner(local_host):
    c = _collector()
    snap = c.collect(100.0)
    assert snap["mode"] == "local" and snap["status"] == "running"
    assert snap["pid"] == SERVER_PID and snap["uptime"] == "1h 1m"
    stats = snap["res_stats"]
    assert stats["procs"] == 2 and stats["mem_kind"] == "rss"
    assert stats["mem_usage"].startswith("7.1GiB / 64.0GiB")
    assert stats["cpu"] == "0.00%"                     # first sample: no delta yet
    r = snap["runners"][0]
    assert r["pid"] == RUNNER_PID and r["engine"] == "ollama"
    assert r["model_name"] == "qwen2.5-coder:32b"      # matched on ctx 32768
    assert r["vram"] == 20 * GIB


def test_cpu_percent_needs_two_samples(local_host):
    advance = local_host
    c = _collector()
    c.collect(100.0)
    advance()
    c.force_slow()
    snap = c.collect(103.0)
    # 250 ticks over the (tiny) real wall-clock gap: anything > 0 proves the
    # delta path ran; the exact figure depends on CLK_TCK and timing.
    assert mtop.to_float(snap["res_stats"]["cpu"].rstrip("%")) > 0


def test_json_main_takes_second_cpu_sample_in_local_mode(local_host, monkeypatch, capsys):
    advance = local_host
    monkeypatch.setattr(mtop.time, "sleep", lambda s: advance())
    args = argparse.Namespace(container="ollama", api_url="http://localhost:11434",
                              interval=1.0, no_gpu=True, mode="local", no_runners=False,
                              runtime="auto", no_env=False)
    rc = mtop.json_main(args)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert "ts" not in out
    assert mtop.to_float(out["res_stats"]["cpu"].rstrip("%")) > 0
    assert out["runners"][0]["model_name"] == "qwen2.5-coder:32b"


def test_json_main_exit_code_when_api_down(local_host, monkeypatch, capsys):
    monkeypatch.setattr(mtop, "http_get_json", lambda url, timeout=5: (False, "refused"))
    args = argparse.Namespace(container="ollama", api_url="http://localhost:11434",
                              interval=1.0, no_gpu=True, mode="local", no_runners=False,
                              runtime="auto", no_env=False)
    assert mtop.json_main(args) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["models_ok"] is False and out["models_err"] == "refused"


def test_raw_ps_passes_api_url_to_cli(local_host, monkeypatch):
    seen = {}

    def run_cmd(cmd, timeout=5, env=None):
        seen["cmd"], seen["env"] = cmd, env
        return True, "NAME  ID  SIZE  PROCESSOR  UNTIL\n"
    monkeypatch.setattr(mtop, "run_cmd", run_cmd)
    c = _collector(api_url="http://127.0.0.1:11435", show_raw_ps=True)
    snap = c.collect(0.0)
    assert seen["cmd"] == ["ollama", "ps"]
    assert seen["env"]["OLLAMA_HOST"] == "http://127.0.0.1:11435"
    assert snap["raw_ps_ok"] is True


def test_systemd_cpu_quota_becomes_cpu_limit(local_host, monkeypatch):
    monkeypatch.setattr(mtop, "systemd_ollama", lambda: ("running", SERVER_PID, 4.0))
    assert _collector().collect(0.0)["cpu_limit"] == 4.0


def test_api_port_only_for_loopback():
    assert _collector(api_url="http://localhost:11434").api_port == 11434
    assert _collector(api_url="http://127.0.0.1:12000").api_port == 12000
    assert _collector(api_url="http://gpu-rig:11434").api_port is None


def test_auto_mode_upgrades_from_api_once_source_appears(monkeypatch):
    monkeypatch.setattr(mtop, "http_get_json", lambda url, timeout=5: (True, {"models": []}))
    monkeypatch.setattr(mtop, "IS_LINUX", True)
    probes = {"docker": False, "local": False}
    monkeypatch.setattr(mtop.Collector, "_detect_docker", lambda self: probes["docker"])
    monkeypatch.setattr(mtop.Collector, "_detect_local", lambda self: probes["local"])
    monkeypatch.setattr(mtop.Collector, "_local_status",
                        lambda self: ("running", SERVER_PID, "1m 0s", None))
    monkeypatch.setattr(mtop.Collector, "_local_stats", lambda self, pid: None)
    c = _collector(mode="auto")
    assert c.collect(0.0)["mode"] == "api"
    probes["local"] = True
    assert c.collect(1.0)["mode"] == "local"
    probes["local"] = False
    assert c.collect(2.0)["mode"] == "local"           # locked in, not re-probed


# ── renderer ─────────────────────────────────────────────────────────────────

def test_models_table_processor_and_forever(win):
    snap = {"models_ok": True, "models": [
        dict(API_PS["models"][0], size_vram=9 * GIB),
    ]}
    mtop.render_models(win, 0, snap)
    text = win.text()
    assert "qwen2.5-coder:32b" in text
    assert "55%/45% CPU/GPU" in text
    assert "forever" in text


def test_models_table_handles_null_sizes(win):
    snap = {"models_ok": True, "models": [{"name": "x", "size": None, "size_vram": None}]}
    mtop.render_models(win, 0, snap)
    assert "100% CPU" in win.text()


def test_header_fields_do_not_overlap_on_narrow_terminal(monkeypatch):
    class U:
        nodename = "a-rather-long-hostname-for-testing"
    monkeypatch.setattr(mtop.os, "uname", lambda: U())
    snap = {"status": "running", "uptime": "3d 14h", "container": "ollama",
            "mode": "local", "pid": 4711, "res_stats": {"procs": 3}}
    for cols in (50, 60, 80, 120):
        w = FakeWin(rows=5, cols=cols)
        mtop.render_header(w, 0, snap, stale=False)
        line = w.line(1)
        assert "host: " in line and "ollama: " in line
        if cols >= 80:
            assert "serve · pid 4711 +2r" in line and "up: 3d 14h" in line
        # clipped fields never run through the right border
        assert line[cols - 2] == "║", f"border overwritten at {cols} cols: {line!r}"
        # every field written in row 1 must start after the previous one ended
        spans = sorted((x, x + len(t)) for y, x, t, _ in w.calls if y == 1 and t.strip())
        for (_, prev_end), (start, _) in zip(spans, spans[1:], strict=False):
            assert start >= prev_end, f"overlap at {cols} cols: {line!r}"
        assert w.line(0)[1] == "╔" and w.line(0)[cols - 2] == "╗"
        assert w.line(2)[1] == "╚" and w.line(2)[cols - 2] == "╝"


def test_header_stale_flag_shown_when_room(monkeypatch):
    snap = {"status": "api-only", "uptime": "", "container": "ollama",
            "mode": "api", "api_url": "http://gpu-rig:11434"}
    w = FakeWin(rows=5, cols=120)
    mtop.render_header(w, 0, snap, stale=True)
    assert "STALE" in w.line(1) and "api: " in w.line(1)


def test_runners_table_extras():
    win = FakeWin(cols=200)
    snap = {"runners": [{"pid": 7, "model_name": "m", "ctx": "8192", "batch": "512",
                         "flash_attn": "on", "kv_k": "q8_0", "kv_v": "q8_0",
                         "vram": 3 * GIB, "rss": GIB, "engine": "ollama",
                         "ngl": "33", "threads": "8", "multiuser_cache": True,
                         "load_mode": "dio"}]}
    mtop.render_runners(win, 0, snap)
    text = win.text()
    assert "ollama-engine,ngl:33,thr:8,multiuser,O_DIRECT" in text
    assert "q8_0" in text and "3.0 G" in text and "1.0 G" in text


def test_draw_table_truncates_and_sizes_rule(win):
    y = mtop.draw_table(win, 0, 0, ["A", "B"], [["averyveryverylongcell", "x"]], [8, 4])
    assert y == 3
    assert win.line(0) == "A         B"
    assert win.line(2).startswith("averyve…")
    assert set(win.line(1)) == {"─"}
    assert len(win.line(1)) == len(win.line(2))


# ── docker mode through an injected runtime ──────────────────────────────────

class FakeRuntime:
    name = "docker-api"

    def __init__(self, status="running", pid=SERVER_PID):
        self.status, self.pid = status, pid
        self.execs = []
        self.stats_calls = 0

    def inspect(self, name):
        if name != "ollama":
            return None
        return {"status": self.status, "started_at": "2026-09-13T00:00:00Z", "pid": self.pid,
                "nano_cpus": 4_000_000_000, "cpu_quota": 0, "cpu_period": 100000,
                "env": [], "image": "ollama/ollama:latest"}

    def stats(self, name):
        self.stats_calls += 1
        return {"cpu": "12.50%", "mem_usage": "1.5GiB / 121GiB", "mem_pct": "1.24%",
                "cpu_pct": 12.5, "mem_used_bytes": 3 * GIB // 2, "mem_limit_bytes": 121 * GIB}

    def exec(self, name, cmd, timeout=5):
        self.execs.append(cmd)
        if cmd[:1] == ["ollama"]:
            return True, "NAME  ID  SIZE\nbielik  b669  14 GB"
        if cmd[:1] == ["sh"]:
            argv = "\0".join(OLLAMA_ENGINE_ARGV)
            return True, f"==> /proc/77/cmdline <==\n{argv}\n==> /proc/1/cmdline <==\nollama\0serve"
        return False, "no"


@pytest.fixture
def docker_host(monkeypatch):
    monkeypatch.setattr(mtop, "IS_LINUX", True)
    monkeypatch.setattr(mtop, "http_get_json", lambda url, timeout=5: (True, API_PS))
    # host /proc walk finds nothing for the container init pid -> exec fallback
    monkeypatch.setattr(mtop, "process_tree", lambda root, children=None: [root])
    monkeypatch.setattr(mtop, "read_proc_cmdline", lambda pid: None)
    return FakeRuntime()


def test_docker_mode_via_injected_runtime(docker_host):
    rt = docker_host
    c = _collector(mode="docker", container_runtime=rt, show_raw_ps=True)
    snap = c.collect(100.0)
    assert snap["mode"] == "docker" and snap["runtime"] == "docker-api"
    assert snap["status"] == "running" and snap["pid"] == SERVER_PID
    assert snap["cpu_limit"] == 4.0                      # NanoCpus honoured
    assert snap["res_stats"]["cpu"] == "12.50%"
    assert snap["raw_ps_ok"] and "bielik" in snap["raw_ps"]
    assert ["ollama", "ps"] in rt.execs
    r = snap["runners"][0]
    assert r["pid"] == 77 and r["rss"] is None           # exec fallback: no RSS
    assert r["model_name"] == "qwen2.5-coder:32b"


def test_docker_mode_auto_detects_and_locks(docker_host, monkeypatch):
    rt = docker_host
    c = _collector(mode="auto", container_runtime=rt)
    assert c.collect(100.0)["mode"] == "docker"
    rt.status = "exited"
    snap = c.collect(103.0)
    assert snap["mode"] == "docker" and snap["status"] == "exited"
    assert "res_stats" not in snap                        # short-circuits when down


def test_docker_mode_without_runtime_is_not_found(monkeypatch):
    monkeypatch.setattr(mtop, "detect_runtime", lambda runner, prefer: None)
    monkeypatch.setattr(mtop, "http_get_json", lambda url, timeout=5: (True, {"models": []}))
    c = _collector(mode="docker")
    snap = c.collect(0.0)
    assert snap["status"] == "not found" and snap["runtime"] is None


def test_needs_second_sample(docker_host):
    c = _collector(mode="docker", container_runtime=docker_host)
    snap = c.collect(100.0)
    assert c.needs_second_sample(snap) is True
    docker_host.name = "docker-cli"
    c2 = _collector(mode="docker", container_runtime=docker_host)
    assert c2.needs_second_sample(c2.collect(100.0)) is False
    assert c.needs_second_sample({"mode": "local", "pid": 1}) is True
    assert c.needs_second_sample({"mode": "api", "pid": None}) is False


def test_nvidia_attempts_include_container_exec(docker_host):
    c = _collector(mode="docker", container_runtime=docker_host, show_gpu=True)
    c.collect(100.0)
    providers = c._build_gpu_providers()
    nv = providers[1]                                   # [0] is NVML
    labels = [label for label, _ in nv._attempts_fn()]
    assert labels == ["host", "container"]
    ok, out = nv._attempts_fn()[1][1](["nvidia-smi", "-L"], 3)
    assert docker_host.execs[-1] == ["nvidia-smi", "-L"]


# ── server config (env + version) and GPU linking ────────────────────────────

VERSION_AND_PS = {"/api/version": (True, {"version": "0.33.2"}), "/api/ps": (True, API_PS)}


def _http(table):
    def get(url, timeout=5):
        return table.get("/api" + url.rsplit("/api", 1)[-1], (False, "nope"))
    return get


def test_server_info_docker_from_inspect_env(docker_host, monkeypatch):
    monkeypatch.setattr(mtop, "http_get_json", _http(VERSION_AND_PS))
    rt = docker_host

    def inspect(name):
        info = FakeRuntime.inspect(rt, name)
        info["env"] = ["PATH=/x", "OLLAMA_KEEP_ALIVE=24h", "OLLAMA_FLASH_ATTENTION=1"]
        return info
    rt.inspect = inspect
    snap = _collector(mode="docker", container_runtime=rt).collect(100.0)
    assert snap["server"] == {"version": "0.33.2", "env_source": "container",
                              "env": {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KEEP_ALIVE": "24h"}}


def test_server_info_local_prefers_process_then_systemd(local_host, monkeypatch):
    monkeypatch.setattr(mtop, "http_get_json", _http(VERSION_AND_PS))
    monkeypatch.setattr(mtop, "read_proc_environ", lambda pid: ["OLLAMA_NUM_PARALLEL=2"])
    monkeypatch.setattr(mtop, "systemd_environment",
                        lambda unit="ollama.service": ["OLLAMA_HOST=x"])
    snap = _collector().collect(100.0)
    assert snap["server"]["env"] == {"OLLAMA_NUM_PARALLEL": "2"}
    assert snap["server"]["env_source"] == "process"

    monkeypatch.setattr(mtop, "read_proc_environ", lambda pid: None)
    snap = _collector().collect(100.0)
    assert snap["server"] == {"version": "0.33.2", "env": {"OLLAMA_HOST": "x"},
                              "env_source": "systemd"}

    monkeypatch.setattr(mtop, "systemd_environment", lambda unit="ollama.service": None)
    snap = _collector().collect(100.0)
    assert snap["server"]["env_source"] is None and snap["server"]["version"] == "0.33.2"


def test_server_info_version_failure_is_none(local_host, monkeypatch):
    monkeypatch.setattr(mtop, "read_proc_environ", lambda pid: [])
    snap = _collector().collect(100.0)      # local_host's http stub only knows /api/ps
    assert snap["server"]["version"] is None and snap["server"]["env_source"] == "process"


def test_runners_linked_to_gpus_in_snapshot(local_host, monkeypatch):
    monkeypatch.setattr(mtop, "read_proc_environ", lambda pid: [])
    gpus = [{"vendor": "nvidia", "index": 0, "name": "x", "util": "1", "mem_used": "1",
             "mem_total": "2", "temp": "1", "procs": [{"pid": RUNNER_PID, "mem_mib": 14000}]}]
    monkeypatch.setattr(mtop.Collector, "_gpu_read", lambda self: gpus)
    snap = _collector(show_gpu=True).collect(100.0)
    r = snap["runners"][0]
    assert r["gpu"] == ["nvidia:0"] and r["gpu_mem_mib"] == 14000
    assert snap["gpus"][0]["procs"][0]["model"] == "qwen2.5-coder:32b"


def test_render_server_config_wraps_and_names_source():
    w = FakeWin(rows=12, cols=60)
    env = {f"OLLAMA_VAR_{i}": "value" for i in range(6)}
    y = mtop.render_server_config(w, 0, {"mode": "docker",
                                          "server": {"env": env, "env_source": "container"}})
    text = w.text()
    assert "SERVER CONFIG" in text and "source: container" in text
    assert "OLLAMA_VAR_0=value" in text and "OLLAMA_VAR_5=value" in text
    assert all(len(w.line(i)) <= 60 for i in range(y))
    assert y >= 5                                       # header + >=2 wrapped lines + source


def test_render_server_config_messages(win):
    mtop.render_server_config(win, 0, {"mode": "local", "server": {"env": {}, "env_source": None}})
    assert "not readable" in win.text()
    w2 = FakeWin()
    mtop.render_server_config(w2, 0, {"mode": "docker",
                                       "server": {"env": {}, "env_source": "container"}})
    assert "defaults (container)" in w2.text()
    w3 = FakeWin()
    assert mtop.render_server_config(w3, 0, {"mode": "api", "server": {}}) == 0


def test_gpu_section_lists_processes(win):
    snap = {"gpus": [{"vendor": "nvidia", "index": 0, "name": "RTX", "util": "5",
                      "mem_used": "100", "mem_total": "200", "temp": "40",
                      "procs": [{"pid": 7, "mem_mib": 13800, "model": "bielik:latest"},
                                {"pid": 8, "mem_mib": None}]}]}
    mtop.render_gpu_stats(win, 0, snap)
    assert "procs: bielik:latest (13.5G), pid 8" in win.text()


def test_runners_table_gpu_column():
    win = FakeWin(cols=200)
    snap = {"runners": [{"pid": 7, "model_name": "m", "ctx": "1", "gpu": ["nvidia:0", "nvidia:1"]},
                        {"pid": 8, "model_name": "n", "ctx": "1"}]}
    mtop.render_runners(win, 0, snap)
    rows = [win.line(i) for i in range(3, 5)]          # section hdr, table hdr, rule, rows
    assert "0,1" in rows[0] and rows[1].rstrip().endswith("—")


def test_header_shows_ollama_version():
    w = FakeWin(rows=4, cols=120)
    snap = {"status": "running", "uptime": "1h 0m", "container": "ollama", "mode": "docker",
            "server": {"version": "0.33.2"}}
    mtop.render_header(w, 0, snap, stale=False)
    assert "ollama 0.33.2" in w.line(1)
