"""GPU providers against fake sysfs trees, canned rocm-smi/nvidia-smi output."""

import json
import os

import pytest

from mtop import gpu


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def make_amd_card(root, name, *, vendor="0x1002", slot, pci_id="1002:744C",
                  vram_total, vram_used, gtt_total, gtt_used, busy=17,
                  product_name=None, temp_edge=61000, power_uw=245_000_000):
    dev = os.path.join(root, name, "device")
    _write(f"{dev}/vendor", vendor + "\n")
    _write(f"{dev}/uevent", f"DRIVER=amdgpu\nPCI_ID={pci_id}\nPCI_SLOT_NAME={slot}\n")
    _write(f"{dev}/mem_info_vram_total", str(vram_total))
    _write(f"{dev}/mem_info_vram_used", str(vram_used))
    _write(f"{dev}/mem_info_gtt_total", str(gtt_total))
    _write(f"{dev}/mem_info_gtt_used", str(gtt_used))
    _write(f"{dev}/gpu_busy_percent", str(busy))
    if product_name is not None:
        _write(f"{dev}/product_name", product_name)
    hw = f"{dev}/hwmon/hwmon4"
    _write(f"{hw}/temp1_label", "junction")
    _write(f"{hw}/temp1_input", "75000")
    _write(f"{hw}/temp2_label", "edge")
    _write(f"{hw}/temp2_input", str(temp_edge))
    _write(f"{hw}/power1_average", str(power_uw))
    return dev


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    root = str(tmp_path / "drm")
    os.makedirs(root)
    monkeypatch.setattr(gpu, "SYSFS_DRM", root)
    return root


@pytest.fixture
def pci_ids(tmp_path, monkeypatch):
    p = tmp_path / "pci.ids"
    p.write_text(
        "# pci.ids excerpt\n"
        "1002  Advanced Micro Devices, Inc. [AMD/ATI]\n"
        "\t744c  Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]\n"
        "\t\t1002 0e3b  Radeon RX 7900 XTX\n"
        "\t15bf  Phoenix1\n"
        "10de  NVIDIA Corporation\n"
        "\t2684  AD102 [GeForce RTX 4090]\n"
    )
    monkeypatch.setattr(gpu, "_PCI_IDS_PATHS", (str(p),))
    monkeypatch.setattr(gpu, "_pci_ids_missing", False)
    gpu._pci_name_cache.clear()
    return p


def test_pci_device_name(pci_ids):
    assert gpu.pci_device_name("1002", "744c") == "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]"
    assert gpu.pci_device_name("10de", "2684") == "AD102 [GeForce RTX 4090]"
    assert gpu.pci_device_name("1002", "ffff") is None
    # subsystem lines (two tabs) are never matched as devices
    assert gpu.pci_device_name("1002", "0e3b") is None


def test_pci_ids_missing_is_remembered(monkeypatch):
    monkeypatch.setattr(gpu, "_PCI_IDS_PATHS", ("/nonexistent/pci.ids",))
    monkeypatch.setattr(gpu, "_pci_ids_missing", False)
    gpu._pci_name_cache.clear()
    assert gpu.pci_device_name("1002", "744c") is None
    assert gpu._pci_ids_missing is True


class TestAmdSysfs:
    def test_discrete_card(self, sysfs, pci_ids):
        make_amd_card(sysfs, "card1", slot="0000:03:00.0",
                      vram_total=24 << 30, vram_used=6 << 30,
                      gtt_total=32 << 30, gtt_used=100 << 20)
        # connector entries and a non-AMD card must be skipped
        os.makedirs(os.path.join(sysfs, "card1-DP-1"))
        make_amd_card(sysfs, "card0", vendor="0x10de", slot="0000:01:00.0",
                      vram_total=1, vram_used=1, gtt_total=1, gtt_used=1)

        out = gpu.AmdSysfsProvider().collect()
        assert len(out) == 1
        g = out[0]
        assert g["vendor"] == "amd"
        assert g["name"] == "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]"
        assert g["util"] == "17"
        assert g["mem_total"] == str(24 * 1024)
        assert g["mem_used"] == str(6 * 1024)
        assert g["temp"] == "61"            # edge, not junction
        assert g["power"] == "245"
        assert "unified" not in g
        assert g["gtt_total"] == str(32 * 1024) and g["gtt_used"] == "100"

    def test_product_name_wins(self, sysfs, pci_ids):
        make_amd_card(sysfs, "card0", slot="0000:03:00.0", product_name="Radeon AI PRO R9700",
                      vram_total=32 << 30, vram_used=0, gtt_total=1 << 30, gtt_used=0)
        assert gpu.AmdSysfsProvider().collect()[0]["name"] == "Radeon AI PRO R9700"

    def test_unknown_id_falls_back_to_pci_id(self, sysfs, pci_ids):
        make_amd_card(sysfs, "card0", slot="0000:03:00.0", pci_id="1002:ABCD",
                      vram_total=32 << 30, vram_used=0, gtt_total=1 << 30, gtt_used=0)
        assert gpu.AmdSysfsProvider().collect()[0]["name"] == "AMD 1002:ABCD"

    def test_apu_reports_unified_gtt_pool(self, sysfs, pci_ids):
        make_amd_card(sysfs, "card0", slot="0000:c4:00.0", pci_id="1002:15BF",
                      vram_total=512 << 20, vram_used=400 << 20,
                      gtt_total=30 << 30, gtt_used=8 << 30)
        g = gpu.AmdSysfsProvider().collect()[0]
        assert g["unified"] is True
        assert g["mem_total"] == str(30 * 1024)
        assert g["mem_used"] == str(400 + 8 * 1024)
        assert "gtt_total" not in g

    def test_ordering_by_pci_slot_not_card_number(self, sysfs, pci_ids):
        make_amd_card(sysfs, "card0", slot="0000:0a:00.0", product_name="second",
                      vram_total=1 << 31, vram_used=0, gtt_total=1 << 30, gtt_used=0)
        make_amd_card(sysfs, "card1", slot="0000:03:00.0", product_name="first",
                      vram_total=1 << 31, vram_used=0, gtt_total=1 << 30, gtt_used=0)
        names = [g["name"] for g in gpu.AmdSysfsProvider().collect()]
        assert names == ["first", "second"]

    def test_no_cards_reprobes_next_cycle(self, sysfs):
        p = gpu.AmdSysfsProvider()
        assert p.collect() is None
        assert p._cards is None          # not cached as "empty forever"
        make_amd_card(sysfs, "card0", slot="0000:03:00.0", product_name="late",
                      vram_total=1 << 31, vram_used=0, gtt_total=1 << 30, gtt_used=0)
        assert p.collect()[0]["name"] == "late"

    def test_missing_sysfs_root(self, monkeypatch):
        monkeypatch.setattr(gpu, "SYSFS_DRM", "/nonexistent/drm")
        assert gpu.AmdSysfsProvider().collect() is None


ROCM5 = {
    "card0": {
        "GPU use (%)": "12",
        "VRAM Total Used Memory (B)": "1073741824",     # deliberately first
        "VRAM Total Memory (B)": "17163091968",
        "Temperature (Sensor edge) (C)": "45.0",
        "Temperature (Sensor junction) (C)": "52.0",
        "Card series": "Radeon RX 6800 XT",
    }
}
ROCM6 = {
    "card:0": {
        "GPU use (%)": "7",
        "VRAM Total Memory (B)": "25753026560",
        "VRAM Total Used Memory (B)": "536870912",
        "Temperature (Sensor edge) (C)": "40.0",
        "Card Series": "Radeon RX 7900 XTX",
    },
    "system": {"Driver version": "6.8.5"},
}


def _runner_for(payload, ok=True):
    def run(cmd, timeout):
        assert cmd[0] == "rocm-smi"
        return ok, json.dumps(payload) if payload is not None else ""
    return run


class TestRocmSmi:
    def test_rocm5_schema_total_is_not_used(self):
        g = gpu.RocmSmiProvider(_runner_for(ROCM5)).collect()[0]
        assert g["mem_total"] == "16368"     # 17163091968 B -> MiB
        assert g["mem_used"] == "1024"
        assert g["util"] == "12"
        assert g["temp"] == "45.0"
        assert g["name"] == "Radeon RX 6800 XT"

    def test_rocm6_schema(self):
        out = gpu.RocmSmiProvider(_runner_for(ROCM6)).collect()
        assert len(out) == 1                 # 'system' block skipped
        assert out[0]["mem_total"] == "24560"
        assert out[0]["mem_used"] == "512"   # < 1 GiB in bytes: still bytes
        assert out[0]["name"] == "Radeon RX 7900 XTX"

    def test_mib_reporting_build(self):
        payload = {"card0": {"VRAM Total Memory (B)": "16368",
                             "VRAM Total Used Memory (B)": "1024", "GPU use (%)": "1"}}
        g = gpu.RocmSmiProvider(_runner_for(payload)).collect()[0]
        assert g["mem_total"] == "16368" and g["mem_used"] == "1024"

    def test_failures_return_none(self):
        assert gpu.RocmSmiProvider(_runner_for(None, ok=False)).collect() is None
        assert gpu.RocmSmiProvider(lambda c, t: (True, "not json")).collect() is None


NVSMI_TWO = ("0, NVIDIA GeForce RTX 4090, 32, 17382, 24564, 42, 285.10\n"
             "1, NVIDIA GeForce RTX 3060, 0, 512, 12288, 31, [N/A]\n")
NVSMI_TEGRA = "0, Orin (nvgpu), 3, [N/A], [N/A], 45, [N/A]\n"


class TestNvidiaSmi:
    def _provider(self, table):
        """table: label -> (ok, out); every call is logged as (label, argv)."""
        calls = []

        def runner_for(label):
            def run(cmd, timeout):
                calls.append((label, cmd))
                return table.get(label, (False, "command not found: nvidia-smi"))
            return run

        attempts = [("host", runner_for("host")), ("container", runner_for("container"))]
        return gpu.NvidiaSmiProvider(lambda: attempts), calls

    def test_parses_two_cards_and_power(self):
        p, _ = self._provider({"host": (True, NVSMI_TWO)})
        out = p.collect()
        assert [g["index"] for g in out] == [0, 1]
        assert out[0]["power"] == "285.10"
        assert "power" not in out[1]
        assert out[1]["mem_used"] == "512"

    def test_container_fallback_is_memoized(self):
        p, calls = self._provider({"container": (True, NVSMI_TWO)})
        assert len(p.collect()) == 2
        first = len(calls)
        assert first == 2                     # host failed, then container
        p.collect()
        assert len(calls) == first + 1        # straight to the good attempt
        assert calls[-1][0] == "container"
        assert calls[-1][1][0] == "nvidia-smi"

    def test_none_when_nothing_answers(self):
        p, _ = self._provider({})
        assert p.collect() is None


class _Fake(gpu.GpuProvider):
    def __init__(self, name, results):
        self.name = name
        self._results = list(results)
        self.calls = 0

    def collect(self):
        self.calls += 1
        r = self._results.pop(0) if self._results else None
        if isinstance(r, Exception):
            raise r
        return r


class TestGpuMonitor:
    def test_concatenates_vendors(self):
        nv = _Fake("nv", [[{"vendor": "nvidia", "index": 0, "mem_total": "24564"}]] * 3)
        amd = _Fake("amd", [[{"vendor": "amd", "index": 0, "mem_total": "16368"}]] * 3)
        mon = gpu.GpuMonitor([nv, amd], lambda: 0.0)
        assert [g["vendor"] for g in mon.collect()] == ["nvidia", "amd"]

    def test_dead_provider_retried_after_backoff(self):
        clock = [0.0]
        p = _Fake("nv", [None, None, [{"vendor": "nvidia", "index": 0, "mem_total": "1"}]])
        mon = gpu.GpuMonitor([p], lambda: clock[0])
        assert mon.collect() is None and p.calls == 1
        clock[0] = 10.0
        assert mon.collect() is None and p.calls == 1       # still backing off
        clock[0] = 31.0
        assert mon.collect() is None and p.calls == 2       # re-probed, failed again
        clock[0] = 62.0
        assert mon.collect() and p.calls == 3

    def test_exception_in_provider_is_contained(self):
        p = _Fake("boom", [RuntimeError("driver exploded")])
        assert gpu.GpuMonitor([p], lambda: 0.0).collect() is None

    def test_tegra_na_memory_patched_from_meminfo_and_deduped(self, monkeypatch):
        monkeypatch.setattr(gpu, "meminfo_mib", lambda: (12000, 65536))
        nv = _Fake("nv", [[{"vendor": "nvidia", "index": 0, "name": "Orin (nvgpu)",
                            "util": "3", "mem_used": "[N/A]", "mem_total": "[N/A]",
                            "temp": "45"}]])
        tegra = gpu.TegraUnifiedProvider(lambda: ("NVIDIA Jetson AGX Orin", 12000, 65536))
        out = gpu.GpuMonitor([nv, tegra], lambda: 0.0).collect()
        assert len(out) == 1
        assert out[0]["unified"] is True
        assert out[0]["mem_used"] == "12000" and out[0]["mem_total"] == "65536"
        assert out[0]["name"] == "Orin (nvgpu)"          # nvidia-smi entry wins

    def test_tegra_provider_alone(self):
        tegra = gpu.TegraUnifiedProvider(lambda: ("NVIDIA GB10", 70000, 131072))
        out = gpu.GpuMonitor([tegra], lambda: 0.0).collect()
        assert out[0]["unified"] and out[0]["name"] == "NVIDIA GB10"
        empty = gpu.TegraUnifiedProvider(lambda: None)
        assert gpu.GpuMonitor([empty], lambda: 0.0).collect() is None


# ── NVML via a fake driver library ───────────────────────────────────────────

import ctypes  # noqa: E402


class FakeNvml:
    """Mimics the handful of libnvidia-ml entry points NvmlProvider calls."""

    def __init__(self, cards, init_rc=0, procs_version="v3"):
        self.cards = cards            # list of dicts: name, util, mem, temp, power, procs
        self.init_rc = init_rc
        self.calls = []
        if procs_version != "v3":
            self.nvmlDeviceGetComputeRunningProcesses_v3 = None
        if procs_version == "v1":
            self.nvmlDeviceGetComputeRunningProcesses_v2 = None

    def nvmlInit_v2(self):
        self.calls.append("init")
        return self.init_rc

    def nvmlDeviceGetCount_v2(self, p):
        p._obj.value = len(self.cards)
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, idx, p):
        p._obj.value = idx + 1
        return 0

    def _card(self, dev):
        return self.cards[dev.value - 1]

    def nvmlDeviceGetName(self, dev, buf, n):
        buf.value = self._card(dev)["name"].encode()
        return 0

    def nvmlDeviceGetUtilizationRates(self, dev, p):
        u = self._card(dev).get("util")
        if u is None:
            return 3
        p._obj.gpu = u
        return 0

    def nvmlDeviceGetMemoryInfo(self, dev, p):
        m = self._card(dev).get("mem")
        if m is None:
            return 3                              # NOT_SUPPORTED (GB10 / Jetson)
        p._obj.total, p._obj.used, p._obj.free = m[1], m[0], m[1] - m[0]
        return 0

    def nvmlDeviceGetTemperature(self, dev, sensor, p):
        p._obj.value = self._card(dev)["temp"]
        return 0

    def nvmlDeviceGetPowerUsage(self, dev, p):
        pw = self._card(dev).get("power")
        if pw is None:
            return 3
        p._obj.value = pw
        return 0

    def _procs(self, dev, count, infos):
        procs = self._card(dev).get("procs") or []
        if infos is None:
            count._obj.value = len(procs)
            return 0 if not procs else 7          # INSUFFICIENT_SIZE with the count
        for i, (pid, mem) in enumerate(procs):
            infos[i].pid = pid
            infos[i].usedGpuMemory = mem
        count._obj.value = len(procs)
        return 0

    nvmlDeviceGetComputeRunningProcesses_v3 = _procs
    nvmlDeviceGetComputeRunningProcesses_v2 = _procs
    nvmlDeviceGetComputeRunningProcesses = _procs


class TestNvml:
    def test_two_cards_with_processes(self):
        lib = FakeNvml([
            {"name": "NVIDIA GeForce RTX 4090", "util": 32, "mem": (17382 << 20, 24564 << 20),
             "temp": 42, "power": 285_100, "procs": [(4711, 13_800 << 20), (4712, 0)]},
            {"name": "NVIDIA GeForce RTX 3060", "util": 0, "mem": (512 << 20, 12288 << 20),
             "temp": 31, "power": None, "procs": []},
        ])
        p = gpu.NvmlProvider(loader=lambda: lib)
        out = p.collect()
        assert [g["name"] for g in out] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3060"]
        g0 = out[0]
        assert g0["util"] == "32" and g0["mem_used"] == "17382" and g0["mem_total"] == "24564"
        assert g0["temp"] == "42" and g0["power"] == "285.10"
        assert g0["procs"] == [{"pid": 4711, "mem_mib": 13800}, {"pid": 4712, "mem_mib": None}]
        assert "power" not in out[1] and out[1]["procs"] == []
        p.collect()
        assert lib.calls.count("init") == 1          # initialised once

    def test_unified_part_reports_na_memory(self):
        lib = FakeNvml([{"name": "NVIDIA GB10", "util": 0, "mem": None, "temp": 41,
                         "power": 10_710, "procs": [(99, gpu.NVML_VALUE_NOT_AVAILABLE)]}])
        g = gpu.NvmlProvider(loader=lambda: lib).collect()[0]
        assert g["mem_used"] == "[N/A]" and g["mem_total"] == "[N/A]"
        assert g["procs"] == [{"pid": 99, "mem_mib": None}]

    @pytest.mark.parametrize("ver", ["v2", "v1"])
    def test_older_process_entry_points(self, ver):
        lib = FakeNvml([{"name": "x", "util": 1, "mem": (1 << 30, 2 << 30), "temp": 1,
                         "procs": [(5, 1 << 30)]}], procs_version=ver)
        g = gpu.NvmlProvider(loader=lambda: lib).collect()[0]
        assert g["procs"] == [{"pid": 5, "mem_mib": 1024}]

    def test_no_library_or_init_failure(self):
        p = gpu.NvmlProvider(loader=lambda: None)
        assert p.collect() is None and p.collect() is None
        p2 = gpu.NvmlProvider(loader=lambda: FakeNvml([], init_rc=9))
        assert p2.collect() is None

    def test_struct_layout_matches_nvml_header(self):
        assert ctypes.sizeof(gpu._NvmlProcessV2) == 24
        assert gpu._NvmlProcessV2.usedGpuMemory.offset == 8
        assert ctypes.sizeof(gpu._NvmlProcessV1) == 16
        assert ctypes.sizeof(gpu._NvmlMemory) == 24

    def test_supersedes_nvidia_smi_in_registry(self):
        lib = FakeNvml([{"name": "nvml-card", "util": 1, "mem": (1 << 30, 2 << 30), "temp": 1}])
        nvml = gpu.NvmlProvider(loader=lambda: lib)
        smi = _Fake("nvidia-smi", [[{"vendor": "nvidia", "index": 0, "name": "smi-card",
                                     "mem_total": "2048"}]] * 3)
        mon = gpu.GpuMonitor([nvml, smi], lambda: 0.0)
        assert [g["name"] for g in mon.collect()] == ["nvml-card"]
        assert smi.calls == 0
        # when NVML is absent, nvidia-smi still answers
        mon2 = gpu.GpuMonitor([gpu.NvmlProvider(loader=lambda: None), smi], lambda: 0.0)
        assert [g["name"] for g in mon2.collect()] == ["smi-card"]


# ── Intel via a fake sysfs tree (no hardware available: ABI-doc driven) ──────

def make_intel_card(root, name, *, slot, pci_id="8086:56A0", driver="xe",
                    vram_total=None, freq=None, temp=None, energy=None, power=None):
    dev = os.path.join(root, name, "device")
    _write(f"{dev}/vendor", "0x8086\n")
    _write(f"{dev}/uevent", f"DRIVER={driver}\nPCI_ID={pci_id}\nPCI_SLOT_NAME={slot}\n")
    if vram_total is not None:
        _write(f"{dev}/tile0/physical_vram_size_bytes", str(vram_total))
    if freq is not None:
        if driver == "xe":
            _write(f"{dev}/tile0/gt0/freq0/cur_freq", str(freq))
        else:
            _write(f"{root}/{name}/gt/gt0/rps_cur_freq_mhz", str(freq))
    hw = f"{dev}/hwmon/hwmon2"
    if temp is not None:
        _write(f"{hw}/temp1_input", str(temp))
    if energy is not None:
        _write(f"{hw}/energy1_input", str(energy))
    if power is not None:
        _write(f"{hw}/power1_input", str(power))
    return dev


@pytest.fixture
def intel_pci_ids(tmp_path, monkeypatch):
    p = tmp_path / "pci.ids"
    p.write_text("8086  Intel Corporation\n\t56a0  DG2 [Arc A770]\n"
                 "\ta7a0  Raptor Lake-P [Iris Xe Graphics]\n")
    monkeypatch.setattr(gpu, "_PCI_IDS_PATHS", (str(p),))
    monkeypatch.setattr(gpu, "_pci_ids_missing", False)
    gpu._pci_name_cache.clear()


class TestIntelSysfs:
    def test_discrete_xe_card(self, sysfs, intel_pci_ids):
        make_intel_card(sysfs, "card1", slot="0000:03:00.0", vram_total=16 << 30, freq=2100,
                        temp=52000, power=95_000_000)
        make_amd_card(sysfs, "card0", slot="0000:01:00.0", vram_total=1, vram_used=0,
                      gtt_total=1, gtt_used=0)                      # other vendor: skipped
        clock = [10.0]
        g = gpu.IntelSysfsProvider(clock=lambda: clock[0]).collect()
        assert len(g) == 1
        g = g[0]
        assert g["vendor"] == "intel" and g["name"] == "DG2 [Arc A770]"
        assert g["driver"] == "xe" and g["mem_total"] == str(16 * 1024) and g["mem_used"] == "N/A"
        assert "unified" not in g and g["util"] == "N/A"
        assert g["freq_mhz"] == 2100 and g["temp"] == "52" and g["power"] == "95"

    def test_igpu_is_unified_and_energy_becomes_power(self, sysfs, intel_pci_ids):
        dev = make_intel_card(sysfs, "card0", slot="0000:00:02.0", pci_id="8086:A7A0",
                              driver="i915", freq=900, energy=1_000_000_000)
        clock = [100.0]
        p = gpu.IntelSysfsProvider(clock=lambda: clock[0], meminfo=lambda: (4000, 32000))
        g = p.collect()[0]
        assert g["name"] == "Raptor Lake-P [Iris Xe Graphics]" and g["unified"] is True
        assert g["mem_used"] == "4000" and g["mem_total"] == "32000"
        assert g["freq_mhz"] == 900 and "power" not in g            # no delta yet
        _write(f"{dev}/hwmon/hwmon2/energy1_input", str(1_000_000_000 + 15_000_000 * 2))
        clock[0] = 102.0
        assert p.collect()[0]["power"] == "15"                      # 30 J over 2 s

    def test_no_intel_cards(self, sysfs):
        assert gpu.IntelSysfsProvider().collect() is None

    def test_unknown_device_id_falls_back_to_pci_id(self, sysfs, intel_pci_ids):
        make_intel_card(sysfs, "card0", slot="0000:03:00.0", pci_id="8086:FFFF", vram_total=1 << 30)
        assert gpu.IntelSysfsProvider().collect()[0]["name"] == "Intel 8086:FFFF"


# ── Apple via canned ioreg plist (no hardware available) ─────────────────────

IOREG_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
  <dict>
    <key>IOClass</key><string>AGXAcceleratorG14X</string>
    <key>gpu-core-count</key><integer>38</integer>
    <key>PerformanceStatistics</key>
    <dict>
      <key>Device Utilization %</key><integer>37</integer>
      <key>Renderer Utilization %</key><integer>35</integer>
      <key>In use system memory</key><integer>7516192768</integer>
      <key>Alloc system memory</key><integer>8000000000</integer>
    </dict>
  </dict>
  <dict>
    <key>IOClass</key><string>IOAcceleratorSomethingElse</string>
    <key>PerformanceStatistics</key><dict><key>Device Utilization %</key><integer>1</integer></dict>
  </dict>
</array>
</plist>
"""


class TestAppleIoreg:
    def _runner(self, table):
        def run(cmd, timeout):
            return table.get(cmd[0] if cmd[0] != "sysctl" else cmd[-1], (False, ""))
        return run

    def test_parse_and_collect(self):
        run = self._runner({"ioreg": (True, IOREG_PLIST),
                            "machdep.cpu.brand_string": (True, "Apple M2 Max\n"),
                            "hw.memsize": (True, str(64 << 30))})
        out = gpu.AppleGpuProvider(run).collect()
        assert len(out) == 1
        g = out[0]
        assert g["vendor"] == "apple" and g["name"] == "Apple M2 Max (38 cores)"
        assert g["util"] == "37" and g["unified"] is True
        assert g["mem_used"] == "7168" and g["mem_total"] == str(64 * 1024)
        assert g["temp"] == "N/A"

    def test_no_agx_entries_or_ioreg_failure(self):
        assert gpu.AppleGpuProvider(self._runner({})).collect() is None
        plist = IOREG_PLIST.replace("AGXAcceleratorG14X", "IntelAccelerator")
        run = self._runner({"ioreg": (True, plist)})
        assert gpu.AppleGpuProvider(run).collect() is None
        assert gpu.AppleGpuProvider.parse_ioreg("not a plist") == []

    def test_missing_sysctl_values(self):
        run = self._runner({"ioreg": (True, IOREG_PLIST)})
        g = gpu.AppleGpuProvider(run).collect()[0]
        assert g["name"] == "Apple GPU (38 cores)" and g["mem_total"] == "N/A"
