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
