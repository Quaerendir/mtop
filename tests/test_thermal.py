"""ACPI thermal zones: sysfs reader, TUI line, Prometheus metric."""

import pytest

from mtop import export, procfs, ui


def _zone(root, n, kind, mdeg, trips=()):
    z = root / f"thermal_zone{n}"
    z.mkdir()
    (z / "type").write_text(kind + "\n")
    (z / "temp").write_text(f"{mdeg}\n")
    for i, (ttype, ttemp) in enumerate(trips):
        (z / f"trip_point_{i}_type").write_text(ttype + "\n")
        (z / f"trip_point_{i}_temp").write_text(f"{ttemp}\n")


@pytest.fixture
def thermal(tmp_path, monkeypatch):
    monkeypatch.setattr(procfs, "SYSFS_THERMAL", str(tmp_path))
    return tmp_path


def test_reads_only_acpitz_in_zone_order(thermal):
    _zone(thermal, 10, "acpitz", 61250, [("passive", 90000), ("critical", 104800)])
    _zone(thermal, 2, "acpitz", 44600, [("critical", 104800)])
    _zone(thermal, 3, "x86_pkg_temp", 99000)
    (thermal / "cooling_device0").mkdir()
    acpi = procfs.read_acpi_thermal()
    assert acpi == {"max": 61.2, "crit": 104.8,
                    "zones": [{"zone": "thermal_zone2", "temp": 44.6},
                              {"zone": "thermal_zone10", "temp": 61.2}]}


def test_no_acpi_zones(thermal):
    _zone(thermal, 0, "x86_pkg_temp", 50000)
    assert procfs.read_acpi_thermal() is None


def test_missing_sysfs(monkeypatch):
    monkeypatch.setattr(procfs, "SYSFS_THERMAL", "/nonexistent/thermal")
    assert procfs.read_acpi_thermal() is None


def test_unreadable_zone_is_skipped(thermal):
    _zone(thermal, 0, "acpitz", 50000)
    _zone(thermal, 1, "acpitz", 0)
    (thermal / "thermal_zone1" / "temp").write_text("garbage")
    acpi = procfs.read_acpi_thermal()
    assert acpi["zones"] == [{"zone": "thermal_zone0", "temp": 50.0}]
    assert acpi["crit"] is None


def test_format_and_color():
    acpi = {"max": 58.4, "crit": 104.8,
            "zones": [{"zone": "z0", "temp": 52.1}, {"zone": "z1", "temp": 58.4}]}
    assert ui.format_acpi_thermal(acpi) == "ACPI  58°C  (max of 2 zones, 52–58°C, crit 105°C)"
    assert ui.acpi_color(acpi) == ui.C_DIM
    assert ui.acpi_color({**acpi, "max": 85.0}) == ui.C_WARN
    assert ui.acpi_color({**acpi, "max": 95.0}) == ui.C_ERR
    one = {"max": 40.0, "crit": None, "zones": [{"zone": "z0", "temp": 40.0}]}
    assert ui.format_acpi_thermal(one) == "ACPI  40°C"
    assert ui.acpi_color({**one, "max": 120.0}) == ui.C_DIM


def test_prometheus_metric():
    snap = {"mode": "local", "status": "running",
            "acpi_thermal": {"max": 58.4, "crit": None,
                             "zones": [{"zone": "thermal_zone0", "temp": 58.4}]}}
    text = export.prometheus_text(snap, "1")
    assert 'mtop_acpi_temperature_celsius{zone="thermal_zone0"} 58.4' in text
    assert "mtop_acpi_temperature_celsius" not in export.prometheus_text(
        {"mode": "local", "status": "running"}, "1")
