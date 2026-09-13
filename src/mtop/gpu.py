"""
mtop.gpu — multi-vendor GPU telemetry providers.

Replaces the single memoized `nvidia-smi`-or-unified probe from v0.3.0 with a
small provider registry, so a box with an RTX 4070 Ti Super *and* a Radeon AI
PRO R9700 shows both cards instead of whichever probe answered first.

Design notes / gotchas encoded here:

* AMD telemetry comes from **sysfs first**, not `rocm-smi`. sysfs needs no ROCm
  install, no fork per cycle, survives ROCm major-version JSON schema churn
  (5.x `card0` → 6.x `card:0` → amd-smi), and works inside containers because
  /sys is bind-mounted read-only by default. `rocm-smi` is kept only as a
  fallback for exotic setups.
* `/sys/class/drm` contains connector entries (`card0-DP-1`); only bare
  `cardN` directories with a `device/vendor` of 0x1002 are amdgpu cards.
* APUs (780M, Strix) report `mem_info_vram_total` as a tiny carve-out; the real
  pool is GTT. Detected and reported as unified memory rather than a 512 MiB
  card that is somehow 900% full.
* Ordering is by PCI slot (`PCI_SLOT_NAME` from uevent), which is stable across
  boots — unlike `cardN` numbering, and unlike HIP device order, which is
  another sort entirely (see `HIP_VISIBLE_DEVICES`).

Everything is stdlib. Set `MTOP_SYSFS_DRM` to point the AMD provider at a fake
tree for testing.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import re
import sys
from typing import Any
from collections.abc import Callable

SYSFS_DRM = os.environ.get("MTOP_SYSFS_DRM", "/sys/class/drm")
_CARD_RE = re.compile(r"^card\d+$")

IS_LINUX = sys.platform.startswith("linux")
IS_WINDOWS = sys.platform == "win32"
IS_DARWIN = sys.platform == "darwin"


# ── small local helpers (kept independent of __init__ to avoid a cycle) ───────

def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _read_int(path: str) -> int | None:
    v = _read(path)
    if v is None:
        return None
    try:
        return int(v.strip())
    except ValueError:
        return None


def _mib_scale(total: float | None) -> float:
    """rocm-smi reports VRAM in bytes on some builds, MiB on others.

    Decide per *card* from the total: a total of 1 GiB-as-a-number or more
    can only be bytes (no card has a PiB of VRAM). Deciding per value, as an
    earlier cut did, misread a used figure under 1 GiB in a bytes-reporting
    build as MiB — 536870912 bytes became "512 GiB used".
    """
    return float(1 << 20) if (total or 0) >= (1 << 30) else 1.0


def meminfo_mib() -> tuple[int, int] | None:
    """(used_mib, total_mib) from /proc/meminfo, with no platform gate.

    Deliberately ungated, unlike the device-tree probe used for *discovery*.
    When a card reports its memory as '[N/A]' that is itself the signal: the
    only NVIDIA parts that do so are the unified-memory ones (Jetson, Orin,
    GB10), where system RAM *is* the video memory. Requiring
    /proc/device-tree/model to also match a keyword added a second condition
    that fails in containers and on DGX OS images that name the model
    differently — and when it failed, the VRAM bar silently vanished.
    """
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable", "MemFree"):
                    info[k] = int(v.strip().split()[0])
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        if not total:
            return None
        return (total - avail) // 1024, total // 1024
    except (OSError, ValueError, IndexError):
        return None


def _to_float(s: Any) -> float | None:
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


# ── PCI marketing-name lookup (lazy, cached) ──────────────────────────────────
#
# amdgpu exposes `product_name` only on some ASICs/kernels, and it is often
# empty. hwdata's pci.ids is present on essentially every distro; parsing the
# ~2 MB file once for a single vendor:device pair is cheaper than shelling out
# to lspci every refresh.

_PCI_IDS_PATHS = (
    "/usr/share/hwdata/pci.ids",
    "/usr/share/misc/pci.ids",
    "/usr/share/pci.ids",
)
_pci_name_cache: dict[str, str] = {}
_pci_ids_missing = False


def pci_device_name(vendor: str, device: str) -> str | None:
    """Marketing name for a vendor:device pair (both 4 hex chars, lowercase)."""
    global _pci_ids_missing
    key = f"{vendor}:{device}"
    if key in _pci_name_cache:
        return _pci_name_cache[key]
    if _pci_ids_missing:
        return None
    path = next((p for p in _PCI_IDS_PATHS if os.path.exists(p)), None)
    if not path:
        _pci_ids_missing = True
        return None
    try:
        in_vendor = False
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                if not line.startswith("\t"):
                    in_vendor = line[:4].lower() == vendor
                    continue
                if (in_vendor and not line.startswith("\t\t")
                        and line[1:5].lower() == device):
                    name = line[5:].strip()
                    _pci_name_cache[key] = name
                    return name
    except OSError:
        _pci_ids_missing = True
    return None


# ── provider protocol ─────────────────────────────────────────────────────────
#
# A provider returns a list of dicts in the shape the renderer already expects:
#   name, util, mem_used, mem_total (MiB, stringly-typed), temp
# plus new optional keys: vendor, index, power, unified, gtt_used, gtt_total.
# Stringly-typed numerics are kept deliberately — '[N/A]' is a legitimate value
# from nvidia-smi on Tegra and the renderer already copes via to_float().


class GpuProvider:
    name = "base"
    # Provider names this one makes redundant when it answers. NVML and
    # nvidia-smi describe the same silicon; the registry concatenates every
    # answer, so without this a CUDA box would list each card twice.
    supersedes: frozenset[str] = frozenset()

    def collect(self) -> list[dict] | None:
        raise NotImplementedError


class AmdSysfsProvider(GpuProvider):
    """amdgpu telemetry straight out of /sys/class/drm/cardN/device."""

    name = "amd-sysfs"

    def __init__(self):
        self._cards: list[str] | None = None   # resolved device dirs, cached

    def _discover(self) -> list[str]:
        cards: list[tuple[str, str]] = []
        for entry in sorted(os.listdir(SYSFS_DRM)) if os.path.isdir(SYSFS_DRM) else []:
            if not _CARD_RE.match(entry):
                continue
            dev = os.path.join(SYSFS_DRM, entry, "device")
            vendor = (_read(os.path.join(dev, "vendor")) or "").lower()
            if vendor != "0x1002":                      # 0x1002 = AMD/ATI
                continue
            slot = ""
            uevent = _read(os.path.join(dev, "uevent")) or ""
            for line in uevent.splitlines():
                if line.startswith("PCI_SLOT_NAME="):
                    slot = line.split("=", 1)[1]
            cards.append((slot or entry, dev))
        cards.sort(key=lambda t: t[0])                  # stable across boots
        return [dev for _, dev in cards]

    def _name_for(self, dev: str) -> str:
        prod = _read(os.path.join(dev, "product_name"))
        if prod and prod not in ("", "0"):
            return prod
        uevent = _read(os.path.join(dev, "uevent")) or ""
        pci_id = ""
        for line in uevent.splitlines():
            if line.startswith("PCI_ID="):
                pci_id = line.split("=", 1)[1].strip()
        if pci_id and ":" in pci_id:
            ven, devid = (x.lower() for x in pci_id.split(":", 1))
            marketing = pci_device_name(ven, devid)
            if marketing:
                return marketing
            return f"AMD {pci_id}"
        return "AMD GPU"

    def _hwmon_temp_power(self, dev: str) -> tuple[str, str | None]:
        """(edge temp °C, power W) — junction/mem sensors are skipped."""
        temp, power = "N/A", None
        for hw in sorted(glob.glob(os.path.join(dev, "hwmon", "hwmon*"))):
            # Prefer the sensor labelled 'edge'; fall back to temp1.
            chosen = None
            for lbl_path in sorted(glob.glob(os.path.join(hw, "temp*_label"))):
                if (_read(lbl_path) or "").lower() == "edge":
                    chosen = lbl_path.replace("_label", "_input")
                    break
            chosen = chosen or os.path.join(hw, "temp1_input")
            mdeg = _read_int(chosen)
            if mdeg is not None:
                temp = f"{mdeg / 1000:.0f}"
            uw = _read_int(os.path.join(hw, "power1_average"))
            if uw is None:
                uw = _read_int(os.path.join(hw, "power1_input"))
            if uw is not None:
                power = f"{uw / 1_000_000:.0f}"
            if temp != "N/A":
                break
        return temp, power

    def collect(self) -> list[dict] | None:
        if self._cards is None:
            self._cards = self._discover()
        if not self._cards:
            self._cards = None      # re-probe next cycle (hotplug / late modprobe)
            return None

        out: list[dict] = []
        for idx, dev in enumerate(self._cards):
            vram_total = _read_int(os.path.join(dev, "mem_info_vram_total"))
            vram_used = _read_int(os.path.join(dev, "mem_info_vram_used"))
            gtt_total = _read_int(os.path.join(dev, "mem_info_gtt_total"))
            gtt_used = _read_int(os.path.join(dev, "mem_info_gtt_used"))
            busy = _read_int(os.path.join(dev, "gpu_busy_percent"))
            temp, power = self._hwmon_temp_power(dev)

            unified = False
            # APU: the "VRAM" is a BIOS carve-out (typically ≤ 1 GiB) and the
            # real working pool is GTT backed by system RAM.
            if vram_total is not None and vram_total <= 1 << 30 and gtt_total:
                unified = True
                mem_used, mem_total = (vram_used or 0) + (gtt_used or 0), gtt_total
            else:
                mem_used, mem_total = vram_used, vram_total

            if mem_total is None:
                continue

            gpu = {
                "vendor": "amd",
                "index": idx,
                "name": self._name_for(dev),
                "util": str(busy) if busy is not None else "N/A",
                "mem_used": f"{(mem_used or 0) / (1 << 20):.0f}",
                "mem_total": f"{mem_total / (1 << 20):.0f}",
                "temp": temp,
                "sysfs": dev,
            }
            if power is not None:
                gpu["power"] = power
            if unified:
                gpu["unified"] = True
            if gtt_total and not unified:
                gpu["gtt_used"] = f"{(gtt_used or 0) / (1 << 20):.0f}"
                gpu["gtt_total"] = f"{gtt_total / (1 << 20):.0f}"
            out.append(gpu)
        return out or None


class RocmSmiProvider(GpuProvider):
    """Fallback for setups where sysfs is unavailable (locked-down /sys, some
    container runtimes) but the ROCm userspace is installed.

    Schema-tolerant on purpose: ROCm renamed keys between 5.x and 6.x
    ('card0' → 'card:0', 'GPU use (%)' → 'gfx_activity'), and amd-smi changed
    them again. We fish for substrings instead of trusting exact keys.
    """

    name = "rocm-smi"

    def __init__(self, runner: Callable[[list[str], int], tuple[bool, str]]):
        self._run = runner

    @staticmethod
    def _pick(d: dict, *needles: str, exclude: tuple[str, ...] = ()) -> str | None:
        """First value whose key contains every needle and none of `exclude`.

        `exclude` matters: 'VRAM Total Used Memory (B)' contains both 'vram'
        and 'total', so without it the *total* lookup returns the used figure
        whenever that key happens to come first in the JSON object.
        """
        for k, v in d.items():
            kl = k.lower()
            if all(n in kl for n in needles) and not any(x in kl for x in exclude):
                return str(v)
        return None

    def collect(self) -> list[dict] | None:
        ok, out = self._run(
            ["rocm-smi", "--showuse", "--showmemuse", "--showtemp",
             "--showproductname", "--json"], 4)
        if not ok or not out:
            return None
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return None
        gpus = []
        for idx, (key, card) in enumerate(sorted(data.items())):
            if not isinstance(card, dict) or "card" not in key.lower():
                continue
            used = _to_float(self._pick(card, "vram", "used") or "") or 0.0
            total = _to_float(self._pick(card, "vram", "total", exclude=("used",)) or "") or 0.0
            scale = _mib_scale(total)
            util = self._pick(card, "gpu", "use") or self._pick(card, "activity")
            temp = self._pick(card, "temperature", "edge") or self._pick(card, "temp")
            name = (self._pick(card, "card", "series")
                    or self._pick(card, "product", "name") or f"AMD GPU {idx}")
            gpus.append({
                "vendor": "amd",
                "index": idx,
                "name": name,
                "util": (util or "N/A").strip("% "),
                # rocm-smi reports VRAM in bytes; some builds in MiB. Heuristic:
                # anything over 1 GiB-as-a-number is bytes.
                "mem_used": f"{used / scale:.0f}",
                "mem_total": f"{total / scale:.0f}",
                "temp": (temp or "N/A").strip("c° "),
            })
        return gpus or None


Runner = Callable[[list[str], int], tuple[bool, str]]


class NvidiaSmiProvider(GpuProvider):
    """nvidia-smi, on the host or inside the container.

    `attempts_fn` returns (label, runner) pairs to try in order — typically
    the host `run_cmd` and then an exec inside the Ollama container — so
    neither the docker CLI nor the Engine API leaks into this class. The
    first label that answers is memoized. Works unchanged on Windows:
    nvidia-smi.exe lives in System32 and is on PATH.
    """

    name = "nvidia-smi"

    def __init__(self, attempts_fn: Callable[[], list[tuple[str, Runner]]]):
        self._attempts_fn = attempts_fn
        self._good: str | None = None

    def _query(self, run: Runner) -> list[dict] | None:
        query = ("index,name,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw")
        ok, out = run(["nvidia-smi", f"--query-gpu={query}",
                       "--format=csv,noheader,nounits"], 3)
        if not ok or not out:
            return None
        gpus = []
        for line in out.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 6:
                continue
            gpus.append({
                "vendor": "nvidia",
                "index": int(f[0]) if f[0].isdigit() else len(gpus),
                "name": f[1],
                "util": f[2],
                "mem_used": f[3],
                "mem_total": f[4],
                "temp": f[5],
                **({"power": f[6]} if len(f) > 6 and _to_float(f[6]) is not None else {}),
            })
        return gpus or None

    def collect(self) -> list[dict] | None:
        attempts = self._attempts_fn()
        if self._good is not None:
            attempts = [a for a in attempts if a[0] == self._good] or attempts
        for label, run in attempts:
            gpus = self._query(run)
            if gpus:
                self._good = label
                return gpus
        self._good = None
        return None


# ── NVML via ctypes ───────────────────────────────────────────────────────────
#
# libnvidia-ml.so.1 ships with every NVIDIA driver, so this needs nothing the
# host does not already have — and it is what nvidia-smi itself calls. Benefits
# over forking nvidia-smi every cycle: no process spawn (a few ms vs ~100 ms),
# and nvmlDeviceGetComputeRunningProcesses, which nvidia-smi's --query-gpu
# cannot express. That call returns (host pid, bytes) per process, which is
# the only way to say *which card holds which runner* on a multi-GPU box.

class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _NvmlProcessV2(ctypes.Structure):
    # nvmlProcessInfo_v2_t / nvmlProcessInfo_t (CUDA 11.1+): natural alignment
    # puts usedGpuMemory at offset 8, total size 24.
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class _NvmlProcessV1(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]


NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1   # usedGpuMemory on WDDM / unsupported
NVML_LIB_NAMES = ("libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll")


def load_nvml() -> Any | None:
    """The NVML shared library, or None when no NVIDIA driver is installed."""
    for name in NVML_LIB_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    if IS_WINDOWS:
        for path in (r"C:\Windows\System32\nvml.dll",
                     r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return None


class NvmlProvider(GpuProvider):
    """NVIDIA telemetry straight from the driver library.

    Emits the same dict shape as NvidiaSmiProvider plus ``procs``: a list of
    ``{"pid": int, "mem_mib": float | None}`` for every compute process on the
    card, host PID namespace. Values the driver reports as unsupported
    (memory on unified parts like GB10, power on some laptops) come out as
    ``"[N/A]"`` — the same spelling nvidia-smi uses, so the registry's
    unified-memory patch keeps working unchanged.
    """

    name = "nvml"
    supersedes = frozenset({"nvidia-smi"})

    def __init__(self, loader: Callable[[], Any | None] = load_nvml):
        self._loader = loader
        self._lib: Any | None = None
        self._inited = False
        self._failed = False

    def _init(self) -> bool:
        if self._inited:
            return True
        if self._failed:
            return False
        lib = self._loader()
        if lib is None or getattr(lib, "nvmlInit_v2", None) is None:
            self._failed = True
            return False
        if lib.nvmlInit_v2() != NVML_SUCCESS:
            self._failed = True
            return False
        self._lib = lib
        self._inited = True
        return True

    def _processes(self, dev) -> list[dict]:
        """(pid, bytes) per compute process; empty when the call is unsupported."""
        lib = self._lib
        for fname, struct in (("nvmlDeviceGetComputeRunningProcesses_v3", _NvmlProcessV2),
                              ("nvmlDeviceGetComputeRunningProcesses_v2", _NvmlProcessV2),
                              ("nvmlDeviceGetComputeRunningProcesses", _NvmlProcessV1)):
            fn = getattr(lib, fname, None)
            if fn is None:
                continue
            count = ctypes.c_uint(0)
            rc = fn(dev, ctypes.byref(count), None)
            if rc == NVML_SUCCESS and count.value == 0:
                return []
            if rc not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
                return []
            n = max(count.value, 1) + 8            # headroom for a spawn in between
            infos = (struct * n)()
            count = ctypes.c_uint(n)
            rc = fn(dev, ctypes.byref(count), infos)
            if rc != NVML_SUCCESS:
                return []
            out = []
            for i in range(count.value):
                used = infos[i].usedGpuMemory
                out.append({
                    "pid": int(infos[i].pid),
                    "mem_mib": (None if used in (0, NVML_VALUE_NOT_AVAILABLE)
                                else round(used / (1 << 20))),
                })
            return out
        return []

    def collect(self) -> list[dict] | None:
        if not self._init():
            return None
        lib = self._lib
        count = ctypes.c_uint(0)
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != NVML_SUCCESS:
            return None
        gpus: list[dict] = []
        for idx in range(count.value):
            dev = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(dev)) != NVML_SUCCESS:
                continue
            name_buf = ctypes.create_string_buffer(96)
            name = "NVIDIA GPU"
            if lib.nvmlDeviceGetName(dev, name_buf, 96) == NVML_SUCCESS:
                name = name_buf.value.decode("utf-8", "replace") or name

            util = _NvmlUtilization()
            util_s = (str(util.gpu)
                      if lib.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(util)) == NVML_SUCCESS
                      else "[N/A]")
            mem = _NvmlMemory()
            if lib.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(mem)) == NVML_SUCCESS and mem.total:
                mem_used, mem_total = f"{mem.used / (1 << 20):.0f}", f"{mem.total / (1 << 20):.0f}"
            else:
                mem_used = mem_total = "[N/A]"
            temp = ctypes.c_uint(0)
            temp_s = (str(temp.value)
                      if lib.nvmlDeviceGetTemperature(dev, 0, ctypes.byref(temp)) == NVML_SUCCESS
                      else "[N/A]")
            mw = ctypes.c_uint(0)
            gpu = {
                "vendor": "nvidia",
                "index": idx,
                "name": name,
                "util": util_s,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "temp": temp_s,
                "procs": self._processes(dev),
            }
            if lib.nvmlDeviceGetPowerUsage(dev, ctypes.byref(mw)) == NVML_SUCCESS:
                gpu["power"] = f"{mw.value / 1000:.2f}"
            gpus.append(gpu)
        return gpus or None


class TegraUnifiedProvider(GpuProvider):
    """Jetson / Orin / GB10 Spark: nvidia-smi exists but memory reads [N/A]."""

    name = "tegra-unified"

    def __init__(self, reader: Callable[[], tuple[str, int, int] | None]):
        self._reader = reader

    def collect(self) -> list[dict] | None:
        uni = self._reader()
        if not uni:
            return None
        model, used_mib, total_mib = uni
        return [{
            "vendor": "nvidia",
            "index": 0,
            "name": model,
            "util": "N/A",
            "mem_used": str(used_mib),
            "mem_total": str(total_mib),
            "temp": "N/A",
            "unified": True,
        }]


# ── registry ──────────────────────────────────────────────────────────────────

class GpuMonitor:
    """Runs every provider that has ever answered, concatenating results.

    Unlike v0.3.0's single memoized strategy, a mixed CUDA + ROCm box shows
    both vendors. Providers that return None are retried at a slower cadence
    so a GPU-less host does not fork `nvidia-smi` and `rocm-smi` every cycle.
    """

    RETRY_AFTER = 30.0  # seconds before re-probing a provider that came back None

    def __init__(self, providers: list[GpuProvider], clock: Callable[[], float]):
        self._providers = providers
        self._clock = clock
        self._dead: dict[str, float] = {}   # provider name -> next retry time

    def collect(self) -> list[dict] | None:
        now = self._clock()
        out: list[dict] = []
        superseded: set[str] = set()
        for p in self._providers:
            if p.name in superseded:
                continue
            deadline = self._dead.get(p.name)
            if deadline is not None and now < deadline:
                continue
            try:
                gpus = p.collect()
            except Exception:
                gpus = None
            if gpus:
                self._dead.pop(p.name, None)
                out.extend(gpus)
                superseded |= p.supersedes
            else:
                self._dead[p.name] = now + self.RETRY_AFTER
        if not out:
            return None
        # Patch nvidia-smi's '[N/A]' memory. No platform gate: a card that
        # cannot report its own memory is a unified-memory part by definition,
        # and system RAM is the honest answer for it.
        for g in out:
            if g["vendor"] == "nvidia" and _to_float(g["mem_total"]) is None:
                mem = meminfo_mib()
                if mem:
                    g["mem_used"], g["mem_total"] = str(mem[0]), str(mem[1])
                    g["unified"] = True
        # De-dup: Tegra provider and nvidia-smi describe the same silicon.
        if any(g.get("unified") and g["vendor"] == "nvidia" for g in out):
            seen_unified = False
            deduped = []
            for g in out:
                if g["vendor"] == "nvidia" and g.get("unified"):
                    if seen_unified:
                        continue
                    seen_unified = True
                deduped.append(g)
            out = deduped
        return out
