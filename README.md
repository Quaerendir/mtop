# mtop

**htop for Ollama** — a curses-based TUI that monitors your models, GPU, and the Ollama server (in Docker, Podman *or* bare-metal) in real time. Zero flicker. Zero dependencies beyond Python 3.10+.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey)

<!-- TODO: Replace with actual screenshot/gif -->
<!-- ![mtop screenshot](screenshot.png) -->

## Why?

There are web dashboards, Prometheus exporters, and chat TUIs for Ollama. But there's no **terminal monitor** — something you SSH into a box and just run, like `htop` or `nvtop`, to see what models are loaded, how much VRAM they're eating, and whether the container is healthy.

`mtop` fills that gap. One file, one command, pure stdlib Python. It auto-detects whether Ollama runs in a Docker/Podman container or as a bare-metal process (systemd or a manual `ollama serve`) and monitors it either way.

## Features

- **Zero-flicker display** — curses double-buffered rendering, no `clear` + print loops
- **Effective inference config** — `RUNNERS` section parses each runner's argv: context, batch, flash attention, KV cache dtype, layer offload. The only place these are observable
- **Whole-tree process stats** — the server *and* its per-model `ollama runner` subprocesses, with PSS accounting where available
- **Server config** — the `OLLAMA_*` / `CUDA_VISIBLE_DEVICES` / `HSA_*` environment the server was started with, from the container, the process, or the systemd unit; next to RUNNERS this is "what was set" vs "what was negotiated"
- **Which model on which card** — NVIDIA per-process VRAM via NVML (ctypes, no `nvidia-smi` fork) joined to the runner PIDs: a `GPU` column in RUNNERS and a `procs:` line under each card
- **Loaded models** — name, VRAM/RAM split, context length, processor type, TTL countdown
- **Container health** — status indicator (●/✗/○), uptime, CPU & memory with progress bars
- **Multi-vendor GPU monitoring** — NVIDIA via NVML (`nvidia-smi` as fallback) *and* AMD via sysfs, side by side on the same host; utilization, VRAM/GTT, temperature, power draw
- **AMD without ROCm** — telemetry comes from `/sys/class/drm/card*/device`, so a bare `amdgpu` driver is enough; `rocm-smi` is only a fallback
- **Jetson / Tegra / NVIDIA Spark** — automatic fallback to unified memory via `/proc/meminfo`
- **Non-blocking UI** — all I/O (docker, nvidia-smi, HTTP) runs in a background collector thread; the interface stays responsive at 100 ms even when the API hangs, and stale data is flagged
- **Interactive** — `q` to quit, `+`/`-` to adjust refresh interval, `o` to toggle raw `ollama ps`
- **Scriptable** — `--json` one-shot mode for cron, Prometheus textfile collectors, or Ansible facts (exit code 1 on unhealthy)
- **API-only mode** — `--no-docker` for monitoring remote Ollama instances without local docker calls
- **cgroup-aware CPU bar** — normalizes against the container's `--cpus`/quota limit, not the host core count
- **Docker *and* bare-metal** — auto-detects the source: a Docker container, a systemd `ollama.service`, or a manual `ollama serve`; monitors process CPU/MEM via `/proc` (Linux, no root) or `ps`/`sysctl` (macOS)
- **Docker Engine API over the socket** — no `docker` binary needed: mtop talks to `/var/run/docker.sock` (or a rootless / Podman socket, or `$DOCKER_HOST`) with stdlib `http.client`; the `docker`/`podman` CLI is only a fallback. One-shot stats mean no 2 s `docker stats` stall per cycle
- **Podman** — same code path via the compat API socket or the `podman` CLI
- **Respects `$OLLAMA_HOST`** — works with remote Ollama instances out of the box
- **Zero external dependencies** — only Python stdlib (`curses`, `urllib`, `json`, `subprocess`)

## Quick Start

### One-liner (no install)

```bash
curl -fsSL https://github.com/Quaerendir/mtop/releases/latest/download/mtop.py -o mtop.py
chmod +x mtop.py
./mtop.py
```

### pip install

> Not yet published to PyPI — coming with the first tagged release. Until then, use the one-liner or install from source.

```bash
pip install ollama-mtop   # (pending)
mtop
```

### From source

```bash
git clone https://github.com/Quaerendir/mtop.git
cd mtop
pip install -e .
mtop
```

### Run directly from a clone (no install)

```bash
git clone https://github.com/Quaerendir/mtop.git
cd mtop
PYTHONPATH=src python -m mtop
```

## Usage

```
mtop [-c CONTAINER] [-i INTERVAL] [-u URL] [-m MODE] [--runtime RT] [--no-gpu] [--json] [-V] [-h]

Options:
  -c, --container NAME   Docker container name (default: ollama)
  -m, --mode MODE        Data source: auto|docker|local|api (default: auto)
                         local = bare-metal `ollama serve` (systemd/proc/ps)
                         api   = models only, no host resource stats
  -i, --interval SECS    Refresh interval in seconds (default: 1.0)
  -u, --api-url URL      Ollama API base URL (default: $OLLAMA_HOST or http://localhost:11434)
                         Scheme-less values (gpu-rig:11434) are accepted, like Ollama itself
      --runtime RT       How to reach the container runtime: auto|api|cli (default: auto)
                         api = Engine API on $DOCKER_HOST or a docker/podman socket
                         cli = docker/podman subprocesses
      --no-gpu           Disable GPU monitoring section
      --no-runners       Hide the RUNNERS section (effective inference config)
      --no-env           Hide the SERVER CONFIG section (OLLAMA_* environment)
      --no-docker        API-only mode: skip all docker calls (remote instances)
      --json             Print one snapshot as JSON and exit (exit 1 on unhealthy)
  -V, --version          Show version
  -h, --help             Show help
```

### Examples

```bash
# Monitor a custom container name
mtop -c my-ollama

# Slower refresh for remote/metered connections
mtop -i 5

# Bare-metal Ollama (systemd service or `ollama serve` in a terminal)
mtop --mode local

# Let mtop figure it out (docker? systemd? manual? — it probes in that order)
mtop

# Monitor a remote Ollama instance — API only, no local docker/GPU noise
mtop -u 192.168.1.100:11434 --mode api

# Podman (rootless): the socket is found automatically; or point at it
DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock mtop

# mtop inside a container, with only the socket mounted — no docker CLI needed
docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
  --network host python:3.12-slim sh -c \
  "curl -fsSL https://github.com/Quaerendir/mtop/releases/latest/download/mtop.py -o m.py && python m.py"

# One-shot health/state snapshot for scripting
mtop --json | jq '.models[].name'

# Using OLLAMA_HOST environment variable
export OLLAMA_HOST=http://gpu-rig:11434
mtop
```

### Interactive Keys

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `+` | Decrease refresh interval (faster) |
| `-` | Increase refresh interval (slower) |
| `o` | Toggle raw `ollama ps` section |
| `r` | Toggle the `RUNNERS` section |
| `e` | Toggle the `SERVER CONFIG` section |

## Display Layout

```
─── mtop v0.4.2 — Ollama Model Monitor ───
host: gpu-rig     container: ● ollama     up: 3d 14h     2026-03-11 15:42:01
────────────────────────────────────────────────────────────────────────────────
CONTAINER RESOURCES
   CPU  [████░░░░░░░░░░░░░░░░░░░░░░░░░░]  12.3%
   MEM  [██████████████░░░░░░░░░░░░░░░░]  45.2%     14.2GiB / 31.4GiB
GPU
   [0] NVIDIA GeForce RTX 4090  42°C
     UTIL [████████░░░░░░░░░░░░░░░░░]  32.0%
     VRAM [██████████████████░░░░░░░]  72.4%     17382 / 24000 MiB
LOADED MODELS
   MODEL                                VRAM        RAM         CTX       PROCESSOR         EXPIRES
   ────────────────────────────────────────────────────────────────────────────────────────────────
   qwen2.5-coder:32b-instruct-q8_0     18.42 G     0.00 G      32768     100% GPU          4m 32s left
OLLAMA PS (raw)
   NAME                                SIZE       PROCESSOR    UNTIL
   qwen2.5-coder:32b-instruct-q8_0     19.8 GB    100% GPU     4 minutes from now
```

## Supported Platforms

| Platform | GPU Monitoring | Notes |
|----------|---------------|-------|
| Linux x86_64 + NVIDIA | ✅ Full | NVML via ctypes; `nvidia-smi` on host or in container as fallback |
| NVIDIA Jetson / Orin | ✅ Unified memory | Falls back to `/proc/meminfo` |
| NVIDIA GB10 Spark | ✅ Unified memory | Tegra-based, same fallback |
| Linux + AMD (amdgpu) | ✅ Full | sysfs — no ROCm install required |
| AMD APU (780M, Strix) | ✅ Unified memory | GTT pool, not the tiny VRAM carve-out |
| Linux without GPU | ✅ (no GPU section) | Use `--no-gpu` to hide the section |
| Ollama in Podman | ✅ Full | compat API on the Podman socket, or the `podman` CLI |
| Bare-metal Ollama (systemd) | ✅ process stats | `--mode local`; CPU/MEM from `/proc`, no root needed |
| Manual `ollama serve` | ✅ process stats | auto-detected via `/proc` cmdline scan |
| macOS | ⚠️ Partial | `--mode local` monitors the process via `ps`/`sysctl`; GPU (Metal) not yet supported |
| WSL2 | ⚠️ Partial | Works if Docker + nvidia-container-toolkit configured |

## Requirements

- **Python 3.10+** (uses `match`-era type hints like `list[str]`, `X | Y`)
- **Docker or Podman** for container monitoring — access to the socket is enough, the CLI is optional
- **Ollama** in a container, as a bare-metal process, or reachable via API
- **NVIDIA driver** (optional, for GPU stats — `libnvidia-ml.so.1`, or `nvidia-smi` as fallback)

## Roadmap

- [ ] Record terminal sessions with `asciinema` for README gif
- [x] AMD GPU support (sysfs first, `rocm-smi` fallback)
- [ ] Apple Silicon GPU stats (via `powermetrics`)
- [ ] Model pull progress tracking
- [ ] Multiple container / multi-host support
- [x] Docker Engine API over the socket, Podman support
- [x] Configurable layout (raw `ollama ps` toggle; more sections to follow)
- [ ] Model actions — unload on keypress (`keep_alive: 0`), extend TTL
- [ ] Sparkline history for CPU/GPU utilization (braille chars, stdlib deque)
- [x] systemd/bare-metal Ollama support (process stats via `/proc`, no Docker required)
- [ ] Log panel (tail Ollama container logs)
- [x] Effective inference config per runner (context, flash attention, KV dtype)
- [ ] Request rate / tokens-per-second from Ollama API
- [x] Effective server environment (`OLLAMA_*`) per source
- [x] Runner → GPU mapping via NVML per-process memory

## Contributing

PRs welcome. Keep it stdlib-only — the zero-dependency constraint is a feature, not a limitation.

```bash
git clone https://github.com/Quaerendir/mtop.git
cd mtop
pip install -e ".[dev]"      # + pytest, ruff
# hack on src/mtop/*.py
mtop

ruff check src tools tests
pytest                       # stdlib-only fakes: no GPU, docker or Ollama needed

# regenerate the single-file artifact shipped with releases
python tools/bundle.py       # -> dist/mtop.py
```

`dist/mtop.py` is generated — never edit it by hand. CI builds it on every
push and attaches it to the GitHub release when a `v*` tag is pushed.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built as a collaboration between a human homelab geek and Claude (Anthropic) during a late-night infrastructure session. The original bash prototype migrated to Python/curses because fighting `tput` and `jq` in a loop was getting old.
