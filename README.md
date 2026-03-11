# mtop

**htop for Ollama** — a curses-based TUI that monitors your models, GPU, and Docker container in real time. Zero flicker. Zero dependencies beyond Python 3.10+.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey)

<!-- TODO: Replace with actual screenshot/gif -->
<!-- ![mtop screenshot](screenshot.png) -->

## Why?

There are web dashboards, Prometheus exporters, and chat TUIs for Ollama. But there's no **terminal monitor** — something you SSH into a box and just run, like `htop` or `nvtop`, to see what models are loaded, how much VRAM they're eating, and whether the container is healthy.

`mtop` fills that gap. One file, one command, pure stdlib Python.

## Features

- **Zero-flicker display** — curses double-buffered rendering, no `clear` + print loops
- **Loaded models** — name, VRAM/RAM split, context length, processor type, TTL countdown
- **Container health** — status indicator (●/✗/○), uptime, CPU & memory with progress bars
- **GPU monitoring** — NVIDIA desktop GPUs via `nvidia-smi`, with utilization and VRAM bars
- **Jetson / Tegra / NVIDIA Spark** — automatic fallback to unified memory via `/proc/meminfo`
- **Interactive** — `q` to quit, `+`/`-` to adjust refresh interval on the fly
- **Docker-aware** — talks to both the Ollama API and `docker exec ollama ps`
- **Respects `$OLLAMA_HOST`** — works with remote Ollama instances out of the box
- **Zero external dependencies** — only Python stdlib (`curses`, `urllib`, `json`, `subprocess`)

## Quick Start

### One-liner (no install)

```bash
curl -fsSL https://raw.githubusercontent.com/Quaerendir/mtop/main/src/mtop/__init__.py -o mtop.py
chmod +x mtop.py
./mtop.py
```

### pip install

```bash
pip install ollama-mtop
mtop
```

### From source

```bash
git clone https://github.com/Quaerendir/mtop.git
cd mtop
pip install -e .
mtop
```

### Run directly (no install needed)

```bash
python -m mtop
```

## Usage

```
mtop [-c CONTAINER] [-i INTERVAL] [-u URL] [--no-gpu] [-V] [-h]

Options:
  -c, --container NAME   Docker container name (default: ollama)
  -i, --interval SECS    Refresh interval in seconds (default: 1.0)
  -u, --api-url URL      Ollama API base URL (default: $OLLAMA_HOST or http://localhost:11434)
      --no-gpu           Disable GPU monitoring section
  -V, --version          Show version
  -h, --help             Show help
```

### Examples

```bash
# Monitor a custom container name
mtop -c my-ollama

# Slower refresh for remote/metered connections
mtop -i 5

# Monitor a remote Ollama instance, skip GPU
mtop -u http://192.168.1.100:11434 --no-gpu

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

## Display Layout

```
─── mtop v0.1.0 — Ollama Model Monitor ───
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
   MODEL                                VRAM        RAM         CTX       PROCESSOR       EXPIRES
   ──────────────────────────────────────────────────────────────────────────────────────────────
   qwen2.5-coder:32b-instruct-q8_0     18.42 G     0.00 G      32768     GPU             4m 32s left
OLLAMA PS (raw)
   NAME                                SIZE       PROCESSOR    UNTIL
   qwen2.5-coder:32b-instruct-q8_0     19.8 GB    100% GPU     4 minutes from now
```

## Supported Platforms

| Platform | GPU Monitoring | Notes |
|----------|---------------|-------|
| Linux x86_64 + NVIDIA | ✅ Full | `nvidia-smi` on host or in container |
| NVIDIA Jetson / Orin | ✅ Unified memory | Falls back to `/proc/meminfo` |
| NVIDIA GB10 Spark | ✅ Unified memory | Tegra-based, same fallback |
| Linux without GPU | ✅ (no GPU section) | Use `--no-gpu` to hide the section |
| macOS | ⚠️ Partial | curses works, no `nvidia-smi`; Docker Desktop only |
| WSL2 | ⚠️ Partial | Works if Docker + nvidia-container-toolkit configured |

## Requirements

- **Python 3.10+** (uses `match`-era type hints like `list[str]`, `X | Y`)
- **Docker** (for container monitoring)
- **Ollama** running in a Docker container (or accessible via API)
- **nvidia-smi** (optional, for GPU stats)

## Roadmap

- [ ] Record terminal sessions with `asciinema` for README gif
- [ ] AMD ROCm GPU support (`rocm-smi`)
- [ ] Apple Silicon GPU stats (via `powermetrics`)
- [ ] Model pull progress tracking
- [ ] Multiple container / multi-host support
- [ ] Configurable layout (hide/show sections)
- [ ] Log panel (tail Ollama container logs)
- [ ] Request rate / tokens-per-second from Ollama API

## Contributing

PRs welcome. Keep it stdlib-only — the zero-dependency constraint is a feature, not a limitation.

```bash
git clone https://github.com/Quaerendir/mtop.git
cd mtop
pip install -e .
# hack on src/mtop/__init__.py
mtop
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built as a collaboration between a human homelab geek and Claude (Anthropic) during a late-night infrastructure session. The original bash prototype migrated to Python/curses because fighting `tput` and `jq` in a loop was getting old.
