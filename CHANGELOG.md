# Changelog

## 0.3.0 — 2026-08-06

### Added
- **Bare-metal / local Ollama monitoring.** `--mode local` monitors an Ollama
  server running outside Docker — the official systemd `ollama.service`, or a
  manual `ollama serve`. Process CPU/MEM come from `/proc/<pid>` on Linux
  (world-readable, no root, works regardless of launch method) and from
  `ps`/`sysctl` on macOS. systemd is used only for discovery/status (active/
  activating/failed + MainPID); the numbers always come from `/proc` to avoid
  the "MemoryAccounting is off" and locale-timestamp pitfalls.
- **`--mode {auto,docker,local,api}`** (default `auto`). Auto probes Docker
  first, then a bare-metal process, else falls back to API-only — and keeps
  trying to upgrade from the api fallback each cycle, so starting mtop before
  Ollama is up self-heals. Docker is probed before /proc because a
  containerized `ollama serve` is also visible in host /proc.
- CPU-usage sampling for the local process (Δ of utime+stime ticks across
  cycles, keyed by pid so a restart resets the baseline instead of spiking).
- Header shows `ollama: ● serve · pid <N>` and a `PROCESS RESOURCES` section
  in local mode; the `o` raw-`ollama ps` toggle works here too (runs
  `ollama ps` directly instead of via `docker exec`).

### Changed
- Default source is now `auto` instead of docker-only. Docker hosts still
  resolve to docker; hosts without a container now find the local process
  instead of showing "container not found".
- `--no-docker` is now an alias for `--mode api` (unchanged behavior), but an
  explicit `--mode` wins over it.
- Internal: `docker_stats` snapshot key → `res_stats`; `render_docker_stats`
  → `render_resources` (mode-agnostic).

### Notes
- Ollama mmaps its GGUF model files, so the local process VmRSS includes
  resident mmapped model pages that also live in the page cache — the MEM bar
  can read ≈ model size. This is the honest footprint, just be aware it
  overlaps buffers/cache. On unified-memory boxes (GB10 Spark, Apple Silicon)
  the process RSS and the GPU "VRAM" measure the same physical bytes.

## 0.2.0 — 2026-07-05

### Architecture
- **Background collector thread.** All blocking I/O (`docker inspect`/`stats`/`exec`,
  `nvidia-smi`, Ollama API calls with up to 5 s timeouts) moved out of the render
  loop into a daemon collector publishing immutable snapshots. The curses loop
  polls keys at a fixed 100 ms and only draws — a hung API or slow docker daemon
  can no longer freeze the UI. Stale snapshots (> 3× interval) are flagged in the
  header.

### Fixed
- `OLLAMA_HOST` values without a scheme (`gpu-rig:11434`, `0.0.0.0:11434`) —
  valid for Ollama itself — crashed urllib. Now normalized to `http://`.
- Unified-memory platforms (GB10 Spark, Jetson/Orin) where `nvidia-smi` exists
  but reports memory as `[N/A]`: the VRAM bar silently vanished. Memory is now
  patched from `/proc/meminfo` while numeric util/temp from `nvidia-smi` is kept.
- CPU bar normalized against the container's effective CPU limit
  (`--cpus` / quota via `HostConfig`) instead of the host core count.
- `GPU_REFRESH`/`STATS_REFRESH` caps were frozen at startup; runtime `+`/`-`
  interval changes now propagate to the slow-path cadence.
- Detail strings (CPU %, MEM usage, VRAM MiB) were drawn at hardcoded columns
  (x=60/62) and overlapped bars on terminals < ~80 cols; now right-aligned to
  the frame and dropped cleanly when there is no room.
- Dead install paths in README: curl one-liner pointed at a nonexistent `main`
  branch; `python -m mtop` claimed to work without install (src-layout).
- Redundant `nodelay(True)` (overridden by `timeout()`) removed.

### Added
- `--json` — one-shot snapshot as JSON on stdout, exit code 1 on unhealthy.
  For cron, Prometheus textfile collectors, Ansible facts.
- `--no-docker` — API-only mode for remote Ollama instances; skips all local
  docker calls instead of mixing remote models with local container stats.
- `o` key — toggle the raw `ollama ps` section (default off; it duplicated
  `/api/ps` at the cost of a `docker exec` per refresh).
- GPU probe strategy memoization (host vs `docker exec` vs unified) — no more
  re-forking failed `nvidia-smi` probes every cycle on GPU-less boxes; memo
  resets on failure so driver restarts are picked up.

### Internal
- Type hints unified on `X | None` (3.10+ baseline), `typing.Optional` dropped.
- `[tool.ruff]` config added (line-length 100, py310).

## 0.1.0

Initial release.
