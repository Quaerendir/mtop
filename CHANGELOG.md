# Changelog

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
