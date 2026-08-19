# Changelog

## 0.4.0 — 2026-08-18

### Fixed
- **Bare-metal MEM/CPU measured the wrong process.** Ollama runs the model in a
  separate child process, so the server found via `serve` in argv holds only the
  server itself. Measured on macOS with a 5.2 GB model resident: `ollama serve`
  reported 64 MiB while its child held 9.47 GiB — a 151x understatement. Stats
  are now rolled up over the whole process tree (`process_tree()`), on Linux via
  `/proc` and on macOS from a single `ps -ax` snapshot.
- The tree walk matches on **ppid, never on process name**. That child has been
  called `ollama_llama_server`, then `ollama runner`, and is `llama-server` in
  current builds — any name-based heuristic silently misses it, and `ps -C
  ollama` misses it twice over because `comm` is truncated to 15 chars.
- Docker mode was never affected: `docker stats` accounts the whole cgroup, and
  the runner lives in the same one.
- CPU deltas are computed per-pid and summed only over pids present in both
  samples. Summing tree totals would report a spike the cycle a runner spawns
  (its accumulated ticks appear at once) and a clamped-to-zero dip when one
  exits.

### Added
- **`RUNNERS` section** — the effective inference config, parsed from each
  runner process's argv. `/api/ps` reports the context length and stops;
  everything Ollama negotiated between the Modelfile, the environment and its
  own heuristics exists only on that command line. Columns: PID, model, CTX,
  BATCH, FA, KV cache dtype, NGL, MEM, plus `mmproj`/`O_DIRECT` markers.
  Toggle with `r`, disable with `--no-runners`.
  Concretely: `--flash-attn on` vs `auto` is how you tell whether
  `OLLAMA_FLASH_ATTENTION=1` actually reached the server — two hosts with the
  same variable set were observed producing different flags.
- Runner discovery works in docker mode too: the container's init PID comes
  from the existing `docker inspect` call and its tree is walked on the host
  `/proc` (same kernel, no exec). Falls back to a single `docker exec` reading
  every cmdline in the container's PID namespace, at the cost of per-runner RSS
  — for when mtop is itself containerized without the host PID namespace.
- **VRAM and HOST are separate columns**, because they measure different things.
  VRAM is Ollama's `size_vram` for the matched model; HOST is the runner
  process's resident set. They converge on CPU inference and diverge on
  accelerators: weights allocated through CUDA/ROCm/Metal are not charged to the
  process, so a GB10 Spark holding an 82 GB model reports ~7.5 GiB of host RSS
  while `/proc/meminfo` — and therefore mtop's unified-memory GPU section — sees
  the full allocation. The gap between the columns is the device-memory
  footprint. Collapsing both into one `MEM` column, as the first cut did, made
  the same header mean host RSS on a Spark and near-total footprint on Metal.
- Blob-to-tag mapping is best effort and says so: the `--model` path carries a
  blob digest while the API exposes manifest digests, so there is no direct
  join. Matched on context length when it uniquely identifies a loaded model,
  then on a single unmatched runner facing a single unmatched model, otherwise
  the short digest is shown. A wrong tag on a monitoring screen is worse than
  no tag.
- **AMD GPU support, from sysfs.** `/sys/class/drm/card*/device` gives
  `gpu_busy_percent`, VRAM/GTT usage, hwmon temperature and power draw with no
  ROCm install, no fork per cycle, and no exposure to the `rocm-smi` JSON
  schema churn across ROCm 5.x/6.x/amd-smi. Works on a bare `amdgpu` driver and
  inside containers, where `/sys` is already bind-mounted.
- `rocm-smi` fallback for hosts where sysfs is unreadable but the ROCm
  userspace is installed. Deliberately schema-tolerant: it matches key
  substrings instead of trusting exact key names.
- **Multi-vendor GPU registry** (`mtop.gpu.GpuMonitor`). Every provider runs
  and the results are concatenated, so a host with an NVIDIA card *and* a
  Radeon shows both. Providers that return nothing are re-probed at a 30 s
  cadence rather than every cycle.
- APU detection: on integrated Radeons `mem_info_vram_total` is a BIOS carve-out
  (typically ≤ 1 GiB) and the working pool is GTT. Previously such a card would
  have rendered as "512 MiB, 900% full"; it is now reported as unified memory.
- GTT bar for discrete cards, shown only when something is actually spilled
  there — the usual first sign that a model did not fit in VRAM.
- Power draw (W) in the GPU header line, where the vendor exposes it.
- PSS-based memory accounting when `/proc/<pid>/smaps_rollup` is readable for
  every process in the tree. Runners map the same GGUF and the same
  CUDA/ROCm libraries, so summing VmRSS double-counts the shared pages. The
  MEM detail is tagged `(pss)` or `(rss)` so the number is not ambiguous.
- Header shows the runner count next to the server pid (`serve · pid 4711 +2r`).
- `tools/bundle.py` — flattens the package into a single runnable `dist/mtop.py`
  for the curl one-liner, which the module split would otherwise have broken.
  Submodules are embedded as exec'd module objects, not concatenated, so
  same-named private helpers cannot clobber each other.

### Changed
- GPU entries in `--json` gained `vendor`, `index`, and optionally `power`,
  `unified`, `gtt_used`, `gtt_total`, `sysfs`. Consumers keying on position
  alone should switch to `vendor`/`index`.
- `res_stats` in `--json` gained `procs` and `mem_kind`; the snapshot gained a
  top-level `runners` list.
- `pid` in `--json` is now populated in docker mode as well, where it is the
  container's init PID (it was always `null` before).
- The single memoized GPU probe strategy (`_gpu_mode`) is gone, along with
  `_nvidia_smi()` and `_patch_unified()`; that logic moved to `mtop/gpu.py`.

### Known limitations
- `/api/ps` does not say which device holds a model, and one Ollama server
  binds one backend — a mixed CUDA + ROCm host needs two instances on two
  ports. Correlating models with cards is deferred to multi-endpoint support.
- Windows is not yet a supported platform for `--mode local` (no `curses` in
  CPython on Windows, no `/proc`). Tracked separately.

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
