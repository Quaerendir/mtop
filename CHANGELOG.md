# Changelog

## 0.8.0 — 2026-09-13

### Added
- **`--prometheus`**: the snapshot in Prometheus text exposition format, one
  shot, exit code as `--json`. Metrics are prefixed `mtop_` and grouped by
  a second token: `mtop_up`, `mtop_api_up{endpoint}`,
  `mtop_models_loaded`, `mtop_model_{size,vram}_bytes`,
  `mtop_model_context_length`, `mtop_model_expires_seconds` (`+Inf` for
  `keep_alive -1`), `mtop_server_{cpu_percent,cpu_limit_cores,
  memory_bytes,memory_limit_bytes,processes,uptime_seconds}`,
  `mtop_runner_info{pid,model,engine,ctx,batch,flash_attn,kv_cache,gpu,…}`,
  `mtop_runner_{rss,vram,gpu_memory}_bytes`, `mtop_gpu_{info,
  utilization_percent,memory_used_bytes,memory_total_bytes,
  temperature_celsius,power_watts,gtt_*,process_memory_bytes}`. Output is
  validated against the official `prometheus_client` parser.
- **`--watch`**: keep emitting every `-i` seconds until Ctrl-C.
  `--json --watch` prints NDJSON (one compact object per line);
  `--prometheus --watch` re-renders the exposition.
- **`-o FILE`**: Prometheus output replaces the file atomically (tmp +
  rename), NDJSON is appended. `mtop --prometheus --watch -o
  /var/lib/node_exporter/textfile/mtop.prom` is therefore a complete
  exporter setup with no cron, no port and no dependency.
- Numeric twins of the display strings in `--json`: `res_stats.cpu_pct`,
  `mem_used_bytes`, `mem_limit_bytes` on every path (parsed from the docker
  CLI strings where that is the source) and a top-level `uptime_sec`.

### Changed
- `mtop.export` holds the ISO-8601 and size parsers the collector and the
  exporter share; `relative_time` now uses the same parser as the exporter.
- `--json --watch | head` exits quietly instead of printing a
  BrokenPipeError at interpreter shutdown.

## 0.7.0 — 2026-09-13

### Added
- **Several endpoints.** `-u` is repeatable, optionally labelled
  (`-u rig=http://gpu-rig:11434`). The first is the *primary* — the one the
  docker/local source, the runner match and the raw `ollama ps` refer to;
  the rest are API-only and get their own LOADED MODELS table with the
  endpoint's label and version. All endpoints are polled in parallel, so a
  dead remote costs its timeout once per cycle, not once per endpoint in
  sequence. `--json` gains an `endpoints` list; the top-level `models` /
  `models_ok` / `api_url` keep describing the primary. This is the answer to
  the mixed CUDA + ROCm host from the 0.4.0 notes: two instances, two `-u`.
- **Auth.** `-H 'Name: value'` (repeatable) adds a header to every API
  request; `OLLAMA_API_KEY` becomes `Authorization: Bearer …`, the same
  variable the ollama CLI uses; credentials in a URL
  (`https://user:pw@host`) become basic auth, which urllib does not do on
  its own. HTTP failures now read `HTTP 401 Unauthorized` on screen instead
  of a bare urllib reason.
- **TLS.** `--insecure` skips certificate verification, `--cacert FILE`
  verifies against a private CA — for Ollama behind a reverse proxy with an
  internal certificate.

## 0.6.0 — 2026-09-13

### Added
- **NVML provider** (`mtop.gpu.NvmlProvider`): NVIDIA telemetry through
  `libnvidia-ml.so.1` via ctypes — the library every driver ships and the
  one `nvidia-smi` itself calls. No process spawn per cycle, and
  `nvmlDeviceGetComputeRunningProcesses` gives host PID + bytes for every
  compute process on each card, which `nvidia-smi --query-gpu` cannot.
  `nvidia-smi` (host or in-container) stays as the fallback and is skipped
  when NVML answers, so a card is never listed twice. Unsupported readings
  (memory on GB10/Jetson, power on some parts) come out as `[N/A]`, which
  the unified-memory patch already understands.
- **Runner ↔ GPU join.** Runner PIDs from the host `/proc` walk are matched
  against NVML's per-card process list: the RUNNERS table gains a `GPU`
  column (card indices, e.g. `0,1` for a tensor-split model) and the GPU
  section lists what is on each card (`procs: qwen2.5-coder:32b (18.4G)`).
  `--json` carries `gpu` and `gpu_mem_mib` per runner and `procs` per GPU.
  Runners found through the in-container exec fallback have container-
  namespace PIDs and do not link; the column shows `—`.
- **SERVER CONFIG section**: the inference-relevant environment the server
  was started with — `OLLAMA_*`, `CUDA_VISIBLE_DEVICES`, `HIP_*`/`HSA_*`/
  `ROCR_*`, `GGML_*` — with the source named: `container` (Config.Env from
  inspect), `process` (`/proc/<pid>/environ`, same user or root), or
  `systemd` (`systemctl show -p Environment`, which covers the unit and its
  drop-ins but not `EnvironmentFile=` contents). Anything that looks like a
  key or token is masked. Toggle with `e`, hide with `--no-env`; `server` in
  `--json`.
- **Ollama version** from `/api/version` in the header.

### Verified
- RTX 5060 (local, systemd): NVML figures match `nvidia-smi`; environment
  falls back to systemd because the unit runs as another user.
- DGX Spark (GB10, docker): environment from the container, NVML with
  `[N/A]` memory patched from `/proc/meminfo`.

## 0.5.0 — 2026-09-13

### Added
- **Docker Engine API over a socket** (`mtop.container.DockerApi`, stdlib
  `http.client` on `AF_UNIX`). mtop no longer needs the `docker` binary: it
  talks to `/var/run/docker.sock`, a rootless `$XDG_RUNTIME_DIR/docker.sock`,
  or a Podman socket (`$XDG_RUNTIME_DIR/podman/podman.sock`,
  `/run/podman/podman.sock`), and honours `DOCKER_HOST` for `unix://` and
  plain `tcp://` endpoints the way the CLI does. Inspect, stats and exec
  (`ollama ps`, the in-container cmdline fallback, `nvidia-smi` inside the
  container) all go through it.
- **Podman support**, via the compat API on its socket or the `podman` CLI
  when no socket answers. The runtime in use is reported as `runtime` in
  `--json` (`docker-api`, `podman-api`, `docker-cli`, `podman-cli`) and in
  the footer.
- `--runtime auto|api|cli` to force one path. `auto` prefers a socket and
  falls back to whichever CLI is on PATH.
- Container stats use `stats?stream=false&one-shot=true` and compute CPU%
  from the previous sample, like the local `/proc` path. The `docker stats
  --no-stream` fork blocked ~2 s per cycle (it waits for two samples), which
  with a 1 s interval kept the collector one hiccup away from the STALE flag.
  Measured on a GB10 Spark: `--json` 0.74 s through the API (including the
  0.5 s second-sample wait) vs 1.20 s through the CLI, with matching memory
  figures. `--json` takes that second sample in docker-api mode as well.
- `inspect` now also returns the container's `Config.Env` and image;
  groundwork for showing the effective `OLLAMA_*` configuration.
- `workflow_dispatch` on the CI workflow, so a release can be re-run by hand.

### Changed
- `NvidiaSmiProvider` takes a list of `(label, runner)` attempts instead of
  argv prefixes, so the in-container probe works through whichever runtime is
  active rather than assuming a `docker` binary.
- `docker` mode keeps its name in `--json` (`mode: "docker"`) even when the
  runtime is Podman; the distinction lives in `runtime`.

### Verified
- DGX Spark (GB10, Ollama 0.33.2 in docker): API and CLI paths side by side,
  the raw `ollama ps` toggle through an API exec, and mtop itself running in a
  `python:3.12-slim` container with only the socket bind-mounted — no docker
  CLI, no host PID namespace — falling back to the in-container cmdline exec
  for runner discovery.

## 0.4.2 — 2026-09-13

### Fixed
- **`--json` in local mode always reported `cpu: "0.00%"`.** Process CPU% is a
  delta between two `/proc` samples and a one-shot run took one. It now takes
  a second sample 0.5 s later; docker mode is unaffected (`docker stats`
  samples internally) and does not pay the extra wait.
- **`keep_alive: -1` rendered as `106394d left`.** Ollama schedules the expiry
  ~292 years out and `ollama ps` prints `Forever`; mtop now says `forever`.
  Go's zero time (`0001-01-01`) renders as `never` instead of `739871d ago`.
- **Runner argv from the Ollama engine was only half parsed.** Current builds
  spell the KV cache dtype as one `--kv-cache-type` flag rather than
  `-ctk`/`-ctv`, so the KV column was empty on every new-engine runner.
  `--threads`, `--ollama-engine` and `--multiuser-cache` are recognised too;
  they show up as `thr:N`, `ollama-engine` and `multiuser` in the flags column.
  0.33 also replaced `--direct-io` with `--load-mode dio`; both render as
  `O_DIRECT`, other load modes as `load:<mode>`.
- **`o` (raw `ollama ps`) in local mode ignored `-u`.** The CLI reads
  `$OLLAMA_HOST`; the subprocess now gets the API URL mtop was pointed at, so
  a second instance on another port lists its own models.
- **`$http_proxy` hijacked the loopback API call.** urllib proxies every host,
  including 127.0.0.1, so on a box with a corporate proxy the API came back as
  a 502 from the proxy. Loopback URLs now bypass env proxies, as Go's
  `ProxyFromEnvironment` (and therefore Ollama's own client) does.
- **`rocm-smi` total could come back as the used figure.** The lookup matched
  key substrings, and `VRAM Total Used Memory (B)` contains both `vram` and
  `total`, so whichever key the JSON emitted first won. The total lookup now
  excludes `used`. Separately, the bytes-vs-MiB heuristic is decided per card
  from the total instead of per value: a used figure under 1 GiB in a
  bytes-reporting build was misread as MiB (536870912 B became "512 GiB").
- **Two `ollama serve` processes picked one at random.** The `/proc` scan
  returned the first directory entry, which is not stable. With several
  candidates mtop now prefers the one holding the listening socket on the API
  port (when `/proc/<pid>/fd` is readable), else the oldest — the same answer
  every cycle.
- **Monochrome terminals crashed on start** (`start_color()` raises on
  TERM=dumb / vt100). Colors are skipped when the terminal has none; the
  locale is set from the environment so box-drawing and block characters
  survive a `LANG=C` shell; `curs_set` failures are ignored.
- **Header fields overlapped under ~90 columns.** The status line flowed onto
  hardcoded columns 32 and 64; it is now a left-to-right flow with a fixed gap,
  clipped before the right border, and the timestamp is dropped when the
  fields already reach it.

### Verified
- Docker mode on a DGX Spark (GB10, Ollama 0.33.2 in `ollama/ollama:latest`,
  Ubuntu 24.04 aarch64, Python 3.12): unified-memory GPU section, runner
  discovery through the container init PID on the host `/proc`, blob-to-tag
  match, raw `ollama ps` toggle, and the header at 70 columns.

### Changed
- **PROCESSOR column matches `ollama ps` exactly**: `100% GPU`, `100% CPU`,
  `45%/55% CPU/GPU`, or `Unknown`, computed from `size_vram`/`size` the way
  `cmd/cmd.go` does. The old `GPU` / `CPU+GPU` / `CPU` heuristic and the dead
  `details.processor` lookup are gone.
- Local-mode CPU budget honours the systemd unit's `CPUQuota=` (via
  `CPUQuotaPerSecUSec`) and the process's CPU affinity
  (`sched_getaffinity`) instead of the raw host core count.
- `runners[]` in `--json` gained `engine` (`ollama` | `llama`), `threads` and
  `multiuser_cache`.

### Added
- **Test suite** (`tests/`, pytest, stdlib-only fakes): runner argv for three
  Ollama vintages, blob-to-tag matching, the AMD sysfs provider against a
  synthetic `/sys/class/drm` tree (dGPU, APU, PCI-slot ordering, hotplug
  re-probe), rocm-smi 5.x/6.x JSON, nvidia-smi CSV with `[N/A]` and the
  docker-exec fallback, the GPU registry's backoff and Tegra de-dup, the
  collector's local-mode rollup and auto-mode upgrade, the renderer through a
  fake curses window, and the bundler end to end.
- **CI** (GitHub Actions): ruff + pytest on Python 3.10–3.14, a bundle job
  that builds and smoke-runs `dist/mtop.py`, and a release job that attaches
  it to the GitHub release on a `v*` tag — the file the README's curl
  one-liner has been pointing at.
- `pip install -e ".[dev]"` pulls pytest and ruff.

## 0.4.1 — 2026-08-19

### Fixed
- **VRAM bar missing on GB10 / Jetson.** `nvidia-smi` reports memory as `[N/A]`
  on unified-memory parts and the `/proc/meminfo` fallback was gated on
  `/proc/device-tree/model` matching a keyword. Where that file is absent or
  names the board differently, the gate failed and the bar silently vanished —
  the exact symptom 0.2.0 set out to fix, reintroduced by a second condition.
  The gate is gone: a card that cannot report its own memory *is* a
  unified-memory part, so system RAM is the honest answer. The device-tree
  probe is kept for discovery when `nvidia-smi` is absent entirely.
- Table rules are sized to the widest visible line instead of the sum of column
  widths. The runner flags column has an empty header, which padded with
  invisible spaces while the rule got the full width in dashes — it overhung
  the content and left the two on-screen tables with mismatched rules.

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
