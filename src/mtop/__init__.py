"""
mtop — htop for Ollama: a curses TUI (and a headless JSON / Prometheus
exporter) for models, runners, GPUs and the server process or container.

Architecture: a background collector thread (mtop.collector) gathers all data
— container runtime (mtop.container), /proc (mtop.procfs), GPU providers
(mtop.gpu), the server log (mtop.logs), the Ollama API — and publishes
immutable snapshots; the curses loop (mtop.ui) only draws the latest snapshot
and handles keys at a fixed 100 ms poll. mtop.cli parses arguments and runs
the headless modes; mtop.export renders Prometheus text.

This package re-exports the public names so `import mtop; mtop.Collector`
keeps working, and so tests can monkeypatch through it (patch the module the
name is *used* in: mtop.collector.http_get_json, not mtop.http_get_json).

Usage:
    mtop [-c CONTAINER] [-i INTERVAL] [-u URL ...] [-H HEADER] [--insecure]
         [-m MODE] [--runtime RT] [--no-gpu] [--no-runners] [--json] [-h]
"""

from . import cli, collector, container, export, gpu, logs, procfs, runner, util
from ._version import __version__
from .cli import (API_KEY_ENV, DEFAULT_API_BASE, DEFAULT_CONTAINER, DEFAULT_INTERVAL,
                  JSON_SCHEMA_VERSION, headless_main, json_main, main, snapshot_healthy,
                  write_output)
from .collector import (DEFAULT_LOG_LINES, HISTORY_LEN, JSON_CPU_WINDOW, LOG_FETCH, SLOW_FLOOR,
                        STALE_FACTOR, Collector)
from .procfs import (find_ollama_pid, find_ollama_pids, host_cpu_count, listening_inodes,
                     parse_systemd_cpu_quota, parse_systemd_environment, parse_systemd_show,
                     pid_owns_socket, proc_children_map, proc_starttime_ticks, proc_uptime_sec,
                     process_tree, read_proc_cmdline, read_proc_cpu_ticks, read_proc_environ,
                     read_proc_ppid, read_proc_pss_bytes, read_proc_rss_bytes, read_unified_memory,
                     systemd_environment, systemd_ollama, total_ram_bytes)
from .runner import (ENV_PREFIXES, RUNNER_BASENAMES, inference_env, link_runners_to_gpus,
                     match_runners_to_models, parse_runner_argv, processor_label)
from .util import (CLK_TCK, FOREVER_AFTER_SEC, IS_DARWIN, IS_LINUX, Endpoint, api_port,
                   bytes_to_gib, fmt_duration, http_get_json, is_loopback_url, make_ssl_context,
                   normalize_api_url, parse_endpoint_arg, parse_header_arg, relative_time,
                   run_cmd, split_userinfo, to_float)

__all__ = [
    "__version__", "main", "Collector", "Endpoint",
    "cli", "collector", "container", "export", "gpu", "logs", "procfs", "runner", "util",
    "API_KEY_ENV", "DEFAULT_API_BASE", "DEFAULT_CONTAINER", "DEFAULT_INTERVAL",
    "JSON_SCHEMA_VERSION", "headless_main", "json_main", "snapshot_healthy", "write_output",
    "DEFAULT_LOG_LINES", "HISTORY_LEN", "JSON_CPU_WINDOW", "LOG_FETCH", "SLOW_FLOOR",
    "STALE_FACTOR",
    "find_ollama_pid", "find_ollama_pids", "host_cpu_count", "listening_inodes",
    "parse_systemd_cpu_quota", "parse_systemd_environment", "parse_systemd_show",
    "pid_owns_socket", "proc_children_map", "proc_starttime_ticks", "proc_uptime_sec",
    "process_tree", "read_proc_cmdline", "read_proc_cpu_ticks", "read_proc_environ",
    "read_proc_ppid", "read_proc_pss_bytes", "read_proc_rss_bytes", "read_unified_memory",
    "systemd_environment", "systemd_ollama", "total_ram_bytes",
    "ENV_PREFIXES", "RUNNER_BASENAMES", "inference_env", "link_runners_to_gpus",
    "match_runners_to_models", "parse_runner_argv", "processor_label",
    "CLK_TCK", "FOREVER_AFTER_SEC", "IS_DARWIN", "IS_LINUX", "api_port", "bytes_to_gib",
    "fmt_duration", "http_get_json", "is_loopback_url", "make_ssl_context",
    "normalize_api_url", "parse_endpoint_arg", "parse_header_arg", "relative_time", "run_cmd",
    "split_userinfo", "to_float",
]


def __getattr__(name: str):
    """Curses-only names (render_*, draw_*, curses_main) load on demand so the
    package imports without curses — Windows can still run --json.

    importlib, not `from . import ui`: the latter probes `hasattr(mtop, "ui")`
    before the attribute exists and would re-enter this hook forever.
    """
    import importlib
    ui = importlib.import_module("mtop.ui")
    if name == "ui":
        return ui
    try:
        return getattr(ui, name)
    except AttributeError:
        raise AttributeError(f"module 'mtop' has no attribute {name!r}") from None
