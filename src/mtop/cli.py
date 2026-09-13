"""
mtop.cli — argument parsing, the headless modes (--json / --prometheus /
--watch) and the entry point.
"""

from __future__ import annotations

import argparse
import json
import locale
import os
import sys
import time

from ._version import __version__
from .collector import DEFAULT_LOG_LINES, JSON_CPU_WINDOW, Collector
from .export import prometheus_text
from .util import Endpoint, parse_header_arg

JSON_SCHEMA_VERSION = 1   # bump when a --json field changes meaning or is removed

DEFAULT_CONTAINER = "ollama"
DEFAULT_INTERVAL = 1.0
DEFAULT_API_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
API_KEY_ENV = "OLLAMA_API_KEY"    # same variable the ollama CLI uses for Bearer auth


def snapshot_healthy(snap: dict) -> bool:
    return snap.get("status") in ("running", "api-only") and bool(snap.get("models_ok", False))


def write_output(text: str, path: str | None, append: bool = False) -> None:
    """stdout, or a file: appended (NDJSON) or replaced atomically (tmp + rename).

    Atomic replacement is what the node_exporter textfile collector wants —
    it must never read a half-written .prom file.
    """
    if not path:
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    if append:
        with open(path, "a") as f:
            f.write(text)
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def headless_main(args) -> int:
    """--json / --prometheus: one snapshot, or a stream with --watch.

    --json           one indented JSON document (the pre-0.8 --json output)
    --json --watch   one compact JSON object per line (NDJSON), every interval
    --prometheus     text exposition format for a textfile collector or a pipe
    --prometheus --watch   re-rendered every interval; with -o the file is
                     replaced atomically each time, so `mtop --prometheus
                     --watch -o /var/lib/node_exporter/textfile/mtop.prom`
                     is a complete exporter setup with no cron and no port.

    Exit code (one-shot): 0 when the source is up and the API answered, 1
    otherwise. --watch runs until Ctrl-C and exits 0.
    """
    collector = Collector(
        container=args.container,
        api_url=args.api_url,
        interval=args.interval,
        show_gpu=not args.no_gpu,
        mode=args.mode,
        show_runners=not args.no_runners,
        runtime=args.runtime,
        show_env=not args.no_env,
        endpoints=args.endpoints,
        show_logs=args.logs,
        log_lines=args.log_lines,
    )
    fmt = "prometheus" if args.prometheus else "json"
    watch = bool(args.watch)

    def render(snap: dict) -> str:
        snap = {"schema_version": JSON_SCHEMA_VERSION, **snap}
        snap.pop("ts", None)  # monotonic value is meaningless outside the process
        snap.pop("history", None)  # screen-only ring buffer
        if fmt == "prometheus":
            return prometheus_text(snap, __version__)
        if watch:
            return json.dumps(snap, separators=(",", ":")) + "\n"
        return json.dumps(snap, indent=2) + "\n"

    try:
        snap = collector.collect(time.monotonic())
        if collector.needs_second_sample(snap):
            # CPU% is a delta between two samples (/proc ticks, or one-shot
            # Engine API stats), so a single pass can only ever say 0.00%.
            # Take a second one. The docker CLI path samples internally and
            # would pay another ~2 s for nothing, so it is excluded.
            time.sleep(JSON_CPU_WINDOW)
            collector.force_slow()
            snap = collector.collect(time.monotonic())
        write_output(render(snap), args.output, append=(watch and fmt == "json"))
        if not watch:
            return 0 if snapshot_healthy(snap) else 1
        while True:
            time.sleep(args.interval)
            snap = collector.collect(time.monotonic())
            write_output(render(snap), args.output, append=(fmt == "json"))
    except KeyboardInterrupt:
        return 0 if watch else 130
    except BrokenPipeError:
        # `mtop --json --watch | head -3`: the reader is gone. Detach stdout
        # so the interpreter's exit-time flush does not print a traceback.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0
    finally:
        collector.close()


def json_main(args) -> int:
    """Kept for callers of the pre-0.8 name."""
    for attr, default in (("prometheus", False), ("watch", False), ("output", None),
                          ("logs", False), ("log_lines", DEFAULT_LOG_LINES)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    return headless_main(args)


def main():
    parser = argparse.ArgumentParser(
        description="mtop — htop for Ollama: models, runners, GPUs and the server "
                    "(container, systemd or manual), as a TUI or a JSON/Prometheus exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q=quit, +=faster, -=slower, o=toggle raw ollama ps, "
               "r=toggle runners, e=toggle server config, l=toggle logs\n\n"
               "https://github.com/Quaerendir/mtop",
    )
    parser.add_argument("-c", "--container", default=DEFAULT_CONTAINER,
                        help=f"Docker container name (default: {DEFAULT_CONTAINER})")
    parser.add_argument("-i", "--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("-u", "--api-url", action="append", dest="api_urls", metavar="URL",
                        help="Ollama API base URL (default: $OLLAMA_HOST or "
                             f"{DEFAULT_API_BASE}). Repeat for several instances; the "
                             "first is the primary (host stats, runners), the rest are "
                             "API-only. Optional label: -u rig=http://gpu-rig:11434. "
                             "Credentials in the URL become basic auth.")
    parser.add_argument("-H", "--header", action="append", default=[], metavar="'Name: value'",
                        help="Extra HTTP header for every API request (repeatable), e.g. "
                             "'Authorization: Bearer ...' for an instance behind a proxy")
    parser.add_argument("--insecure", action="store_true",
                        help="Skip TLS certificate verification for https:// endpoints")
    parser.add_argument("--cacert", metavar="FILE",
                        help="CA bundle (PEM) to verify https:// endpoints against")
    parser.add_argument("-m", "--mode", choices=["auto", "docker", "local", "api"],
                        default="auto",
                        help="Data source (default: auto — probe docker, then a "
                             "bare-metal ollama process, else api). "
                             "local: monitor a systemd/manual `ollama serve`. "
                             "api: models only, no host resource stats.")
    parser.add_argument("--runtime", choices=["auto", "api", "cli"], default="auto",
                        help="How to reach the container runtime (default: auto — "
                             "Engine API on $DOCKER_HOST or a docker/podman socket, "
                             "else the docker/podman CLI). api: socket only. "
                             "cli: subprocesses only.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU stats section")
    parser.add_argument("--no-runners", action="store_true",
                        help="Hide the RUNNERS section (effective inference "
                             "config parsed from each runner process argv)")
    parser.add_argument("--no-env", action="store_true",
                        help="Hide the SERVER CONFIG section (OLLAMA_* environment "
                             "the server was started with)")
    parser.add_argument("--logs", action="store_true",
                        help="Show the LOGS section from the start (container logs or "
                             "journalctl), with request counts parsed from the GIN lines. "
                             "Toggle at runtime with 'l'. Also adds `logs` to --json")
    parser.add_argument("--log-lines", type=int, default=DEFAULT_LOG_LINES, metavar="N",
                        help=f"Log lines to show (default: {DEFAULT_LOG_LINES})")
    parser.add_argument("--no-docker", action="store_true",
                        help="Alias for --mode api (kept for compatibility)")
    parser.add_argument("--json", action="store_true",
                        help="One-shot: print a single snapshot as JSON and exit "
                             "(exit code 1 on unhealthy)")
    parser.add_argument("--prometheus", action="store_true",
                        help="One-shot: print the snapshot in Prometheus text exposition "
                             "format and exit (exit code 1 on unhealthy)")
    parser.add_argument("--watch", action="store_true",
                        help="With --json/--prometheus: keep emitting every INTERVAL "
                             "seconds until Ctrl-C. --json --watch prints NDJSON (one "
                             "compact object per line)")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Write to FILE instead of stdout. Prometheus output "
                             "replaces the file atomically (tmp + rename) — point it at "
                             "the node_exporter textfile directory; NDJSON is appended")
    parser.add_argument("-V", "--version", action="version",
                        version=f"mtop {__version__}")
    args = parser.parse_args()
    headers: dict[str, str] = {}
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for h in args.header:
        try:
            name, val = parse_header_arg(h)
        except ValueError as e:
            parser.error(str(e))
        headers[name] = val
    if args.cacert and not os.path.exists(args.cacert):
        parser.error(f"--cacert: no such file: {args.cacert}")
    try:
        args.endpoints = [Endpoint(u, headers, args.insecure, args.cacert)
                          for u in (args.api_urls or [DEFAULT_API_BASE])]
    except ValueError as e:
        parser.error(str(e))
    args.api_url = args.endpoints[0].url
    # --no-docker is the v0.2.0 spelling of "api only"; let it win only when the
    # user didn't pass an explicit --mode, so `--mode local --no-docker` errors
    # toward the explicit choice rather than silently overriding it.
    if args.no_docker and args.mode == "auto":
        args.mode = "api"

    if args.watch and not (args.json or args.prometheus):
        parser.error("--watch needs --json or --prometheus")
    if args.json and args.prometheus:
        parser.error("--json and --prometheus are mutually exclusive")
    if args.json or args.prometheus:
        sys.exit(headless_main(args))

    # curses encodes with the C library's locale; without this, a LANG=C shell
    # (minimal containers, some SSH jump hosts) turns every █ ░ ─ ║ into '?'.
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass
    import curses                        # only the TUI needs it (no curses on Windows)

    from .ui import curses_main
    try:
        curses.wrapper(curses_main, args)
    except KeyboardInterrupt:
        pass
