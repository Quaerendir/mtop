"""
mtop.util — small shared helpers: subprocess and HTTP wrappers, the API
Endpoint (auth, TLS, labels), number/time formatting, platform flags.

No imports from the rest of the package except mtop.export (pure parsers).
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .export import parse_iso

FOREVER_AFTER_SEC = 10 * 365 * 86400   # expires_at this far out == keep_alive -1
IS_LINUX = sys.platform.startswith("linux")
IS_DARWIN = sys.platform == "darwin"
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def normalize_api_url(url: str) -> str:
    """Accept Ollama-style OLLAMA_HOST values without a scheme.

    Ollama itself treats ``OLLAMA_HOST=0.0.0.0:11434`` or ``gpu-rig:11434``
    as valid; urllib does not. Prepend http:// when no scheme is present.
    """
    url = url.strip().rstrip("/")
    if url and "://" not in url:
        url = "http://" + url
    return url


def run_cmd(cmd: list[str], timeout: int = 5,
            env: dict[str, str] | None = None,
            merge_stderr: bool = False) -> tuple[bool, str]:
    """Run a command, return (success, stdout_or_stderr).

    merge_stderr: interleave stderr into the returned text (docker logs
    replays the container's stderr — where Ollama logs — on our stderr).
    """
    try:
        r = subprocess.run(cmd, text=True, timeout=timeout, env=env,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE)
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, (r.stderr or "").strip() or r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, f"command not found: {cmd[0]}"
    except Exception as e:
        return False, str(e)


# urllib honors $http_proxy for *every* host, including 127.0.0.1, so on a
# box with a corporate proxy the local Ollama call comes back as a 502 from the
# proxy. Go's ProxyFromEnvironment — and therefore Ollama's own client — skips
# loopback. Mirror that: env proxies for remote hosts, none for loopback.
_PROXY_OPENER = urllib.request.build_opener()
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}


def is_loopback_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def api_port(url: str) -> int | None:
    """Port of the API URL, defaulting to Ollama's 11434 when unspecified."""
    try:
        parts = urllib.parse.urlsplit(url)
        return parts.port or (443 if parts.scheme == "https" else 11434)
    except ValueError:
        return None


def http_get_json(url: str, timeout: int = 5, headers: dict[str, str] | None = None,
                  context: ssl.SSLContext | None = None) -> tuple[bool, Any]:
    """GET JSON from URL, return (success, data_or_error_string).

    `headers` carry auth (Bearer / Basic / anything a reverse proxy wants);
    `context` a custom TLS setup (--insecure, --cacert). HTTP errors surface
    as "HTTP 401 Unauthorized"-style strings so the screen says *why*.
    """
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   **(headers or {})})
        if context is not None:
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=context),
                *([urllib.request.ProxyHandler({})] if is_loopback_url(url) else []))
        else:
            opener = _DIRECT_OPENER if is_loopback_url(url) else _PROXY_OPENER
        with opener.open(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def make_ssl_context(insecure: bool = False,
                     cacert: str | None = None) -> ssl.SSLContext | None:
    """TLS context for --insecure / --cacert; None means urllib's default."""
    if not insecure and not cacert:
        return None
    ctx = ssl.create_default_context(cafile=cacert) if cacert else ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def split_userinfo(url: str) -> tuple[str, dict[str, str]]:
    """`https://user:pw@host` -> (`https://host`, {"Authorization": "Basic ..."}).

    urllib does not turn URL credentials into a header on its own, and a
    reverse proxy in front of Ollama is most often protected with basic auth.
    """
    parts = urllib.parse.urlsplit(url)
    if not parts.username and not parts.password:
        return url, {}
    user = urllib.parse.unquote(parts.username or "")
    pw = urllib.parse.unquote(parts.password or "")
    cred = f"{user}:{pw}"
    host = parts.hostname or ""
    if ":" in host:                      # IPv6 literal
        host = f"[{host}]"
    netloc = host + (f":{parts.port}" if parts.port else "")
    clean = urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    token = base64.b64encode(cred.encode()).decode()
    return clean, {"Authorization": f"Basic {token}"}


def parse_header_arg(value: str) -> tuple[str, str]:
    """`Name: value` -> ("Name", "value"); raises ValueError otherwise."""
    name, sep, val = value.partition(":")
    if not sep or not name.strip():
        raise ValueError(f"expected 'Name: value', got {value!r}")
    return name.strip(), val.strip()
_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_endpoint_arg(value: str) -> tuple[str | None, str]:
    """`label=URL` or bare `URL` -> (label_or_None, url).

    A label is a plain word before the first '='; anything with a '/' or ':'
    before the '=' is part of a URL (query strings and the like).
    """
    head, sep, rest = value.partition("=")
    if sep and _LABEL_RE.match(head) and rest:
        return head, rest
    return None, value


class Endpoint:
    """One Ollama API base URL plus how to talk to it (auth, TLS, label)."""

    def __init__(self, spec: str, headers: dict[str, str] | None = None,
                 insecure: bool = False, cacert: str | None = None):
        label, url = parse_endpoint_arg(spec)
        url = normalize_api_url(url)
        url, basic = split_userinfo(url)
        self.url = url
        self.headers = {**(headers or {}), **basic}
        self.context = make_ssl_context(insecure, cacert)
        self.label = label or (urllib.parse.urlsplit(url).netloc or url)

    def get_json(self, path: str, timeout: int = 5) -> tuple[bool, Any]:
        return http_get_json(self.url + path, timeout, self.headers or None, self.context)

    def describe(self) -> dict:
        return {"label": self.label, "url": self.url,
                "auth": "Authorization" in self.headers,
                "tls": ("insecure" if self.context is not None
                        and self.context.verify_mode == ssl.CERT_NONE
                        else "custom-ca" if self.context is not None else "default")}


def bytes_to_gib(b: int | float) -> str:
    return f"{b / (1024**3):.2f}"


def to_float(s: Any) -> float | None:
    """Parse a numeric field that may be '[N/A]', 'N/A', '' or garbage."""
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


def relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to relative future/past string."""
    if not iso_str:
        return "—"
    target = parse_iso(iso_str)
    if target is None:
        return iso_str[:19]
    try:
        if target.year <= 1:
            return "never"          # Go zero time: no expiry scheduled
        now = datetime.now(timezone.utc)
        delta = target - now
        total_sec = int(delta.total_seconds())
        if total_sec >= FOREVER_AFTER_SEC:
            # keep_alive -1: Ollama schedules expiry ~292 years out and
            # `ollama ps` prints "Forever". "106394d left" is not helpful.
            return "forever"
        suffix = " left" if total_sec >= 0 else " ago"
        total_sec = abs(total_sec)
        if total_sec < 60:
            return f"{total_sec}s{suffix}"
        elif total_sec < 3600:
            m, s = divmod(total_sec, 60)
            return f"{m}m {s}s{suffix}"
        elif total_sec < 86400:
            h, rem = divmod(total_sec, 3600)
            m = rem // 60
            return f"{h}h {m}m{suffix}"
        else:
            d = total_sec // 86400
            return f"{d}d{suffix}"
    except Exception:
        return iso_str[:19]


def fmt_duration(sec: float) -> str:
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60}m {sec % 60}s"
    if sec < 86400:
        return f"{sec // 3600}h {(sec % 3600) // 60}m"
    return f"{sec // 86400}d {(sec % 86400) // 3600}h"
