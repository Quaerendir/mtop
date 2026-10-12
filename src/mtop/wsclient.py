"""
mtop.wsclient — just enough of a WebSocket client (RFC 6455) to follow
ComfyUI's `/ws`, on the standard library only.

Receive side: text frames are returned, binary frames (ComfyUI's preview
images) are read and dropped without being kept, fragments are joined, pings
are answered. Send side: only control frames (pong, ping, close), masked as a
client must. No extensions and no proxy: the socket goes straight to the host.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
import urllib.parse

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_TEXT = 8 * 1024 * 1024      # a text message bigger than this is a protocol error here
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(Exception):
    pass


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def ws_url(http_url: str, path: str) -> str:
    """http://host:8188 + /ws?x -> ws://host:8188/ws?x (https -> wss)."""
    parts = urllib.parse.urlsplit(http_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return f"{scheme}://{parts.netloc}{parts.path.rstrip('/')}{path}"


def encode_frame(opcode: int, payload: bytes = b"", mask: bytes | None = None) -> bytes:
    """One final client frame; `mask` (4 bytes) is random unless given (tests)."""
    mask = mask if mask is not None else os.urandom(4)
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 1 << 16:
        head += bytes([0x80 | 126]) + struct.pack("!H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack("!Q", n)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


class WebSocket:
    """A connected client socket. `recv()` blocks for the next text message
    and raises `TimeoutError` when nothing arrives within the timeout."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = b""
        # True while recv() is inside a frame: a timeout then leaves the
        # stream mid-frame, and the only safe way on is a new connection.
        self.mid_frame = False
        self.frames = 0                     # frames read so far, pongs included

    @classmethod
    def connect(cls, url: str, timeout: float, headers: dict[str, str] | None = None,
                context: ssl.SSLContext | None = None) -> WebSocket:
        parts = urllib.parse.urlsplit(url)
        secure = parts.scheme == "wss"
        host = parts.hostname or "localhost"
        port = parts.port or (443 if secure else 80)
        raw = socket.create_connection((host, port), timeout=timeout)
        try:
            sock: socket.socket = raw
            if secure:
                ctx = context or ssl.create_default_context()
                sock = ctx.wrap_socket(raw, server_hostname=host)
            ws = cls(sock)
            ws._handshake(parts, headers or {})
            return ws
        except BaseException:
            raw.close()
            raise

    def _handshake(self, parts: urllib.parse.SplitResult, headers: dict[str, str]) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        lines = [f"GET {path} HTTP/1.1", f"Host: {parts.netloc}", "Upgrade: websocket",
                 "Connection: Upgrade", f"Sec-WebSocket-Key: {key}",
                 "Sec-WebSocket-Version: 13"]
        lines += [f"{k}: {v}" for k, v in headers.items()]
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        while b"\r\n\r\n" not in self._buf:
            if len(self._buf) > 65536:
                raise WebSocketError("handshake reply too long")
            self._fill()
        head, self._buf = self._buf.split(b"\r\n\r\n", 1)
        status, *fields = head.decode("latin-1").split("\r\n")
        if " 101 " not in status + " ":
            raise WebSocketError(status.strip() or "no handshake reply")
        got = {k.strip().lower(): v.strip() for k, _, v in (f.partition(":") for f in fields)}
        if got.get("sec-websocket-accept") != accept_key(key):
            raise WebSocketError("bad Sec-WebSocket-Accept")

    def _fill(self) -> None:
        chunk = self.sock.recv(65536)
        if not chunk:
            raise WebSocketError("connection closed")
        self._buf += chunk

    def _read(self, n: int, keep: bool = True) -> bytes:
        """n bytes from the stream; with keep=False they are dropped as they
        arrive, so a large preview image never sits in memory whole."""
        out = []
        while n:
            if not self._buf:
                self._fill()
            take = self._buf[:n]
            self._buf = self._buf[len(take):]
            n -= len(take)
            if keep:
                out.append(take)
        return b"".join(out)

    def _frame_head(self) -> tuple[bool, int, int, bytes | None]:
        b0, b1 = self._read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack("!H", self._read(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", self._read(8))[0]
        mask = self._read(4) if b1 & 0x80 else None
        return bool(b0 & 0x80), b0 & 0x0F, n, mask

    def recv(self) -> str:
        parts: list[bytes] = []
        size = 0
        text = None                         # opcode of the message being joined
        while True:
            if not self._buf:
                self._fill()                # an idle timeout lands here, between frames
            self.mid_frame = True
            fin, op, n, mask = self._frame_head()
            self.frames += 1
            if op >= OP_CLOSE:              # control frames: never fragmented
                payload = self._read(n)
                if op == OP_PING:
                    self.send(OP_PONG, payload)
                elif op == OP_CLOSE:
                    raise WebSocketError("closed by server")
                self.mid_frame = text is not None    # between fragments
                continue
            if op != OP_CONT:
                text = op
            keep = text == OP_TEXT
            if keep:
                size += n
                if size > MAX_TEXT:
                    raise WebSocketError("text message too large")
            payload = self._read(n, keep)
            if mask and keep:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if keep:
                parts.append(payload)
            if fin:
                self.mid_frame = False
                if text == OP_TEXT:
                    return b"".join(parts).decode("utf-8", errors="replace")
                parts, size, text = [], 0, None

    def send(self, opcode: int, payload: bytes = b"") -> None:
        self.sock.sendall(encode_frame(opcode, payload))

    def close(self) -> None:
        try:
            self.send(OP_CLOSE)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def shutdown(self) -> None:
        """Wake a thread blocked in recv() (close() from another thread waits)."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
