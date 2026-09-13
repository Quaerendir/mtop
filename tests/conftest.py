"""Shared fixtures: import path, a fake curses window, common runner argv."""

import curses
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeWin:
    """Just enough of a curses window to render into and read back."""

    def __init__(self, rows: int = 40, cols: int = 120):
        self.rows, self.cols = rows, cols
        self.grid = [[" "] * cols for _ in range(rows)]
        self.calls: list[tuple[int, int, str, int]] = []

    def getmaxyx(self):
        return self.rows, self.cols

    def addstr(self, y, x, text, attr=0):
        if y >= self.rows or x >= self.cols:
            raise curses.error("out of bounds")
        self.calls.append((y, x, text, attr))
        for i, ch in enumerate(text):
            if x + i < self.cols:
                self.grid[y][x + i] = ch

    def line(self, y: int) -> str:
        return "".join(self.grid[y]).rstrip()

    def text(self) -> str:
        return "\n".join(self.line(y) for y in range(self.rows)).rstrip()


@pytest.fixture
def win():
    return FakeWin()


@pytest.fixture(autouse=True)
def no_curses_colors(monkeypatch):
    """color_pair() needs initscr(); the renderer only ORs the result."""
    monkeypatch.setattr(curses, "color_pair", lambda n: 0)


# Real-world runner command lines, by Ollama vintage.
LLAMA_SERVER_ARGV = [
    "/usr/lib/ollama/runners/cuda_v12/ollama_llama_server", "--model",
    "/root/.ollama/models/blobs/sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa",
    "--ctx-size", "8192", "--batch-size", "512", "--n-gpu-layers", "33",
    "--threads", "8", "--flash-attn", "--parallel", "1", "--port", "40561",
]
LLAMA_SHORT_ARGV = [
    "llama-server", "-m", "/blobs/sha256-deadbeef", "-c", "4096", "-b", "256",
    "-ub", "64", "-ngl", "99", "-ctk", "q8_0", "-ctv", "q4_0", "-fa", "on",
    "-np", "2", "-ts", "1,1", "-mg", "0", "--no-mmap",
]
# Ollama 0.33.2 on a GB10 Spark (docker), verbatim minus the digests.
LLAMA_033_ARGV = [
    "/usr/lib/ollama/llama-server", "--model", "/root/.ollama/models/blobs/sha256-819dce06",
    "--port", "37893", "--host", "127.0.0.1", "--no-webui", "--offline", "-c", "65536",
    "-np", "1", "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
    "--no-jinja", "--chat-template", "chatml",
    "--mmproj", "/root/.ollama/models/blobs/sha256-819dce06",
    "--load-mode", "dio", "--cache-type-k", "f16", "--cache-type-v", "f16",
    "--flash-attn", "on", "-b", "2048", "-ub", "2048", "--context-shift", "--keep", "4",
]
OLLAMA_ENGINE_ARGV = [
    "/usr/local/bin/ollama", "runner", "--ollama-engine", "--model",
    "/usr/share/ollama/.ollama/models/blobs/sha256-cafebabe", "--ctx-size", "32768",
    "--batch-size", "512", "--n-gpu-layers", "65", "--threads", "12",
    "--flash-attn", "--kv-cache-type", "q8_0", "--parallel", "1", "--port", "34567",
    "--multiuser-cache",
]
