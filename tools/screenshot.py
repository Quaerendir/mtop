#!/usr/bin/env python3
"""Render a colored tmux capture of mtop into docs/screenshot.png.

The README screenshot is a real session, not a mock-up. To refresh it:

    tmux new-session -d -s shot -x 136 -y 38 "mtop --logs --log-lines 3"
    sleep 14; tmux capture-pane -e -p -t shot > shot.ansi; tmux send-keys -t shot q
    python tools/screenshot.py shot.ansi docs/screenshot.png

Needs google-chrome (headless) and Pillow; stdlib otherwise. `-e` keeps the
SGR color codes, which this script turns into spans on a dark palette.
"""
import html
import re
import sys

PALETTE = ["#45475a", "#f38ba8", "#a6e3a1", "#f9e2af", "#89b4fa", "#f5c2e7", "#94e2d5", "#bac2de",
           "#585b70", "#f38ba8", "#a6e3a1", "#f9e2af", "#89b4fa", "#f5c2e7", "#94e2d5", "#a6adc8"]
BG, FG = "#1e1e2e", "#cdd6f4"
SGR = re.compile(r"\x1b\[([0-9;]*)m")

def render(text: str) -> str:
    out = []
    st = {"fg": None, "bg": None, "bold": False, "dim": False, "rev": False}
    def open_span():
        fg = PALETTE[st["fg"]] if st["fg"] is not None else FG
        bg = PALETTE[st["bg"]] if st["bg"] is not None else BG
        if st["rev"]:
            fg, bg = bg, fg
        style = f"color:{fg};background:{bg};"
        if st["bold"]:
            style += "font-weight:bold;"
        if st["dim"]:
            style += "opacity:.7;"
        return f'<span style="{style}">'
    for line in text.splitlines():
        pos = 0
        out.append(open_span())
        for m in SGR.finditer(line):
            out.append(html.escape(line[pos:m.start()]))
            pos = m.end()
            codes = [int(c) if c else 0 for c in m.group(1).split(";")]
            i = 0
            while i < len(codes):
                c = codes[i]
                if c == 0:
                    st.update(fg=None, bg=None, bold=False, dim=False, rev=False)
                elif c == 1:
                    st["bold"] = True
                elif c == 2:
                    st["dim"] = True
                elif c == 7:
                    st["rev"] = True
                elif c == 22:
                    st["bold"] = st["dim"] = False
                elif c == 27:
                    st["rev"] = False
                elif 30 <= c <= 37:
                    st["fg"] = c - 30
                elif 90 <= c <= 97:
                    st["fg"] = c - 90 + 8
                elif c == 39:
                    st["fg"] = None
                elif 40 <= c <= 47:
                    st["bg"] = c - 40
                elif 100 <= c <= 107:
                    st["bg"] = c - 100 + 8
                elif c == 49:
                    st["bg"] = None
                elif c in (38, 48) and i + 2 < len(codes) and codes[i + 1] == 5:
                    n = codes[i + 2]
                    st["fg" if c == 38 else "bg"] = n if n < 16 else None
                    i += 2
                i += 1
            out.append("</span>" + open_span())
        out.append(html.escape(line[pos:]) + "</span>\n")
    body = "".join(out)
    return f"""<!doctype html><meta charset="utf-8"><style>
html,body{{margin:0;background:{BG}}}
pre{{margin:0;padding:18px 22px;font:15px/1.28 "Hack","DejaVu Sans Mono",monospace;color:{FG};
background:{BG};white-space:pre;display:inline-block}}
</style><pre>{body}</pre>"""

def trim(text: str) -> str:
    """Drop trailing blank rows and collapse the gap before the footer line."""
    strip = lambda ln: re.sub(r"\x1b\[[0-9;]*m", "", ln).strip()  # noqa: E731
    rows = text.split("\n")
    while rows and not strip(rows[-1]):
        rows.pop()
    if not rows:
        return ""
    footer = rows.pop()
    while rows and not strip(rows[-1]):
        rows.pop()
    return "\n".join([*rows, "", footer]) + "\n"


def main() -> int:
    import os
    import shutil
    import subprocess
    import tempfile
    if len(sys.argv) != 3:
        print("usage: screenshot.py CAPTURE.ansi OUT.png", file=sys.stderr)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser")
                   if shutil.which(c)), None)
    if not chrome:
        print("no headless chrome found", file=sys.stderr)
        return 1
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        html_path = os.path.join(td, "shot.html")
        png_path = os.path.join(td, "shot.png")
        with open(src, encoding="utf-8", errors="replace") as f:
            capture = f.read()
        with open(html_path, "w") as f:
            f.write(render(trim(capture)))
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1300,800",
                        f"--screenshot={png_path}", f"file://{html_path}"],
                       check=True, capture_output=True)
        im = Image.open(png_path)
        bbox = Image.eval(im.convert("L"), lambda p: 255 if p > 40 else 0).getbbox()
        pad = 28
        box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
               min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        im.crop(box).save(out, optimize=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
