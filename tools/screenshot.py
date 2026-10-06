#!/usr/bin/env python3
"""Render a colored tmux capture of mtop into docs/screenshot.png.

The README screenshot is a real session, not a mock-up. To refresh it:

    tmux new-session -d -s shot -x 136 -y 38 "mtop --logs --log-lines 3"
    sleep 14; tmux capture-pane -e -p -t shot > shot.ansi; tmux send-keys -t shot q
    python tools/screenshot.py shot.ansi docs/screenshot.png

Needs Pillow, plus google-chrome (headless) for the best text rendering;
without Chrome the capture is drawn cell by cell with Pillow and DejaVu Sans
Mono. `-e` keeps the SGR color codes, which this script turns into spans on a
dark palette.
"""
import html
import re
import sys

PALETTE = ["#45475a", "#f38ba8", "#a6e3a1", "#f9e2af", "#89b4fa", "#f5c2e7", "#94e2d5", "#bac2de",
           "#585b70", "#f38ba8", "#a6e3a1", "#f9e2af", "#89b4fa", "#f5c2e7", "#94e2d5", "#a6adc8"]
BG, FG = "#1e1e2e", "#cdd6f4"
SGR = re.compile(r"\x1b\[([0-9;]*)m")

def runs(text: str):
    """Per line, a list of (text, fg, bg, bold, dim) runs with colors resolved."""
    st = {"fg": None, "bg": None, "bold": False, "dim": False, "rev": False}
    def style():
        fg = PALETTE[st["fg"]] if st["fg"] is not None else FG
        bg = PALETTE[st["bg"]] if st["bg"] is not None else BG
        if st["rev"]:
            fg, bg = bg, fg
        return fg, bg, st["bold"], st["dim"]
    lines = []
    for line in text.splitlines():
        pos = 0
        cur = []
        for m in SGR.finditer(line):
            if m.start() > pos:
                cur.append((line[pos:m.start()], *style()))
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
        if pos < len(line):
            cur.append((line[pos:], *style()))
        lines.append(cur)
    return lines


def render(text: str) -> str:
    out = []
    for line in runs(text):
        for chunk, fg, bg, bold, dim in line:
            css = f"color:{fg};background:{bg};"
            if bold:
                css += "font-weight:bold;"
            if dim:
                css += "opacity:.7;"
            out.append(f'<span style="{css}">{html.escape(chunk)}</span>')
        out.append("\n")
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


FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def draw_pillow(text: str, out: str, scale: int = 2) -> None:
    """Chrome-less fallback: one fixed-size cell per character."""
    import os

    from PIL import Image, ImageDraw, ImageFont
    size = 15 * scale
    reg = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSansMono.ttf"), size)
    bold = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf"), size)
    cw = round(reg.getlength("M"))
    ch = round(size * 1.28)
    lines = runs(text)
    cols = max((sum(len(c[0]) for c in ln) for ln in lines), default=0)
    padx, pady = 22 * scale, 18 * scale
    im = Image.new("RGB", (cols * cw + 2 * padx, len(lines) * ch + 2 * pady), BG)
    d = ImageDraw.Draw(im)

    def mix(fg: str, bg: str, a: float) -> str:
        f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x * a + y * (1 - a)):02x}" for x, y in zip(f, b, strict=True))

    for row, line in enumerate(lines):
        x, y = padx, pady + row * ch
        for chunk, fg, bg, is_bold, dim in line:
            w = len(chunk) * cw
            if bg != BG:
                d.rectangle([x, y, x + w - 1, y + ch - 1], fill=bg)
            color = mix(fg, bg, 0.7) if dim else fg
            font = bold if is_bold else reg
            for i, c in enumerate(chunk):
                cx = x + i * cw
                if c in "█░▁▂▃▄▅▆▇":
                    # Block elements fill the whole cell, as in a terminal.
                    if c == "░":
                        d.rectangle([cx, y, cx + cw - 1, y + ch - 1], fill=mix(fg, bg, 0.25))
                    else:
                        frac = 1.0 if c == "█" else ("▁▂▃▄▅▆▇".index(c) + 1) / 8
                        d.rectangle([cx, y + round(ch * (1 - frac)), cx + cw - 1, y + ch - 1],
                                    fill=color)
                elif c != " ":
                    d.text((cx, y + ch / 2), c, font=font, fill=color, anchor="lm")
            x += w
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    im.save(out, optimize=True)


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
    with open(src, encoding="utf-8", errors="replace") as f:
        capture = f.read()
    if not chrome:
        draw_pillow(trim(capture), out)
        print(f"wrote {out} (no headless chrome: drawn with Pillow)")
        return 0
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        html_path = os.path.join(td, "shot.html")
        png_path = os.path.join(td, "shot.png")
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
