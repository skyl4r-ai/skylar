#!/usr/bin/env python3
"""Render an asciinema v2 .cast (real recorded terminal session) to an animated GIF.

A tiny terminal emulator (SGR colours, cursor moves, CR/LF, ED/EL clears) replays
the cast onto a character grid; each sampled instant is drawn with Pillow and the
frames are assembled into a GIF. Content = the REAL recorded bytes (here: Skylar
generating COBOL, cobc compiling/running) — nothing is re-typed or faked, this only
*renders* what was recorded.

    python cast_to_gif.py in.cast out.gif [--fps 20] [--font-size 22]
"""
import argparse
import json
import sys

from PIL import Image, ImageDraw, ImageFont

# --- Catppuccin Mocha-ish palette ---
BG = (30, 30, 46)
FG = (205, 214, 244)
CURSOR = (245, 224, 220)
# 16-colour ANSI map (0-7 normal, 8-15 bright)
ANSI = {
    0: (69, 71, 90), 1: (243, 139, 168), 2: (166, 227, 161), 3: (249, 226, 175),
    4: (137, 180, 250), 5: (203, 166, 247), 6: (148, 226, 213), 7: (186, 194, 222),
    8: (88, 91, 112), 9: (243, 139, 168), 10: (166, 227, 161), 11: (249, 226, 175),
    12: (137, 180, 250), 13: (203, 166, 247), 14: (148, 226, 213), 15: (166, 173, 200),
}


class Cell:
    __slots__ = ("ch", "fg", "bold")

    def __init__(self, ch=" ", fg=FG, bold=False):
        self.ch = ch
        self.fg = fg
        self.bold = bold


class Term:
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.clear()

    def clear(self):
        self.grid = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]
        self.cx = self.cy = 0
        self.fg = FG
        self.bold = False

    def _newline(self):
        self.cy += 1
        if self.cy >= self.rows:           # scroll up
            self.grid.pop(0)
            self.grid.append([Cell() for _ in range(self.cols)])
            self.cy = self.rows - 1

    def _put(self, ch):
        if self.cx >= self.cols:
            self.cx = 0
            self._newline()
        self.grid[self.cy][self.cx] = Cell(ch, self.fg, self.bold)
        self.cx += 1

    def _sgr(self, params):
        for p in params or [0]:
            if p == 0:
                self.fg, self.bold = FG, False
            elif p == 1:
                self.bold = True
            elif p == 22:
                self.bold = False
            elif 30 <= p <= 37:
                self.fg = ANSI[p - 30]
            elif 90 <= p <= 97:
                self.fg = ANSI[p - 90 + 8]
            elif p == 39:
                self.fg = FG

    def feed(self, data):
        i, n = 0, len(data)
        while i < n:
            c = data[i]
            if c == "\x1b" and i + 1 < n and data[i + 1] == "[":
                j = i + 2
                while j < n and not (0x40 <= ord(data[j]) <= 0x7E):
                    j += 1
                if j < n:
                    seq = data[i + 2:j]
                    cmd = data[j]
                    nums = [int(x) for x in seq.split(";") if x.isdigit()] if seq.replace(";", "").isdigit() else []
                    if cmd == "m":
                        self._sgr(nums)
                    elif cmd == "H":
                        self.cy = (nums[0] - 1) if len(nums) >= 1 else 0
                        self.cx = (nums[1] - 1) if len(nums) >= 2 else 0
                        self.cy = max(0, min(self.cy, self.rows - 1))
                        self.cx = max(0, min(self.cx, self.cols - 1))
                    elif cmd == "J":
                        self.clear() if (not nums or nums[0] == 2) else None
                    elif cmd == "K":
                        for x in range(self.cx, self.cols):
                            self.grid[self.cy][x] = Cell()
                    i = j + 1
                    continue
                i += 1
                continue
            if c == "\n":
                self._newline()
            elif c == "\r":
                self.cx = 0
            elif c == "\t":
                self.cx = (self.cx // 8 + 1) * 8
            elif c == "\b":
                self.cx = max(0, self.cx - 1)
            elif c == "\x1b":
                pass
            elif ord(c) >= 32:
                self._put(c)
            i += 1


def render(term, font, fontb, cw, ch, pad, show_cursor):
    W = term.cols * cw + 2 * pad
    H = term.rows * ch + 2 * pad
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for y in range(term.rows):
        for x in range(term.cols):
            cell = term.grid[y][x]
            if cell.ch != " ":
                px, py = pad + x * cw, pad + y * ch
                d.text((px, py), cell.ch, font=(fontb if cell.bold else font), fill=cell.fg)
    if show_cursor:
        px, py = pad + term.cx * cw, pad + term.cy * ch
        d.rectangle([px, py + 1, px + cw - 1, py + ch - 1], fill=CURSOR)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--font-size", type=int, default=22)
    ap.add_argument("--hold", type=float, default=2.2, help="extra seconds on the last frame")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 = faster")
    ap.add_argument("--max-gap", type=float, default=0.7,
                    help="cap idle gaps between events to this many seconds")
    args = ap.parse_args()

    lines = open(args.cast).read().splitlines()
    hdr = json.loads(lines[0])
    cols, rows = hdr["width"], hdr["height"]
    raw = [json.loads(l) for l in lines[1:] if l.strip()]
    raw = [(t, kind, data) for (t, kind, data) in raw if kind == "o"]
    # re-time: cap long idle gaps so the GIF stays tight, then apply speed
    events, prev, acc = [], 0.0, 0.0
    for t, kind, data in raw:
        gap = min(t - prev, args.max_gap)
        acc += gap
        events.append((acc / args.speed, kind, data))
        prev = t
    total = events[-1][0]

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", args.font_size)
    fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", args.font_size)
    bb = font.getbbox("M")
    cw = bb[2] - bb[0] + 1
    ch = int(args.font_size * 1.32)
    pad = 18

    dt = 1.0 / args.fps
    frames, durations = [], []
    term = Term(cols, rows)
    ei = 0
    t = 0.0
    blink = True
    while t <= total + 1e-6:
        while ei < len(events) and events[ei][0] <= t:
            term.feed(events[ei][2])
            ei += 1
        blink = (int(t / 0.5) % 2 == 0)
        frames.append(render(term, font, fontb, cw, ch, pad, blink))
        durations.append(int(dt * 1000))
        t += dt
    # flush remaining + hold final frame
    while ei < len(events):
        term.feed(events[ei][2])
        ei += 1
    last = render(term, font, fontb, cw, ch, pad, False)
    frames.append(last)
    durations.append(int(args.hold * 1000))

    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    print(f"wrote {args.out}: {len(frames)} frames, {sum(durations)/1000:.1f}s, {frames[0].size}")


if __name__ == "__main__":
    main()
