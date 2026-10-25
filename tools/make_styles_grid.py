#!/usr/bin/env python3
"""Renders the ten Figure 10 styles and composites them into one grid image.

    tools/make_styles_grid.py [--size 220] [--out docs/images/styles.png]

The renders come from `./build/SilkCrowd --screenshot`, so the grid always shows exactly what
the renderer produces. PNG decode/encode is done here with zlib only (no Pillow), matching the
dependency-free spirit of the rest of the repository.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

STYLE_NAMES = [
    "long straight", "wavy medium", "curly short", "afro", "bob",
    "ponytail", "beach waves", "curly long", "pixie", "platinum bob",
]


# --------------------------------------------------------------------------- PNG
def read_png(path: Path) -> tuple[int, int, list[list[tuple[int, int, int]]]]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    pos, width, height, bit_depth, color_type = 8, 0, 0, 0, 0
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    assert bit_depth == 8, "only 8-bit PNGs are supported"
    raw = zlib.decompress(idat)
    stride = width * channels
    rows: list[bytearray] = []
    prev = bytearray(stride)
    i = 0
    for _ in range(height):
        filt = raw[i]
        i += 1
        line = bytearray(raw[i : i + stride])
        i += stride
        if filt == 1:      # Sub
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif filt == 2:    # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif filt == 3:    # Average
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif filt == 4:    # Paeth
            for x in range(stride):
                a = line[x - channels] if x >= channels else 0
                b = prev[x]
                c = prev[x - channels] if x >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 0xFF
        rows.append(line)
        prev = line
    pixels = [
        [tuple(line[x * channels : x * channels + 3]) for x in range(width)]
        for line in rows
    ]
    return width, height, pixels


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for line in pixels:
        raw.append(0)  # filter: none
        for rgb in line:
            raw.extend(rgb)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
                     chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=220)
    ap.add_argument("--out", default="docs/images/styles.png")
    ap.add_argument("--binary", default="./build/SilkCrowd")
    ap.add_argument("--strands", type=int, default=20000)
    args = ap.parse_args(argv[1:])

    binary = Path(args.binary)
    if not binary.exists():
        print(f"error: {binary} not found; run make first", file=sys.stderr)
        return 1

    size = args.size
    labels_height = 14
    cols, rows = 5, 2
    grid: list[list[tuple[int, int, int]]] = [
        [(18, 18, 20)] * (cols * size) for _ in range(rows * (size + labels_height))
    ]

    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(STYLE_NAMES):
            out = Path(tmp) / f"style{i}.png"
            cmd = [str(binary), "--screenshot", str(out), "--scene", "head", "--style", str(i),
                   "--strands", str(args.strands), "--width", str(size),
                   "--height", str(size), "--ssaa", "1", "--software"]
            print(f"[{i + 1}/10] {name}")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            w, h, px = read_png(out)
            ox = (i % cols) * size
            oy = (i // cols) * (size + labels_height)
            for y in range(min(h, size)):
                for x in range(min(w, size)):
                    grid[oy + y][ox + x] = px[y][x]
            # Style index stripe, so the grid is self-describing.
            for y in range(size, min(size + labels_height, size + labels_height)):
                for x in range(ox, ox + size):
                    on = (x - ox) < (i + 1) * (size // 11)
                    grid[oy + y][x] = (90, 140, 200) if on else (30, 32, 36)

    write_png(Path(args.out), grid)
    print(f"wrote {args.out} ({cols * size}x{rows * (size + labels_height)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
