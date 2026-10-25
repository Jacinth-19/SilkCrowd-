#!/usr/bin/env python3
"""Composites the per-platform CI screenshots into the README table.

Usage: tools/update_readme.py docs/images/ci

The CI jobs upload ``screenshots-<platform>/<platform>.png`` artifacts (and, when the
renderer printed a statistics line, a ``<platform>.txt`` sidecar).  This script rewrites
the block between the README markers so the table always shows real renders from the
platforms that actually ran:

    | Platform | Screenshot | FPS | Device | Memory |
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BEGIN = "<!-- BEGIN CI TABLE -->"
END = "<!-- END CI TABLE -->"
ORDER = ["windows", "linux", "macos", "android", "ios"]
LABELS = {
    "windows": "Windows",
    "linux": "Linux (lavapipe)",
    "macos": "macOS (MoltenVK)",
    "android": "Android (API 33)",
    "ios": "iOS simulator",
}


def read_stats(platform_dir: Path, platform: str) -> dict:
    """Finds the sidecar statistics written by the CI step, if any."""
    stats: dict = {}
    for candidate in (platform_dir / f"{platform}.json", platform_dir / f"{platform}.txt"):
        if not candidate.exists():
            continue
        text = candidate.read_text(errors="replace")
        if candidate.suffix == ".json":
            try:
                stats.update(json.loads(text))
                continue
            except json.JSONDecodeError:
                pass
        m = re.search(r"([\d.]+)\s*ms", text)
        if m:
            ms = float(m.group(1))
            if ms > 0:
                stats["fps"] = 1000.0 / ms
        m = re.search(r"device:\s*(.+)", text)
        if m:
            stats["device"] = m.group(1).strip()
        m = re.search(r"textures\s+([\d.]+)\s*MB", text)
        if m:
            stats["memory_mb"] = float(m.group(1))
        m = re.search(r"memory:\s*([\d.]+)\s*MB", text)
        if m:
            stats["memory_mb"] = float(m.group(1))
    return stats


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "docs/images/ci")
    readme = Path(argv[2] if len(argv) > 2 else "README.md")
    if not readme.exists():
        print(f"error: {readme} not found", file=sys.stderr)
        return 1

    rows: list[str] = []
    for platform in ORDER:
        png = None
        for candidate in root.rglob(f"{platform}*.png"):
            png = candidate
            break
        if png is None:
            continue
        rel = png.relative_to(readme.parent).as_posix()
        stats = read_stats(png.parent, platform)
        fps = f"{stats['fps']:.0f}" if "fps" in stats else "n/a"
        device = stats.get("device", "n/a")
        mem = f"{stats['memory_mb']:.2f} MB" if "memory_mb" in stats else "n/a"
        rows.append(f"| {LABELS.get(platform, platform)} | ![{platform}]({rel}) | {fps} | "
                    f"{device} | {mem} |")

    if not rows:
        rows.append("| - | _no CI renders yet: run the workflow_ | - | - | - |")

    table = "\n".join([
        "| Platform | Screenshot | FPS | Device | Memory |",
        "| --- | --- | --- | --- | --- |",
        *rows,
    ])

    text = readme.read_text()
    if BEGIN in text and END in text:
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
        text = pattern.sub(f"{BEGIN}\n{table}\n{END}", text)
    else:
        text += f"\n{BEGIN}\n{table}\n{END}\n"
    readme.write_text(text)
    print(f"updated {readme} with {len(rows)} platform row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
