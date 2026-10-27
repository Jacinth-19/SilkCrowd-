#!/bin/sh
# Regenerates every asset in assets/ from the procedural sources:
#   assets/heads/female_head.gltf + .bin      the scalp/head model (GLTF 2.0)
#   assets/hairmeshes/NN_<name>.skhm          the ten Figure 10 presets (hero quality)
#   assets/hairmeshes/crowd/NN_<name>.skhm    the Figure 1 crowd budget (13-21 KB each)
#
# The .skhm container is documented in src/app/assets_export.cpp: a header followed by the
# five hair mesh textures (Table 1) stored as IEEE binary16 texels.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT/build/SilkCrowd"
if [ ! -x "$BIN" ]; then
    echo "building $BIN first"
    make -C "$ROOT" -j
fi
mkdir -p "$ROOT/assets/heads" "$ROOT/assets/hairmeshes/crowd"
cd "$ROOT"
"$BIN" --assets
