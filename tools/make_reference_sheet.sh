#!/bin/bash
# Builds docs/images/reference_floor.png: the three supplied reference heads on top, the three
# matching SilkCrowd styles underneath, one-to-one and in the same order.
#
#   tools/make_reference_sheet.sh [--binary ./build/SilkCrowd]
#
# The reference is docs/images/reference_three.jpg (the three panels as supplied). The mapping is
# fixed and is the point of the image: reference 1 -> --style 4 (blunt bob), reference 2 ->
# --style 2 (ringlet curls), reference 3 -> --style 9 (longer layered blonde). The yaw matches the
# reference's three-quarter/profile framing.
set -euo pipefail

BINARY=./build/SilkCrowd
[ "${1:-}" = "--binary" ] && BINARY="$2"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -x "$BINARY" ] || { echo "error: $BINARY not found; run make first" >&2; exit 1; }
command -v convert >/dev/null 2>&1 || { echo "error: ImageMagick (convert) not found" >&2; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

REF=docs/images/reference_three.jpg
[ -f "$REF" ] || { echo "error: $REF missing" >&2; exit 1; }

# Reference panels, left to right, each 1/3 of the supplied sheet.
convert "$REF" -crop 341x460+0+0    +repage -resize 380x "$TMP/ref1.png"
convert "$REF" -crop 342x460+341+0  +repage -resize 380x "$TMP/ref2.png"
convert "$REF" -crop 341x460+683+0  +repage -resize 380x "$TMP/ref3.png"

i=0
for style in 4 2 9; do
    i=$((i + 1))
    "$BINARY" --screenshot "$TMP/our$i.png" --scene head --style "$style" --strands 55000 \
        --width 380 --height 480 --ssaa 2 --studio --head-scan female --yaw 62 >/dev/null 2>&1
    convert "$TMP/our$i.png" -resize 380x "$TMP/our$i.png"
done

convert "$TMP/ref1.png" "$TMP/ref2.png" "$TMP/ref3.png" +append \
    -bordercolor '#1c1c1f' -border 6 -resize 1160x "$TMP/top.png"
convert "$TMP/our1.png" "$TMP/our2.png" "$TMP/our3.png" +append \
    -bordercolor '#1c1c1f' -border 6 -resize 1160x "$TMP/bot.png"

convert "$TMP/top.png" "$TMP/bot.png" -append -background '#121214' "$TMP/stack.png"
H=$(identify -format %h "$TMP/top.png")
convert "$TMP/stack.png" -background '#121214' -gravity northwest -splice 0x56 \
    -font DejaVu-Sans -pointsize 20 -fill '#dfe4ea' \
    -annotate +14+26 'Reference heads (as supplied): blunt bob / ringlet curls / honey bob' \
    -annotate "+14+$((H + 34))" \
    'SilkCrowd, one-to-one: --style 4 / --style 2 / --style 9, --yaw 62 --studio' \
    docs/images/reference_floor.png

echo "wrote docs/images/reference_floor.png"
