#!/bin/bash
# Renders every clip in docs/media and the reference comparison sheet in docs/images.
#
#   tools/make_media.sh [--binary ./build/SilkCrowd] [--ffmpeg ffmpeg]
#
# Each clip is a turntable of this repository's own renders: one `--screenshot` call per
# frame, then ffmpeg turns the frames into an mp4 (H.264) and a palette-optimised GIF.
# Nothing here is pre-rendered anywhere else.
set -euo pipefail

BINARY=./build/SilkCrowd
FFMPEG=${FFMPEG:-ffmpeg}
while [ $# -gt 0 ]; do
    case "$1" in
        --binary) BINARY="$2"; shift 2 ;;
        --ffmpeg) FFMPEG="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -x "$BINARY" ] || { echo "error: $BINARY not found; run make first" >&2; exit 1; }
command -v "$FFMPEG" >/dev/null 2>&1 || { echo "error: $FFMPEG not found" >&2; exit 1; }

FRAMES=$(mktemp -d)
trap 'rm -rf "$FRAMES"' EXIT
mkdir -p docs/media docs/images

render() {   # render <dir> <count> <fps> <yaw step> -- <SilkCrowd args...>
    local dir=$1 count=$2 fps=$3 step=$4; shift 5
    mkdir -p "$FRAMES/$dir"
    for ((i = 0; i < count; ++i)); do
        "$BINARY" --screenshot "$FRAMES/$dir/f$(printf %03d "$i").png" --yaw $((i * step)) "$@" \
            >/dev/null 2>&1
        printf '\r%s %d/%d' "$dir" $((i + 1)) "$count"
    done
    printf '\n'
}

encode() {   # encode <dir> <fps> <out base> <gif width> <gif height> <mp4 width> <mp4 height>
    local dir=$1 fps=$2 base=$3 gw=$4 gh=$5 mw=$6 mh=$7
    "$FFMPEG" -y -loglevel error -framerate "$fps" -i "$FRAMES/$dir/f%03d.png" \
        -vf "scale=$mw:$mh:flags=lanczos" -c:v libx264 -pix_fmt yuv420p -crf 20 \
        -movflags +faststart "docs/media/$base.mp4"
    "$FFMPEG" -y -loglevel error -framerate "$fps" -i "$FRAMES/$dir/f%03d.png" \
        -vf "scale=$gw:$gh:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];\
[b][p]paletteuse=dither=bayer:bayer_scale=3" "docs/media/$base.gif"
    echo "  docs/media/$base.{mp4,gif}"
}

echo "bob turntable (36 frames, 10 deg apart)"
# The reference cut is a chin-length blunt bob, so the turntable is a hero head at 45k strands.
render bob 36 12 10 --scene head --style 4 --strands 45000 --width 320 --height 400 \
    --ssaa 1 --studio --head-scan female
encode bob 12 bob_turntable 280 350 320 400

echo "crowd orbit (16 frames, 22 deg apart)"
render crowd 16 8 22 --scene crowd --crowd 48 --width 480 --height 270 --ssaa 1 --studio
encode crowd 8 crowd_orbit 440 248 480 270

echo "teapot orbit (10 frames, 36 deg apart)"
# The teapot grid is the slowest scene here (~20 s a frame), so it gets the fewest frames.
render tea 10 5 36 --scene teapot --width 400 --height 225 --ssaa 1 --studio
encode tea 5 teapot_orbit 400 224 400 224

echo "reference comparison sheet"
./tools/make_reference_sheet.sh

echo "done: docs/media/, docs/images/reference_floor.png"
