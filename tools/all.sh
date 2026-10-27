#!/bin/bash
set -e
cd /home/user/SilkCrowd-
FF=/tmp/vv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
./build/SilkCrowd --screenshot docs/images/head_linux.png --scene head --style 4 --strands 60000 --width 640 --height 800 --ssaa 2 --studio --head-scan female >/dev/null 2>&1
echo HERO_DONE
./build/SilkCrowd --screenshot docs/images/head_curly_linux.png --scene head --style 2 --strands 60000 --width 640 --height 800 --ssaa 2 --studio --head-scan female >/dev/null 2>&1
echo CURLY_DONE
python3 tools/make_styles_grid.py --size 220 --studio --out docs/images/styles.png >/dev/null 2>&1
python3 tools/make_styles_grid.py --size 240 --studio --head-scan female --out docs/images/styles_female.png >/dev/null 2>&1
echo GRIDS_DONE
./tools/make_reference_sheet.sh >/dev/null 2>&1
echo SHEET_DONE
FFMPEG=$FF ./tools/make_media.sh > /tmp/media3.log 2>&1
echo MEDIA_DONE
./build/SilkCrowd --screenshot docs/images/crowd_100_linux.png --scene crowd --crowd 100 --width 800 --height 460 --ssaa 1 --studio >/dev/null 2>&1
echo CROWD_DONE
./build/SilkCrowd --screenshot docs/images/teapot_200_linux.png --scene teapot --width 800 --height 450 --ssaa 1 --studio >/dev/null 2>&1
echo TEAPOT_DONE
