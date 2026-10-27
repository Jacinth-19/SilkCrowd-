#!/bin/sh
# Self-test for the Vulkan diagnostics (--gpu-info) on a machine with no GPU and no driver.
#
# It builds a synthetic libvulkan.so.1 (tools/gpu_stub_loader.c) and links the real probe
# code against it, then walks the four states the report has to tell apart:
#
#   ready                 Vulkan 1.3 device with a graphics queue         -> exit 0
#   no-device             loader fine, device list empty                  -> exit 3
#   no-driver             vkCreateInstance fails (no ICD answered)         -> exit 3
#   unsupported-device    driver caps out below Vulkan 1.3                 -> exit 3
#
# Needs Vulkan headers (libvulkan-dev / the SDK). Skips with exit 0 when they are absent:
# the diagnostics are meant to work *without* Vulkan, but the self-test needs the enum
# definitions to build a stub that answers correctly.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CXX=${CXX:-c++}
CC=${CC:-cc}
INC=${VULKAN_INCLUDE_DIR:-/usr/include}

if [ ! -f "$INC/vulkan/vulkan.h" ]; then
    echo "gpu_probe_selftest: skipped (no Vulkan headers at $INC; set VULKAN_INCLUDE_DIR)"
    exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/icd.d"

"$CC" -std=c11 -O1 -shared -fPIC -I"$INC" \
    -o "$WORK/libvulkan.so.1" "$ROOT/tools/gpu_stub_loader.c" \
    -Wl,-soname,libvulkan.so.1
cp "$WORK/libvulkan.so.1" "$WORK/icd.d/libvulkan.so.1"

"$CXX" -std=c++17 -O1 -I"$ROOT/include" -I"$ROOT/src/app" -I"$INC" \
    -DSILKCROWD_WITH_VULKAN=1 -o "$WORK/harness" \
    "$ROOT/tools/gpu_probe_harness.cpp" "$ROOT/src/app/gpu_probe.cpp" \
    "$ROOT/src/core/gpu_diag.cpp" \
    -L"$WORK" -l:libvulkan.so.1 -ldl -Wl,-rpath,"$WORK"

cat > "$WORK/icd.d/stub_icd.json" <<'JSON'
{ "file_format_version": "1.0.0",
  "ICD": { "library_path": "libvulkan.so.1", "api_version": "1.3.289" } }
JSON

export VK_DRIVER_FILES="$WORK/icd.d/stub_icd.json"
export LD_LIBRARY_PATH="$WORK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

failures=0
check() {  # check <name> <expected exit> <expected substring> [env assignments...]
    name=$1
    expected_rc=$2
    needle=$3
    shift 3
    out=$(env "$@" "$WORK/harness" 2>&1) && rc=0 || rc=$?
    ok=1
    [ "$rc" -eq "$expected_rc" ] || ok=0
    printf '%s\n' "$out" | grep -q -- "$needle" || ok=0
    if [ "$ok" -eq 1 ]; then
        printf '  ok    %-22s exit %s, "%s"\n' "$name" "$rc" "$needle"
    else
        printf '  FAIL  %-22s exit %s (wanted %s), looking for "%s"\n' \
            "$name" "$rc" "$expected_rc" "$needle"
        printf '%s\n' "$out" | sed 's/^/        /'
        failures=$((failures + 1))
    fi
}

echo "gpu_probe_selftest: synthetic loader + the real probe code"
check "ready" 0 "verdict          ready" STUB_DEVICES=1
check "two devices" 0 "Stub Old Driver" STUB_DEVICES=2
check "no-device" 3 "verdict          no-device" STUB_DEVICES=0
check "no-driver" 3 "verdict          no-driver" STUB_FAIL_INSTANCE=1
check "unsupported" 3 "verdict          unsupported-device" STUB_REFUSE_13=1
check "icd reported" 0 "driver (ICD)     1 manifest(s), 1 usable" STUB_DEVICES=1

if [ "$failures" -ne 0 ]; then
    echo "gpu_probe_selftest: $failures failure(s)"
    exit 1
fi
echo "gpu_probe_selftest: all cases passed"
