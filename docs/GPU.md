# Vulkan: why a missing driver is now a diagnosis, not a mystery

SilkCrowd has two renderers:

| renderer | source | needs |
|---|---|---|
| Vulkan 1.3 (task/mesh pipeline + compute fallback) | `src/gpu`, `shaders/` | libvulkan headers at configure time, a loader, an ICD, a device with a graphics queue |
| software rasteriser | `src/render/software_renderer.cpp` | nothing |

The software rasteriser implements the same pipeline (Figure 3e textures, Eq. 1–12 frames,
the same LOD, the same shading model), which is why every image in `docs/` is reproducible on
a machine without a GPU. The Vulkan path is an *option*: when any of its four requirements is
missing, `--gpu` falls back to the software renderer.

That fallback used to be silent. A build without Vulkan and a machine without a driver printed
almost the same sentence, so "the hair looks wrong in the GPU build" and "there is no driver
here" were indistinguishable. `SilkCrowd --gpu-info` fixes the diagnosis:

```text
$ ./build/SilkCrowd --gpu-info
SilkCrowd Vulkan diagnostics
  build            SILKCROWD_WITH_VULKAN undefined (CMake did not find libvulkan)
  shaders          no SPIR-V blobs found (not needed by the software path)
  loader           not found (tried libvulkan.so.1, libvulkan.so)
  driver (ICD)     no manifest found (searched /usr/share/vulkan/icd.d, /etc/vulkan/icd.d,
                   /usr/local/share/vulkan/icd.d, /home/user/.local/share/vulkan/icd.d)
  device           not queried (no usable loader/driver)
  verdict          build-without-vulkan - this build has no Vulkan renderer (configured
                   without libvulkan headers)
  remedy           sudo apt-get install -y libvulkan-dev mesa-vulkan-drivers glslang-tools libgl1-mesa-dri
  remedy           then reconfigure: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  remedy           the software rasteriser needs no driver at all: SilkCrowd --software
```

Exit code **0** means the Vulkan path would actually run here; **3** means it would not. The
report is printed either way, so CI can gate on the code and a human can read the reason.

## The five verdicts

The probe reports the *first* thing the application would trip over, in the order it happens:

| verdict | meaning | fix |
|---|---|---|
| `ready` | loader + driver + a Vulkan 1.3 device with a graphics queue | nothing |
| `build-without-vulkan` | this binary has no Vulkan renderer (`libvulkan` headers were missing at configure time) | install the headers and rebuild |
| `no-loader` | `libvulkan.so.1` / `vulkan-1.dll` is not present | install the loader (`libvulkan1`) |
| `no-driver` | the loader is there but no driver answered `vkCreateInstance` — no ICD manifest, a manifest whose library is gone, or a driver that refuses the requested version | install an ICD (`mesa-vulkan-drivers`, the GPU vendor driver, the SDK) |
| `no-device` | an instance was created but the device list is empty | expose the GPU (`--device /dev/dri`) or use lavapipe |
| `unsupported-device` | a device exists but has no Vulkan 1.3 or no graphics queue | update the driver, or use the software path |

Two details matter in practice:

* **A device that enumerated outranks the manifest bookkeeping.** Drivers installed outside
  `/usr/share/vulkan/icd.d` (SDK builds, containers, `LD_LIBRARY_PATH`) answer the loader but
  leave no manifest behind. A machine that enumerates a usable device is reported `ready`
  even when the ICD scan found nothing — otherwise a working GPU would be called "no driver".
* **A manifest is not a driver.** A stale `*_icd.json` pointing at an uninstalled `.so` is
  printed with `(missing)` next to the path, because that is the failure mode of a trimmed
  container image and of a driver half-removed by `apt`.

## Getting a driver

**Linux (Debian/Ubuntu), CPU-only — this is what CI uses:**

```bash
sudo apt-get install -y libvulkan-dev mesa-vulkan-drivers vulkan-tools glslang-tools
vulkaninfo --summary          # should list "llvmpipe (LLVM ..., 256 bits)"
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/SilkCrowd --gpu-info  # verdict ready
./build/SilkCrowd --screenshot gpu.png --scene head --style 4 --strands 60000
```

`mesa-vulkan-drivers` provides **lavapipe**, a software Vulkan 1.3 implementation. It is slow
but complete: it runs the real task/mesh-shader pipeline, which is what makes the GPU path
testable without a GPU. Fedora/Arch equivalents: `mesa-vulkan-drivers` / `vulkan-swrast`.

**Linux with a GPU:** install the vendor driver (`nvidia-driver-*`, `mesa-vulkan-drivers` for
AMD/Intel) and make sure the container has `/dev/dri`.

**Windows:** install the LunarG Vulkan SDK (it ships `vulkan-1.dll`, `glslc` and the ICD), then
`cmake -B build && cmake --build build --config Release`. Device discovery there is
registry-based, so the report shows what the loader enumerated rather than an ICD directory.

**macOS:** `brew install vulkan-loader vulkan-headers molten-vk shaderc`, then

```bash
export VK_ICD_FILENAMES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
./build/SilkCrowd --gpu-info
```

## This sandbox

Nothing above can be installed here: there is no network route to a package mirror, and

```text
$ ls /usr/lib/x86_64-linux-gnu/libvulkan*   -> nothing
$ ls /usr/share/vulkan/icd.d                -> nothing
$ ls /usr/include/vulkan                     -> nothing
```

so the build has no Vulkan renderer at all (`SILKCROWD_WITH_VULKAN` undefined) and every image
in `docs/` comes from the software rasteriser. What *is* verified here:

* `tools/gpu_probe_selftest.sh` builds a synthetic `libvulkan.so.1`
  (`tools/gpu_stub_loader.c`) and links the **real** probe code against it, then walks all five
  verdicts — `ready`, `no-device`, `no-driver`, `unsupported-device`, and a two-device list
  where only one is usable. It needs Vulkan headers but no driver, and it is a CI step.
* The Vulkan branch of `src/app/gpu_probe.cpp` is compiled under `-Wall -Wextra -Werror`
  against the official Khronos headers before every push.

What is still **not** verified here: the actual rendering of `src/gpu` on a device. That is what
CI is for. The `linux` job is the hard gate: it installs lavapipe, runs the self-test, runs
`--gpu-info` (**fails** unless the verdict is `ready`) and then renders the headless Vulkan
screenshot. Until this change it did not: that step ended in `|| true` and the ICD filename was
hard-coded, so a broken GPU build could sit behind a green run.

The `windows` and `macos` jobs run `--gpu-info` too, but only report (with a `::warning::`
annotation) and keep their existing tolerance: a hosted Windows runner has the loader and no ICD,
and MoltenVK depends on the runner exposing Metal. Their reports are uploaded as
`vulkan-info-*` artifacts, so a red render step there can be read against the machine's actual
state instead of guessed at.

## Why not just ship a software ICD?

It was tried, and it is not possible from inside this sandbox: PyPI has no `libvulkan` binary
(`vulkan` is Python bindings only; `vulkan-loader`, `lavapipe`, `swiftshader` and
`mesa-vulkan-drivers` are not published there), npm has no ICD package, and building Mesa needs
a toolchain and a dependency set this image does not have. A driver dropped into a temporary
directory would not survive the sandbox anyway — the useful fix is the one above: make the
failure explicit, testable, and gated in CI.
