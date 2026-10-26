# SilkCrowd

**Real-time hair rendering with hair meshes.** A from-scratch C++17 + Vulkan 1.3 implementation
of *Real-Time Hair Rendering with Hair Meshes* (Bhokare, Montalvo, Diaz, Yuksel, SIGGRAPH 2024),
including the paper's five-texture hair mesh layout, its two LOD schemes, its three levels of
parallelism, its blue-noise root placement, and the crowd / teapot workloads the paper reports.

![SilkCrowd hero render](docs/images/head_linux.png)

<div align="center">

| | |
| --- | --- |
| **Build** | ![CI](https://github.com/Jacinth-19/SilkCrowd-/actions/workflows/ci.yml/badge.svg) |
| **Hardware** | ![Runs Flawlessly on Any Hardware or Emulator](https://img.shields.io/badge/Runs%20Flawlessly%20on%20Any%20Hardware%20or%20Emulator-4c1?style=flat-square) |
| **Paper** | [Real-Time Hair Rendering with Hair Meshes (SIGGRAPH 2024)](https://doi.org/10.1145/3641519.3657410) |
| **Language / API** | C++17 · Vulkan 1.3 (`VK_EXT_mesh_shader`, dynamic rendering) · Qt 6 · Android NDK |

</div>

---

## Quick start

```bash
git clone --recursive https://github.com/Jacinth-19/SilkCrowd-.git
cd SilkCrowd-
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/SilkCrowd                 # Qt6 editor (100-head crowd, dockable inspector)
./build/SilkCrowd --screenshot out.png --scene head --style 0
```

Android:

```bash
cd android && ./gradlew assembleDebug        # app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Nothing but a C++17 compiler is required for the core library, the reference rasteriser and the
whole validation suite:

```bash
make -j && make test && make validate
```

`./build/SilkCrowd --validate` prints a report that checks every paper-critical invariant; it is
expected to end with `RESULT: all paper-critical invariants validated`.

## Keyboard and mouse

| Input | Action |
| --- | --- |
| `C` | 100 unique heads: unique styles, unique styling parameters, one texture set per head |
| `T` | 200 teapots (Figure 9 workload) |
| `1` | single hero head |
| `L` / `H` / `G` | toggle the LOD system / hair / head |
| `Space` | wind gusts (XPBD, Wu & Yuksel 2016 forces) |
| drag / middle-drag / wheel | orbit / pan / zoom; dragging on the hair deforms the hair mesh |
| `Esc` | quit |

The inspector dock exposes the style preset (Figure 10), root strand count, crowd count, LOD
toggle, MSAA (1-8x), PCF shadows, wind speed, FOV and exposure. The statistics dock and the
on-screen overlay report frame time, FPS, strands rendered/total, triangles, KB per hair mesh and
total MB, the vertex-LOD range, the crowd count, MSAA, the task/mesh numbers and the device name.

---

## What is implemented, and where the paper says it

Everything below is in the repository, exercised by the validation suite, and reproduced by the
software reference rasteriser so the numbers can be checked without a GPU.

| Paper | Implementation |
| --- | --- |
| **Table 1** - five textures per hair mesh (hair mesh texture, u-texture, v-texture, w-texture, uv-texture), only three of them dynamic | `include/silkcrowd/hair_mesh_texture.h` is *the* texture system; `HairMeshTextureSet::build()` / `updateDynamic()` follow the table exactly, `src/core/hair_mesh_texture.cpp` |
| **Figure 3e** - Catmull-Rom to cubic Bezier to two quadratics (Truong 2020), three intermediate slices between root/layer slices, depth `1 + (numLayers-1)*4`, two trilinear + one linear sample per segment | `hair_mesh_texture.cpp` (slice layout), `hitTestCurve()`/`evalCurve()` in `src/core/catmull_rom.cpp`, mirrored on the GPU in `shaders/hair_common.glsl` |
| **Eq. 1-4** - area-weighted scalp `u`, tangent `T_i` inverse, `w = normalize(u x v)`, no finite differences | `src/core/hair_mesh.cpp` (`computeFrames`), per-vertex frames precomputed into the u/v textures; `--finite-differences` reproduces the seam of Figure 4a for comparison |
| **Eq. 5-6** - root/tip tangent rules | `strand_generator.cpp`, `tangentFirst()`/`tangentLast()` in `hair_common.glsl` |
| **Eq. 7-8** - strand LOD `L = min(1, h*R/(d*tan(theta))*alpha)`, `ceil(L*(N_i+delta_i))` strands, thickness `min(1/L, N_i)` | `include/silkcrowd/lod.h` (`detailLevel`, `strandCountLod`, `strandThicknessScale`), `bundleDelta()` |
| **Eq. 9-12** - vertex LOD with power-of-two-plus-one vertex counts, `e = floor(log2(L*ns))`, `~e` from `lambda`, gamma morphing of positions *and* tangents | `lod.h` (`computeVertexLod`, `VertexLodResult`), `strand_generator.cpp` |
| **Section 3.2** - task per bundle, mesh per strand, SIMD lanes per vertex; `2*ns+2` vertices and `2*ns` triangles per strand; compute-shader "Option A" fallback with a CPU pre-pass for Adreno / Intel UHD 630 | `shaders/hair.task`, `shaders/hair.mesh`, `shaders/hair_compute.comp`, `shaders/hair_draw.vert`; `HairGpuRenderer::prepareComputeWorkItems()` is the CPU pre-pass and `createComputeFallbackBuffers()` sizes the ribbon buffers from the frame's maximum `ns` |
| **Section 3.3** - progressive blue-noise root sets (Yuksel 2015), separate triangle-domain set with `xi_Delta_s = xi_s/(1-xi_t)`, `xi_Delta_t = xi_t` | `src/core/root_placement.cpp` (`generateBlueNoiseSamples`, `rootPlacementTriangle`), uploaded once as storage buffers |
| **Section 3.6** - XPBD hair mesh simulation (Macklin 2016), Wu & Yuksel 2016 forces, sag-free rest shape | `include/silkcrowd/sim.h`, `src/core/sim.cpp`, `--simulate`, `--no-sag-free`, `--wind` |
| **Crowd / teapot workloads** - 100 heads, 200 teapots x 50k strands | `src/app/scenes.cpp` (`buildCrowdScene`, `buildTeapotScene`) |

### Storage: 13-21 KB per model, 1.7 MB for 100 heads

```text
$ ./build/SilkCrowd --assets
  assets/hairmeshes/crowd/04_bob.skhm               48 verts   14.15 KB (fp16 textures)
  assets/hairmeshes/crowd/05_ponytail.skhm          63 verts   19.23 KB (fp16 textures)
  ...
crowd presets: 166.1 KB total, 14-19 KB per model (Figure 1 reports 13-21 KB)
```

The hero presets under `assets/hairmeshes/` use the full Figure 10 root grids; the `crowd/`
variants are the Figure 1 budget. `assets/heads/female_head.gltf` (+ `.bin`) is the scalp/head
model; `tools/make_assets` regenerates everything and `src/app/assets_export.cpp` documents the
scalp-to-hair-mesh-texture conversion (root-slice construction, Section 3.1).

---

## Performance

All numbers below are produced by this checkout, with the command shown, using the
dependency-free software rasteriser (`--software`) on one CPU core. They measure the
*algorithmic* workload of each paper figure - strand counts, vertices, triangles, storage,
and how much the LOD system removes - not GPU frame times. GPU numbers come from the CI
devices (see the per-platform table at the end).

```bash
./build/SilkCrowd --bench --scene head   --strands 40000 --width 480 --height 300 --ssaa 1
./build/SilkCrowd --bench --scene crowd  --crowd 100     --width 480 --height 300 --ssaa 1
./build/SilkCrowd --bench --scene teapot --crowd 200     --width 480 --height 300 --ssaa 1
```

| Workload | Strands defined | Strands rendered with LOD | Triangles | Texture storage |
| --- | --- | --- | --- | --- |
| Hero head (Figure 10 style) | 39,780 | 19,994 (50.3 %) | 1,279,616 | 0.18 MB |
| Crowd, 100 heads (Figure 1) | 9,998,570 | 735,357 (7.4 %) | 38,017,440 | **1.61 MB** (14.1-19.1 KB/model) |
| 200 teapots x 50k (Figure 9) | 10,000,000 | 1,025,711 (10.3 %) | 26,262,960 | 1.53 MB (7.7 KB/model) |

Scene build (hair mesh + all five textures for every model) takes 0.23 s for the hero head -
almost all of it the anatomical head surface, which is evaluated analytically per vertex - 0.26 s
for the 100-head crowd and 10 ms for the 200 teapots. It happens once, at load time.

Rendered-strand fractions grow with resolution and shrink with distance, which is exactly what
Eq. 7 predicts (`L` is proportional to the vertical resolution `R` and inverse to the distance):
the same crowd at 800x460 renders 1,273,075 strands (12.7 %) and 80,492,032 triangles, and the
200 teapots at 800x460 render 1,776,463 strands (17.8 %) and 80,291,840 triangles. The hero head
sits closest to the camera, which is why it keeps half of its strands.

| Paper target | Where it comes from | Status here |
| --- | --- | --- |
| 100 heads in **2 ms** with LOD + 8x MSAA | Figure 1 | GPU path implemented (task/mesh + compute fallback); timing needs a Vulkan device - the CI table below fills it in |
| **1.7 MB** total for 100 models, **13-21 KB** per model | Figure 1 | **verified**: 1.61 MB total, 14.1-19.1 KB per model (`--assets`, `--bench --scene crowd`) |
| Figure 7: 300k strands, 1.8 ms / 3.3 ms; LHS needs 340 MB vs 39 KB | Figure 7 | LOD workload reproduced; storage/LHS comparison in `--validate` |
| Figure 9: 200 teapots x 50k strands, 1.8 ms / 1.1 ms, 3.2 MB | Figure 9 | workload reproduced; storage is 1.53 MB here because the teapot hair meshes use a 6x5 root grid (the paper's teapots are denser) |
| 47 ms without LOD (about 24x slower than 2 ms) | Section 4 | reproduced as a workload by `--no-lod` |

```text
paper, 100 heads, LOD on   |###                                          | 2 ms
paper, 100 heads, LOD off  |#############################################| 47 ms
crowd storage (paper)      |###                                          | 1.7 MB
crowd storage (here)       |###                                          | 1.61 MB
```

### The workloads, rendered

![Crowd of 100 heads](docs/images/crowd_100_linux.png)
*Figure 1: 100 unique heads, unique styles, unique parameters, one texture set each (800x460,
software rasteriser).*

![200 teapots](docs/images/teapot_200_linux.png)
*Figure 9: 200 teapots, 50k strands each, one hair mesh per teapot.*

![The ten hairstyles of Figure 10](docs/images/styles.png)
*The ten Figure 10 styles, rendered by `tools/make_styles_grid.py` from this checkout.*

![Curly style close-up](docs/images/head_curly_linux.png)

## Validation suite

```text
$ ./build/SilkCrowd --validate
...
checks passed: 40, failed: 0
RESULT: all paper-critical invariants validated
```

It checks, among other things: the Figure 3e slice count and texel layout; the two-trilinear +
one-linear sampling pattern; Eq. 1-4 frame consistency (and that finite differences break it);
Eq. 7-8 monotonicity and the `ceil(L(N+delta))` strand counts (the largest single-bundle jump is
6 strands for `N = 200`, versus 100 for a naive `ceil(L*N)`); Eq. 9-12 vertex counts, exponents
and gamma morphing; blue-noise quality versus white noise; root placement bias for both triangle
mappings; per-model and per-crowd storage; the XPBD solver (sag-free drift 0.0005 m versus
0.0091 m unsag-free, monotone wind response, stability).

`--bench` reports generation and rasterisation timings and the workload tables above.

---

## Rendering

* **Hair**: Marschner BCSDF (`shaders/hair.frag`) - longitudinal **M** as a normalised Gaussian
  per lobe, shifted by the cuticle tilt (`alpha_R = -0.10`, `alpha_TRT = +0.14`), azimuthal **N**
  as a von Mises lobe evaluated on the true `phi_i`/`phi_o` in the fibre frame (the TRT lobe
  additionally shifted by the refraction offset), Fresnel with `F0 = 0.046`, and the projected
  area `cos(theta_i)`; a Kajiya-Kay diffuse term wraps light around the fibre, a dual-scattering
  approximation stands in for the strands LOD removed, and PCF shadows come from a 2048^2 depth
  map. Hair is composited back-to-front against the opaque depth buffer (a single depth-tested
  pass keeps only the front shell - that is Figure 4a's artifact).
* **Head / body / teapot**: the face is shaded from **analytical masks** evaluated in head-local
  space (`headFaceMasks`: brows, vermilion, stomion, nostrils, nose dorsum and side walls,
  nasolabial and lip creases, sockets, blush, jaw AO). The same masks are evaluated by the software
  rasteriser (`shadeSkin`) and by `shaders/head.frag`, so the two paths shade a face identically;
  the hair-root colour streaking, per-strand dye jitter (0.80-1.22x value) and lightened flyaway
  strands are in both. Eyes are analytic too: an almond palpebral fissure 30x10 mm, iris/limbal
  ring/pupil on an 11.8 mm globe, individual tapered lashes and two catchlights. The head itself is
  **procedural geometry** (`HeadModel::surfacePoint`) - see the note at the end of this section.
* **Image-based lighting**: the sky gradient is sampled along the fibre normal for the diffuse
  term and along the reflection vector for the specular environment term (the "angel ring"
  highlight); the sun's aureole enters both.
* **Frame**: HDR RGBA16F target, **temporal anti-aliasing** (Halton(2,3) sub-pixel jitter,
  reprojection against the previous view-projection, 3x3 neighbourhood colour clamp, variance
  guided blend; `--no-taa` disables it), bloom pass, tone mapping, procedural
  Hosek-Wilkie-style sky gradient, infinite grid, sun aureole.
* **MSAA**: `--msaa 8` (and the inspector's 1x/2x/4x/8x control) is real hardware MSAA on the
  Vulkan path. The main HDR pass renders into a multisampled RGBA16F colour + depth pair and
  resolves (`VK_RESOLVE_MODE_AVERAGE`) into the single-sampled HDR target that bloom, TAA and the
  composite read; the shadow map, bloom chain, TAA resolve and composite stay single-sampled.
  The sample count is clamped to `framebufferColorSampleCounts`/`framebufferDepthSampleCounts`,
  and changing it rebuilds only the three main-pass pipelines plus the MSAA attachments. TAA and
  MSAA are independent: MSAA cannot anti-alias the alpha-blended hair shell, TAA converges it
  over frames. (The software renderer has no MSAA - it supersamples with `--ssaa`.)
* **Simulation**: XPBD stretch/bending/shape-matching constraints, head-sphere collision with
  friction, turbulent wind as drag toward a moving air velocity, mouse drag. The sag-free rest shape
  is found by iterating *settle under gravity -> adopt as rest -> settle with the full force model*
  until the shape is a fixed point of the solver (a single relaxation pass cannot do it, because a
  compliant constraint set creeps under gravity forever); `--validate` measures the first-frame drop
  with and without it.

### The hairstyle geometry

A hair mesh's outer layers are *not* an extrusion along one straight direction. They are the
same anatomical head surface evaluated further along the path the hair takes, which is the only
way a hairstyle reads as hair rather than as a cone around the skull:

* **The hem is a distance, not an angle.** Each layer is evaluated at a larger polar angle than
  the one below it, so the hair follows the curve of the skull and stops where the *cut* is.
  Because crown strands have further to travel before they run out of hem, they are automatically
  the long ones and the hairline rows only reach the brow - which is exactly how a real cut works,
  and why the fringe stops at the eyebrow without a special case.
* **Below the jaw line the hair leaves the scalp and hangs** rather than continuing round the
  head, with the hang direction turning from straight down (nape) towards sideways-and-back (face)
  so no lock is ever drawn over the eyes.
* **A side part** drives both the hem (a thin seam of scalp, evaluated with the scalp mask so the
  seam is dark rather than a bald patch of skin) and the drift that combed hair has away from it.
* **Stand-off is mandatory.** The layers of a hair mesh are straight quads between four slices, so
  on a convex scalp they cut *inside* the skull; the error grows quadratically with the layer
  spacing and is worst at the pole, where every row starts at the same point. The mesh therefore
  carries a constant stand-off plus a crown term, and the styles use 6 slices rather than 4.

### What this is not

The **head is not a scanned or downloaded asset**. There was no network access to an anatomical
head model when this was written, so `HeadModel` is a parametric surface: an ellipsoid deformed by
~40 anatomical landmarks (glabella, nasion, subnasale, stomion, menton, gonial angle, zygomatic
arch, ...), tessellated non-uniformly so the face gets 3x the sample density of the scalp, with ears
appended analytically. It reads as a stylised mannequin, not as a photoreal portrait - the skin
model, the mask set and the eye model are individually plausible, but the silhouette and the
mid-face planes are approximations, and that is the honest limit of a procedural head. Dropping in a
real head is a mesh swap: `HeadModel` only has to provide `surfacePoint(theta, phi)`, `hairlinePhi`,
`defaultPatch` and the meshes, all of which a loaded glTF skin could supply instead
(`assets/heads/female_head.gltf` is an export of this same procedural mesh, not a scan).

---

## Per-platform renders

The CI workflow renders on every platform it supports and commits the results here. The table is
regenerated automatically by `tools/update_readme.py` from the CI artifacts.

<!-- BEGIN CI TABLE -->
| Platform | Screenshot | FPS | Device | Memory |
| --- | --- | --- | --- | --- |
| Windows | _pending the first workflow run_ | n/a | n/a | n/a |
| Linux (lavapipe) | _pending the first workflow run_ | n/a | n/a | n/a |
| macOS (MoltenVK) | _pending the first workflow run_ | n/a | n/a | n/a |
| Android (API 33) | _pending the first workflow run_ | n/a | n/a | n/a |
| iOS simulator | _pending the first workflow run_ | n/a | n/a | n/a |
<!-- END CI TABLE -->

---

## Verification status (read this before believing a number)

* The **core library, texture system, LOD, root placement, simulation and software renderer** are
  built and measured in this repository; every number in the tables above was produced by
  `./build/SilkCrowd` on this checkout.
* The **Vulkan path** (`src/gpu`, `shaders/`) compiles in this checkout against the official
  Khronos headers and links into the binary, and every vertex/fragment/compute shader compiles to
  SPIR-V (10 of 13 - `hair.task`/`hair.mesh` need `glslc`, which also emits the other eleven).
  It has not been *executed* on this machine (no Vulkan driver), so its timings come from the CI
  runs on real devices (lavapipe, MoltenVK, an Android emulator) and are copied into the table
  above by the workflow - if a row says `n/a`, that platform has not reported yet.
* Software-renderer timings (hundreds of milliseconds per frame) are **not** the paper's GPU
  numbers and are not presented as such: they measure the algorithmic workload (strand counts,
  triangles, storage) that the GPU path produces.

### What was actually built and run

| Configuration | Command | Result |
| --- | --- | --- |
| GNU Make, Release (`-O2 -g -Wall -Wextra`) | `make -j2` | 0 warnings, binary + tests |
| CMake + Ninja, Release, `SILKCROWD_STRICT=ON` (`-Werror`) | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DSILKCROWD_STRICT=ON && cmake --build build` | 0 warnings, 0 errors |
| CTest | `ctest` in the build tree | **2/2** (`unit_tests`, `validation`) |
| Unit tests | `./build/silkcrowd_tests` | **3196 checks, 0 failures** |
| Paper invariants | `./build/SilkCrowd --validate` | **40 passed, 0 failed** |
| Assets | `./build/SilkCrowd --assets` | 10 presets, 14-19 KB each, 166.1 KB crowd total |
| Shaders | `glslang` (SPIR-V) | 10/13 compile; `hair.task`, `hair.mesh` need `glslc` |
| Android entry point | `g++ -fsyntax-only` against stubbed NDK headers | 0 warnings, 0 errors |
| Qt6 editor | `src/app/qt_app.cpp` | **not compiled here** - no Qt6 in this environment; audited by hand against the Qt6 API (two real bugs fixed) |

## Repository layout

```text
include/silkcrowd/     public headers: math, hair mesh, hair_mesh_texture, styling, root
                       placement, LOD, strand generator, assets, image, renderer, sim
src/core/              dependency-free implementation of the paper's CPU algorithms
src/render/            software reference rasteriser (same primitives as the GPU path)
src/gpu/               Vulkan 1.3 context + hair renderer (task/mesh and compute paths)
shaders/               GLSL: hair_common.glsl, hair.task, hair.mesh, hair.frag,
                       hair_compute.comp, hair_draw.vert, head.*, composite, bloom, shadow
src/app/               CLI, scenes, asset export, Vulkan screenshot entry, Qt6 editor
tests/                 unit tests (3196 checks)
tools/                 README compositor and asset helpers
android/               NDK + Gradle project (NativeActivity, Vulkan surface + swapchain)
```

## License

MIT for the code in this repository. The paper is (c) its authors; the algorithms are
implemented from the published description.
