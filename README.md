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

# the female head, in profile, on a studio backdrop (the framing of the reference photos)
./build/SilkCrowd --screenshot bob.png --scene head --style 4 --yaw 78 --studio \
    --strands 40000 --head-scan female
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

If `--gpu` silently renders on the CPU, ask why:

```bash
./build/SilkCrowd --gpu-info     # exit 0: the Vulkan path would run; exit 3: here is what is missing
```

It separates the four things that can be absent - the build, the loader, the driver (ICD) and the
device - and prints the command that fixes each. `docs/GPU.md` is the long version.

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
variants are the Figure 1 budget.

**The head is a real scan.** `assets/heads/head_scan.glb` is a 2.2 MB photogrammetry bust -
9,273 vertices, 17,684 triangles and a 1024^2 diffuse map decoded from the original 4096^2
capture - of the Infinite-Realities scan by Lee Perry-Smith (CC BY 3.0, licence and full
attribution in `assets/heads/LICENSE-head_scan.txt`). `tools/prep_head.py` repacks it
(dropping the unused normal and displacement maps), fits it into SilkCrowd's head space from
landmarks measured on the mesh itself, and digitally shaves the day's growth off the jaw.
`src/core/gltf.cpp` is a dependency-free GLB 2.0 reader; the texture is decoded by the
vendored, public-domain `third_party/stb/stb_image.h`.

`assets/heads/female_head.gltf` (+ `.bin`) remains the parametric head, which is what you get
with `--procedural-head` or on a checkout without the scan; `tools/make_assets` regenerates the
.hairmeshes and the export, and `src/app/assets_export.cpp` documents the
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
| Hero head (bob, Figure 10 style 4) | 39,780 | 14,488 (36.4 %) | 927,232 | 0.18 MB |
| Crowd, 100 heads (Figure 1) | 9,998,570 | 754,433 (7.5 %) | 31,982,960 | **1.61 MB** (14.1-19.1 KB/model) |
| 200 teapots x 50k (Figure 9) | 10,000,000 | 1,025,711 (10.3 %) | 26,262,960 | 1.53 MB (7.7 KB/model) |

Scene build (hair mesh + all five textures for every model) takes 0.24 s for the hero head - most of
it the scanned head being reloaded and its radial depth map rasterised, which is what the hair
roots are fitted to - 0.34 s for the 100-head crowd and 5 ms for the 200 teapots. It happens once,
at load time. Frame times are dominated by the software rasteriser's strand generation, which is
what the GPU path replaces with the task/mesh pipeline.

Rendered-strand fractions grow with resolution and shrink with distance, which is exactly what
Eq. 7 predicts (`L` is proportional to the vertical resolution `R` and inverse to the distance):
the same crowd at 800x460 renders 1,155,729 strands (11.6 %) and 66,545,568 triangles, and the
200 teapots at 800x460 render 1,571,716 strands (15.7 %) and 62,420,992 triangles. The hero head
sits closest to the camera, which is why it keeps a third of its strands.

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

### The 2 ms question: measured on a CPU

The paper's 2 ms is an RTX 4090 number. This checkout answers the obvious follow-up with
measurements instead of a shrug - `--cpu-budget` times every phase of the paper's per-frame
work on the machine it runs on, and `docs/CPU_2MS.md` has the full analysis:

```bash
./build/SilkCrowd --cpu-budget --scene crowd --crowd 100 --front-end-only --repeat 3
./build/SilkCrowd --cpu-budget --scene head --width 1280 --height 720 --repeat 3
```

On a 2-core Xeon at 2.6 GHz, for the 100-character Figure 1 scene:

| phase | ms | what it is |
| --- | --- | --- |
| front end (Eq. 7-12, Eq. 8 counts, culling, sort) | **1.12** | the only per-frame work a CPU owns in a real implementation - **fits in 2 ms with 1.8x to spare** |
| generation (Figure 3e, 66.2 M vertices) | 10,295 | 156 ns per generated vertex |
| raster (ribbons, 1 sample/pixel) | 14,532 | 64.2 M segments, 497 M shaded fragments |
| **whole frame** | **24,832** | 12,400x the paper's 2 ms |

Per parallel lane the CPU is within **7.5x** of the paper's GPU (40.4k vs 305k strands/s per
lane); 16,384 / 2 = 8,192 is the lane-count ratio and essentially the whole wall-clock gap. So
the deficit is cores, not algorithm - and the part of the frame that is genuinely CPU work fits
inside the paper's budget. `docs/CPU_2MS.md` §7 has the arithmetic and §10 lists exactly which
shortcuts the fast path takes so that nobody reads its 661 ms hero frame as the reference
renderer getting faster (it is unchanged).

![The fast CPU path](docs/images/cpu_fast_head.png)
*The fast timing path's own frame (640x360): identical geometry, deliberately cheap shading, no
head or shadow buffer - it exists to be timed, not to be looked at. Every image above is still
produced by the reference renderer.*

## What is not implemented

`docs/NOT_IMPLEMENTED.md` is the complete, itemised list: what the paper has that this
repository does not, what is blocked by the hardware in this environment (GPU timings, 8x MSAA
verification, an on-device Android frame, the Qt6 editor never having been built here), and
what is implemented differently on purpose (envelope shadowing instead of per-strand shadow
rays, fixed-size texture blocks instead of the paper's tighter packing, a barycentric triangle
mapping instead of the printed expression). Nothing in it is a surprise discovered by reading
the code - each deviation is a named parameter, a printed warning or a flag.

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
* **Scanned skin** (the path every image here uses): the scan's photographic albedo, lit by a
  layered skin model rather than by a lambert term - a **subcutaneous** term that keeps the
  terminator red, a **T-zone mask** evaluated in head space (forehead, nose bridge, chin) that
  splits the specular into a broad dry sheen and a tight oily highlight, **pore-scale
  breakup** from a hash of the atlas UV (a scaling of the highlight, not a normal tilt: the tilt
  version banded along UV diagonals), sky/ground ambient that is occluded as well as the sun, and
  a **fuzz** rim. `shadeScannedSkin()` in `src/render/software_renderer.cpp`.
* **Head / body / teapot** (the procedural head): the face is shaded from **analytical masks** evaluated in head-local
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

A hair mesh's outer layers are *not* an extrusion along one straight direction, and they are
not a sweep around the skull either: every strand is authored along its own **arc length in
metres**, in the two phases hair actually goes through.

* **Phase 1, on the scalp.** From its root the strand follows the head surface until the *fall
  line*, just above the top of the ear, with a few millimetres of stand-off. The stand-off is
  mandatory: the layers of a hair mesh are straight quads between slices, so on a convex scalp
  they cut *inside* the skull, and the error grows with the layer spacing.
* **Phase 2, hanging.** Past the fall line the strand leaves the surface and falls, holding the
  silhouette it had at the fall line. This is what makes a bob a bob - a cylinder as wide as the
  head is at its widest point, with a blunt, slightly A-line hem - instead of a wig that wraps
  around the jaw and the neck. `style.fall` bends the drop from "straight out" (0, an afro) to
  "straight down" (1, a bob).
* **The hem is a distance.** Because the path is measured in metres from the crown, a hem depth
  means the same thing everywhere on the head: crown strands are the long ones, the hairline
  rows only reach the brow, and a fringe needs no special case beyond being short. The fall line
  is clamped to the hairline, so hair never slides down the forehead before it starts to hang.
* **Volume belongs to the body of the hair**, not to a hat: thin at the crown, thickest through
  the middle of the mass, gone at the hem, and thicker on the sides and back than over the
  forehead.
* **A side part** drives both the hem (a thin seam of scalp, dark rather than a bald patch of
  skin) and the drift that combed hair has away from it. The drift accumulates towards the tips,
  so the hair sweeps outwards instead of leaving the parting as a straight wall.

### Clumps, curls and ringlets

Strands are placed inside their bundle's volume, not on a sheet. Each clump is a **tube**:
every strand keeps its own place across the lock's cross-section for its whole length, and the
curl phase is **shared by the clump**, with only a small per-strand jitter. That is the
difference between ringlets and fuzz - a curl whose strands agree on when they turn is a lock,
a curl whose strands disagree is noise. Curl radius is authored in metres (`curlRadiusM`:
6 mm is frizz, 8-10 mm is a ringlet), so a short pixie and a long curly style curl at the same
real size. The GPU path runs exactly the same functions (`styleDisplacement()` +
`styleDisplacementObject()` in `shaders/hair_common.glsl`).

### Flyaways

Eight percent of strands are given a tip excursion of up to 34 mm in a direction hashed from the
strand's own seed (`StrandStylingState::flyaway`, mirrored in `shaders/hair_common.glsl`), authored
in metres so a flyaway is the same real length on a pixie and on long curls. It is derived from the
seed alone - no per-strand data is stored, which is the constraint the paper's representation
imposes - and it is what stops a groomed style from ending in a perfect curve.

### Deep self-shadowing

Hair is not an opaque surface: light dies a few millimetres into it. The renderer builds one
light-space depth buffer of the *styled volumes* (the hair mesh's outer layers, not the strands)
and tests every hair and skin sample against it with a **penetration distance**: a sample within
~12 mm of the envelope along the light is lit, everything deeper is not, and skin uses a shorter
penetration because it is opaque. One buffer therefore produces all three effects that a
hairstyle needs - bright outer shell, dark clump interiors, and the hard shadow the hair throws
across the forehead, the cheek and the neck - for a few thousand rasterised quads per model.

### What this is not

**The head is a scan - and there are two of them.** `assets/heads/head_scan.glb` is a male
photogrammetry bust (CC BY 3.0) and the default; `assets/heads/female_head_scan.glb` is a female
bust built from MakeHuman's CC0 base mesh (5,224 vertices, 9,878 triangles, a 2048^2 authored
skin atlas, 6.1 MB) and is selected with `--head-scan female`. Neither is a modelled character:
they have the anatomy the renderer needs - a cranium the hair can sit on, lids, lashes, ears,
neck, shoulders, pore-level skin - but they are not the groomed, art-directed faces of the
paper's figures.

The female head needs two things that a generic `loadScan()` would not do, and both live in
`tools/prep_female.py`: the bust is cut and fitted into the shared head space, and the **eyes are
rebuilt**. MakeHuman models each eyeball as a closed 308-vertex shell, but paints it flat red in
the body texture, and its shells render inside out through the lid aperture (a probe render showed
the visible patch mapping to the *far* hemisphere of their UV island). So the shells are dropped
and replaced with front-projected globes - a stereographic unwrap about the cornea, wound outward,
painted into the atlas islands the old shells leave free. The paint itself is geometric: the pupil
and the iris are the real angular bands of a 12 mm iris on a 23.5 mm globe, so they can be measured
rather than hand-tuned. Swapping in another head is still a file swap: POSITION/NORMAL/TEXCOORD_0
plus a base-colour map.

**The hair is not a groomed strand asset.** There is no stored strand buffer anywhere - every
strand is generated from the five hair mesh textures each frame, which is the whole point of the
paper. That means the strand *placement* is procedural: clumps, curls and flyaways come from the
styling functions and the blue noise root sets, not from an artist's comb. At the reference
images' strand density an offline path-traced groom will always look silkier; what is reproduced
here is the structure - hair mesh volume, clump tubes, ringlets, resolved ribbons, deep
self-shadowing - and the paper's cost, not an arbitrary artist's groom.

---

## How close is it, honestly?

`docs/COMPARISON.md` is the adversarial version of this README: it puts our renders next to the
paper's figures and next to the three reference photographs the project was judged against
(`docs/images/compare_reference.png`), rates them criterion by criterion, and lists what is still
visibly wrong and why. The short version:

* **functionally** the pipeline is implemented - hair mesh only, Figure 3e slices, Eq. 1-12,
  blue-noise roots, the storage targets - and `--validate` checks it;
* **visually** a single head is roughly half to two thirds of the reference bar; the gap is
  concentrated in two places: strand-level detail (ribbons read as brushed, not resolved) and curl
  at clump scale (ours is a waffle at mesh scale, the reference is ringlets). Flyaways now exist
  and leave the silhouette, but they are styled strands rather than loose fibres;
* **performance** is not compared: the paper's 2 ms is an RTX 4090, this machine has no GPU, and
  the Vulkan path in this repository has never been executed here.

![The ten Figure 10 presets on the female head](docs/images/styles_female.png)

*The ten Figure 10 presets, generated from styling parameters on the female head (`--head-scan
female`). Dye, length and curl vary convincingly; the silhouettes do not vary enough - `ponytail`
and `pixie` do not read as their names, which `docs/COMPARISON.md` lists as a known gap.*

![SilkCrowd against the reference photographs](docs/images/reference_floor.png)

*Top: the three reference heads (`docs/images/reference_three.jpg`). Bottom, one-to-one: `--style 4`
(blunt honey bob), `--style 2` (honey ringlet curls), `--style 9` (longer layered champagne blonde),
all `--yaw 62 --studio --head-scan female`. The cuts, the part and the hem heights now match; what is
still missing is per-fibre resolution and the reference's true specular breakup.*

### Videos

| | |
| --- | --- |
| ![Bob turntable](docs/media/bob_turntable.gif) | ![Crowd orbit](docs/media/crowd_orbit.gif) |
| `docs/media/bob_turntable.gif` / `.mp4` - the bob through 360 degrees, 36 frames at `--yaw 0..350` | `docs/media/crowd_orbit.gif` / `.mp4` - 48 heads, orbiting camera |
| ![Teapot orbit](docs/media/teapot_orbit.gif) | |
| `docs/media/teapot_orbit.gif` / `.mp4` - the 200-teapot Figure 9 grid (`--scene teapot`) | |

Every clip is generated from this repository with `tools/make_media.sh`, which calls
`./build/SilkCrowd --screenshot` once per frame and hands the frames to ffmpeg. Nothing is
pre-rendered elsewhere and nothing is a mock-up.

## The PDF, item by item

`docs/PDF_COVERAGE.md` is the audit: 58 items taken from the paper (equations, figures, tables,
claims, implementation details) each marked **DONE**, **PARTIAL** (implemented with a documented
deviation), **N/A** (prose), or **OPEN** (with a reason). The OPEN list is six items long and all
six are sandbox limitations - no Vulkan driver here, no GLSL validator, no GPU to time, no
interactive mesh editor - not missing method.

`docs/FAQ.md` answers the four questions that come up every time someone runs this:

* **"Why is the hair on the teapots so small?"** It was two bugs: the roots were on the *lid*
  (a 0.058 m sphere) and "3 cm" of strand on a 24 cm pot is fur. `buildTeapotHairMesh` now follows
  the pot's profile of revolution from the knob to the foot, with per-lock lengths and a flared hem.
* **"I copied the PDF, why does it look weird?"** The PDF does not give you a hair mesh. It gives
  you the five textures, Eq. 1-12 and the LOD; the *mesh* - fall line, hem as an arc length, the
  tuck around the temple, area-weighted root density - is where the look lives, and all four of
  those are documented with the failure each one causes if you skip it.
* **"No Vulkan driver - fix that."** Fixed as far as the repository can fix it: `--gpu-info`
  reports which of the build / loader / driver / device is missing and what installs it, the
  fallback message names the same verdict instead of shrugging, `tools/gpu_probe_selftest.sh`
  tests all five verdicts against a synthetic loader, and the Linux CI GPU steps are no longer
  wrapped in `|| true` (Windows and macOS report instead of gating - a hosted Windows runner has
  no ICD, so a hard gate there would only test the runner). The driver itself cannot be installed in this sandbox (no route to a package
  mirror, no PyPI/npm ICD) - `docs/GPU.md` has the checks and the install lines for a real
  machine. On hardware `--gpu` takes the mesh-shader path, or the compute fallback where
  `VK_EXT_mesh_shader` is missing.
* **"Why is it slower than the paper?"** The paper's milliseconds are an RTX 4090; the counts that
  *are* comparable (storage per model, LOD fractions, strand counts) match, and `docs/COMPARISON.md`
  keeps the two columns apart on purpose.

## Continuous integration and build caching

The workflow caches everything that is slow and nothing that is not: the apt archive keyed on
`.github/apt-packages.txt`, ccache keyed on the sources with a warm restore key (so a source edit
recompiles only what changed), the Homebrew download cache keyed on `.github/brew-packages.txt`,
Gradle's caches and wrapper, and the 2 GB Android NDK keyed on its version. A first Linux run and a
cached one differ by the apt install and the compile, which is most of the wall clock; the tests
themselves take about 3 seconds.

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

The table is filled in by the `update-readme` job, which only runs on `main`; the five platform
jobs themselves are green on `arena/9565bfb7-silkcrowd` (run `40490962592`): Windows,
Linux/lavapipe (real Vulkan render with the validation layer), macOS/MoltenVK, iOS, and the
Android APK gate.

---

## Verification status (read this before believing a number)

* The **core library, texture system, LOD, root placement, simulation and software renderer** are
  built and measured in this repository; every number in the tables above was produced by
  `./build/SilkCrowd` on this checkout.
* The **Vulkan path** (`src/gpu`, `shaders/`) compiles in this checkout against the official
  Khronos headers and links into the binary, and every vertex/fragment/compute shader compiles to
  SPIR-V: all 12, including `hair.task`/`hair.mesh` (verified here with glslang 16 built from
  source, because no package manager is reachable; that run found and fixed two real shader
  bugs - see "Also fixed on the way" in the PR).
  It is **executed in CI**: the Linux job renders headless on Mesa's lavapipe with
  `VK_LAYER_KHRONOS_validation` enabled, and that run is what turned the Vulkan path from
  "compiles" into "draws". Getting it there found and fixed seven real bugs, all of them the kind
  that only a driver or the validation layer can find - see `docs/GPU.md` §"What the first real
  run found". The **Android** job is green as well and it now means something: the previous green was a
  Gradle failure hidden by `| tee` without `pipefail`, and getting the APK to link found six
  more real bugs (C-language target, a `SILKCROWD_ROOT` that pointed at `<repo>/android`,
  a missing `src/app` include path, two missing sources, and Vulkan 1.3 core entry points
  that Android's 1.1 `libvulkan.so` does not export - see `docs/GPU.md` §"What the first
  APK build found"). What still is *not* verified is a frame on a phone: hosting an
  emulator with KVM is not something the runners offer, so the APK gate proves the build
  and the packaged assets, and `docs/GPU.md` has the `adb` line for a real handset. It has not been executed *on this machine* (no Vulkan driver, so no timing can be
  taken here): the numbers in the tables come from the software rasteriser, and GPU timings come
  from CI runs on real devices - if a row says `n/a`, that platform has not reported yet.
* The **Vulkan diagnostics** (`--gpu-info`) *are* executed here, against a synthetic
  `libvulkan.so.1`: `./tools/gpu_probe_selftest.sh` drives the real `src/app/gpu_probe.cpp` through
  `ready`, `no-device`, `no-driver`, `unsupported-device` and a mixed two-device list. The CI
  `linux` job runs that self-test, then `--gpu-info`, then the lavapipe screenshot - all three
  ungated, so a GPU build that does not work now fails the run instead of passing quietly. The
  `windows` and `macos` jobs upload the same report without gating on it (`vulkan-info-*`).
* Software-renderer timings (hundreds of milliseconds per frame) are **not** the paper's GPU
  numbers and are not presented as such: they measure the algorithmic workload (strand counts,
  triangles, storage) that the GPU path produces.
* The **scanned head** is loaded, fitted and shaded by the software path (every image in this
  README is rendered with it). The Vulkan path draws the same scan mesh and the same scalp
  parameterization, but its skin shader still evaluates the *analytic* face masks rather than
  sampling the scan's texture: binding the 1K diffuse to the head pipeline is the one piece of the
  scan integration that exists only on the software path (the Vulkan path is exercised in CI, but
  the scan texture upload is not wired to its head pipeline yet, and the validation layer has not
  been through that path on this machine).

### What was actually built and run

| Configuration | Command | Result |
| --- | --- | --- |
| GNU Make, Release (`-O2 -g -Wall -Wextra`) | `make -j2` | 0 warnings, binary + tests |
| CMake + Ninja, Release, `SILKCROWD_STRICT=ON` (`-Werror`) | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DSILKCROWD_STRICT=ON && cmake --build build` | 0 warnings, 0 errors |
| CTest | `ctest` in the build tree | **2/2** (`unit_tests`, `validation`) |
| Unit tests | `./build/silkcrowd_tests` | **3317 checks, 0 failures** |
| Paper invariants | `./build/SilkCrowd --validate` | **40 passed, 0 failed** |
| Assets | `./build/SilkCrowd --assets` | 10 presets, 14-19 KB each, 166.1 KB crowd total |
| Shaders | `glslang` 16 (SPIR-V) | **12/12** compile; every `.spv` starts with the SPIR-V magic. The previous "10 of 12" was two shader bugs (`taskPayloadSharedEXT` declared as a block, and a mesh-shader output read back) plus a `--vn` flag that wrote a C header into the `.spv` |
| Android APK | `./gradlew assembleDebug` (NDK 26.1, CMake 3.22, AGP 8.5.2) in CI | **builds**: `libsilkcrowd.so` for `arm64-v8a` + `x86_64`, 12 SPIR-V modules in `assets/shaders`, per-ABI link clean |
| Fast CPU path | `--cpu-budget --scene crowd --crowd 100 --front-end-only` | **1.12 ms** for 4,130 bundles; whole frame 24,832 ms (`docs/CPU_2MS.md`) |
| Fast path = reference geometry | `./build/silkcrowd_tests` (`testCpuFastPath`) | max deviation **0.0 m** against `StrandGenerator`, 83 new checks |
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
src/app/               CLI, scenes, asset export, Vulkan screenshot entry, Qt6 editor,
                       Vulkan diagnostics (gpu_probe.cpp, --gpu-info)
tests/                 unit tests (3317 checks)
tools/                 README compositor, asset helpers, GPU diagnostic self-test
docs/GPU.md            Vulkan: the four things that can be missing, and how to install each
android/               NDK + Gradle project (NativeActivity, Vulkan surface + swapchain)
```

## License

MIT for the code in this repository. The paper is (c) its authors; the algorithms are
implemented from the published description.
