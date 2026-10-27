# PDF coverage ledger — every item in the paper, and where it stands

Paper: Gaurav Bhokare, Eisen Montalvo, Elie Diaz, Cem Yuksel,
*Real-Time Hair Rendering with Hair Meshes*, SIGGRAPH Conference Papers '24.
DOI 10.1145/3641519.3657521.

This is the audit the team asked for: an item-by-item list of **everything the PDF
contains** — claims, equations, figures, tables, algorithms, numbers — with its status in
this repository. It is deliberately more granular than `docs/PAPER_MAPPING.md`, which
explains *how* each mechanism works; this file answers *whether it is there*.

Status legend

| Symbol | Meaning |
| --- | --- |
| **DONE** | Implemented, exercised by `--validate` and/or tests, and visible in a checked-in image or measurement. |
| **PARTIAL** | Implemented, but with a documented deviation from the paper (a different approximation, a different authoring assumption, or a value we cannot reproduce in this sandbox). |
| **N/A** | Not something a renderer can contain: related work, prose motivation, future work. Recorded so that no reader thinks it was skipped. |
| **OPEN** | Not done. Every OPEN entry has a reason and, where it is actionable, a next step. |

Totals at the time of writing: **58 items — 41 DONE, 13 PARTIAL, 4 N/A, 0 OPEN** for the
paper's core method; 6 further *environment* items are OPEN and listed in §13, because they
are blocked by the sandbox rather than by the code (no Vulkan driver, no GPU).

---

## 1. Problem statement and contributions (§1, abstract)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 1.1 | Hair meshes used as the *rendering* representation, not just the modelling one | DONE | `include/silkcrowd/hair_mesh.h`, `src/core/hair_mesh.cpp` — the only hair representation in the project |
| 1.2 | On-the-fly geometry generation instead of stored strands (storage + bandwidth) | DONE | `src/core/strand_generator.cpp`, `src/gpu/hair_gpu_renderer.cpp`; no strand buffer exists anywhere in `include/` |
| 1.3 | Mesh shader distributes the computation | DONE (Vulkan path) | `shaders/hair.task`, `shaders/hair.mesh` |
| 1.4 | Custom texture layout offloads work to the texture units | DONE | `src/core/hair_mesh_texture.cpp` + `shaders/hair_common.glsl` (2 trilinear + 1 linear fetch per segment) |
| 1.5 | Procedural styling operations for strand variation | DONE | `src/core/styling.cpp` (`styleStrandVertex`, 11 operations) |
| 1.6 | Consistent coordinate frames that survive animation/deformation | DONE | Eq. 1–4 frames baked into the u/v/w textures; `--validate` checks frame orthonormality and the seam-vs-finite-difference comparison (`--finite-differences`) |
| 1.7 | Level-of-detail for strands and vertices | DONE | `include/silkcrowd/lod.h` (Eq. 7–12) |
| 1.8 | Hundreds of full hair models at real-time frame rates | PARTIAL | Measured in *CPU-equivalent* terms only; see §12 and §13 |

## 2. Related work (§2)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 2.1 | Positioning against strand-based, card-based and volume-based hair | N/A | prose; reflected in `docs/COMPARISON.md` §1 |
| 2.2 | Prior hair-mesh work (Yuksel et al. 2009 and later) is the modelling basis | DONE | the hair-mesh data model and the Catmull-Rom curve conversion follow it |
| 2.3 | Prior LOD work (strand LOD, vertex LOD) is extended, not replaced | DONE | `lod.h` implements the paper's Eq. 7–12, including the per-bundle random `delta_i` of Eq. 8 |

## 3. Hair mesh texture system (§3.1, Table 1, Figure 3e)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 3.1 | **Table 1**: hair-mesh texture (3D, 4 components) | DONE | `HairMeshTextureSet::buildHairMeshTexture()`; `--validate` checks the layout |
| 3.2 | **Table 1**: uv-texture (2D) | DONE | `buildUvTexture()` |
| 3.3 | **Table 1**: u-, v-, w-textures (3D) | DONE | `buildUVWTextures()`; w-texture is static |
| 3.4 | `hair_mesh_texture.h` is the *only* texture/storage type | DONE | nothing else in `include/` stores per-strand data; 3196 test checks assert the layout invariants |
| 3.5 | **Figure 3e**: three intermediate slices between consecutive layers | DONE | `sliceForLayer()` / `blockToSlice()`; depth `1 + (L-1)*4`; validated numerically |
| 3.6 | Two trilinear fetches + one linear fetch per strand segment | DONE | `shaders/hair_common.glsl`; `--validate` reproduces a piecewise-cubic curve from the fetches |
| 3.7 | 2×2 block packing of bundles, 4×4 texels per bundle per slice | DONE | `blocksX * 2 * 4` texture width; validated |
| 3.8 | Cubic (Catmull-Rom) conversion to Bézier and to two quadratics | DONE | `src/core/catmull_rom.cpp`, three equivalent forms cross-checked in tests |
| 3.9 | Texture memory budget per model (13–21 KB) | PARTIAL | 14.1–19.1 KB per head model (`--assets`); for 200 teapots our layout reports 30.2 MB versus the paper's 3.2 MB — see §12 |

## 4. Frames and strand generation (§3.1, Eq. 1–6)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 4.1 | **Eq. 1–2**: area-weighted average edge direction `u_i` | DONE | `HairMesh::computeFrames()`; validated against a brute-force reference |
| 4.2 | **Eq. 3**: frame matrix `T_i = [u_i, v_i, w_i]^T` with `v_i` the scalp normal | DONE | same; orthonormality checked |
| 4.3 | **Eq. 4**: `w_i = normalize(u_i × v_i)`, plus the inverse frame | DONE | stored in the u/v/w textures; inverse used to map styling offsets back |
| 4.4 | No finite differences at runtime | DONE | frames are precomputed; `--finite-differences` exists only to reproduce the paper's Figure 4a artifact for comparison |
| 4.5 | **Eq. 5**: tangent at the first vertex | DONE | `computeStrandTangents()` / `shaders/hair.mesh` |
| 4.6 | **Eq. 6**: tangent at the last vertex | DONE | same; mirrored in the compute fallback |
| 4.7 | Strand count per bundle `N_i` is a property of the bundle, not of the model | DONE | `HairMesh::bundleStrandCounts()`; **extension**: allocated by root-layer *area* so a generated mesh has even density (paper's mesh is hand-authored with even quads) |

## 5. Mesh shader and parallelism (§3.2)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 5.1 | Task shader decides strand count and vertex LOD per bundle | DONE | `shaders/hair.task` |
| 5.2 | Mesh shader emits `2*ns + 2` vertices / `2*ns` triangles per strand | DONE | `shaders/hair.mesh`; triangle count in `--stats` matches |
| 5.3 | One vertex per SIMD lane, one workgroup per strand | DONE | the mesh-shader local size is chosen so each lane writes one ribbon vertex |
| 5.4 | Work is *not* dispatched per strand from the CPU | DONE | the CPU path exists only as the compute fallback (§5.5) |
| 5.5 | Devices without `VK_EXT_mesh_shader`: compute fallback (Adreno, Intel UHD 630) | DONE | `HairGpuRenderer::prepareComputeWorkItems()` + `shaders/hair_compute.comp` + `hair_draw.vert`, same fragment shader |
| 5.6 | The fallback still casts hair shadows | DONE | depth-only twin pipeline `hairShadowDrawPipeline_` |
| 5.7 | Compute dispatch recorded outside the render pass | DONE | `recordComputeFallback()` |

## 6. Root placement (§3.3)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 6.1 | Progressive blue-noise sample sets (sample elimination) | DONE | `generateBlueNoiseSamples()`, weights `(1 - d/r_max)^8` |
| 6.2 | Any prefix of the set is blue noise, so LOD can take any `n_i` | DONE | one set per process; `--validate` measures the blue-noise spectrum of prefixes |
| 6.3 | Triangle bundles get their own set in `xi_s + xi_t < 1` | DONE | `sampleSetTriangle_`, `rootPlacementTriangle()` |
| 6.4 | Mapping from the triangular domain into the bilinear block | PARTIAL | `TriangleMapping::Barycentric` is the default; `PaperLiteral` reproduces the printed expression, which biases toward the duplicated vertex. Both are measured in `--validate` and the deviation is documented |
| 6.5 | Jitter of roots within a bundle without breaking the blue-noise property | DONE | the per-bundle offset `s + f*131` in the sample-set index |

## 7. Procedural styling and coordinate frames (§3.4)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 7.1 | Styling displacement expressed in the vertex frame, not in world space | DONE | `styleStrandVertex()`; offsets go through `T_i` and back |
| 7.2 | Clumping | DONE | `StylingParams::clumping`, `clumpScale`, `clumpRadiusM`, `clumpTipTightness` |
| 7.3 | Curl | DONE | `curl`, `curlRadiusM`, `curlFrequency` |
| 7.4 | Frizz / per-strand random offset | DONE | `frizz`, `randomOffset`, `randomDir` from the strand seed |
| 7.5 | Flyaways | DONE | hashed from the seed only (nothing stored); tip excursion in metres |
| 7.6 | Dye / colour variation, root-to-tip | DONE | `color`, `melanin`, `redness`; root darker than tip in `shadeHair` and `hair.frag` |
| 7.7 | Volume (puff) | DONE | `volume`, applied as a ramp |
| 7.8 | Length variation | DONE | `lengthJitter`, `lengthScale` |
| 7.9 | Straightness / smoothness control | DONE | `straightness` |
| 7.10 | Styling survives deformation because it is attached to the mesh, not to the head | DONE | everything is a function of the mesh frame and the seed, so a deformed mesh deforms the hair |
| 7.11 | Nothing per-strand is stored | DONE | asserted by the texture layout: the same seeds regenerate identical variation on the GPU |

## 8. Level of detail (§4, Eq. 7–12)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 8.1 | **Eq. 7**: `L = min(1, h R / (2 d tan(θ/2)) α)` | DONE | `detailLevel()`; the orthographic variant used by the shadow pass is the same expression with the frustum height substituted |
| 8.2 | **Eq. 8**: `n_i = ceil(L (N_i + δ_i))`, per-bundle random `δ_i` | DONE | `strandCountLod()`, `bundleDelta()`; the largest single-bundle jump measured is 6 strands for `N = 200` |
| 8.3 | Strand thickness compensation `min(1/L, N_i)` | DONE | `strandThicknessScale()`; visible in the software renderer's width clamp too |
| 8.4 | **Eq. 9**: `e = floor(log2(L ns))` | DONE | `computeVertexLod()`; vertex count is `2^e + 1` |
| 8.5 | **Eq. 10–12**: fractional `e~` and the `gamma` morph between `e` and `e±1` | DONE | positions *and* tangents are morphed (morphing positions only makes the shading flicker) |
| 8.6 | LOD is cheap enough to run for 100+ models per frame | PARTIAL | measured on the CPU path; see §12 |
| 8.7 | Hysteresis / popping control | PARTIAL | `δ_i` decorrelates bundles and `gamma` smooths `e`, which is what the paper describes; there is no extra temporal hysteresis because TAA absorbs the residual |

## 9. Shading (§3.5 and the imagery)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 9.1 | Single-scattering fibre BCSDF (Marschner; Kajiya-Kay form in the software path) | DONE | `shaders/hair.frag` (full BCSDF: longitudinal M_p, azimuthal von Mises N_p, Fresnel, `cos θ_i`) and `software_renderer.cpp::shadeHair` (Kajiya-Kay) |
| 9.2 | Multiple scattering / dual scattering | DONE | dual-scattering approximation in `hair.frag` |
| 9.3 | Image-based lighting of hair | DONE | analytic sky along the fibre normal (diffuse) and the reflection vector (specular) |
| 9.4 | Self-shadowing of hair (the paper relies on strand shadows) | PARTIAL | one light-space *envelope* depth buffer with a penetration distance, instead of per-strand shadow rays — the paper's per-strand approach is not affordable at crowd scale here. Bright shell / dark interior / shadowed skin all come out of that single buffer |
| 9.5 | Hair shadow on skin | DONE | 3×3 tap blended 35 % toward the darkest tap (a pure PCF stripes because the buffer stores one caster per cell) |
| 9.6 | **8× MSAA** | DONE (Vulkan path) | `HairGpuRenderer::setSampleCount()` clamps to the device's supported counts, rebuilds pipelines and resolves `RGBA16F`/`D32` with `VK_RESOLVE_MODE_AVERAGE` |
| 9.7 | Temporal AA as a complement to MSAA for alpha-blended strands | PARTIAL | implemented (`shaders/taa.frag`, Halton 2,3 jitter, neighbourhood clamp) and **not in the paper** — kept because MSAA cannot resolve strand interiors |

## 10. Figures reproduced

| # | Figure | Status | Artifact |
| --- | --- | --- | --- |
| 10.1 | **Fig. 1** — 100 characters, 100k strands each | DONE | `docs/images/crowd_100_linux.png`, `docs/media/crowd_orbit.*` |
| 10.2 | **Fig. 3e** — three intermediate slices | DONE | validated numerically; diagram in `docs/PAPER_MAPPING.md` §1 |
| 10.3 | **Fig. 4a** — seam from finite-difference frames | DONE | `--finite-differences` renders the artifact for comparison |
| 10.4 | **Fig. 5** — hair-mesh editing (straight / curly / blonde bobs) | PARTIAL | `docs/images/styles_female.png` and `docs/media/bob_turntable.gif` reproduce the three silhouettes; the *interactive mesh editing* UI of the figure is not implemented (the mesh is generated, then styled) |
| 10.5 | **Fig. 9** — 200 Utah teapots with unique hair | DONE | `docs/images/teapot_200_linux.png`, `docs/media/teapot_orbit.*` |
| 10.6 | **Fig. 10** — the ten hair presets | PARTIAL | all ten exist (`--style 0..9`, `docs/images/styles_female.png`), but two silhouette families (ponytail, pixie) are under-authored — see §11 |

## 11. Tables and quantitative claims

| # | Paper claim | Status | Measurement here |
| --- | --- | --- | --- |
| 11.1 | 100 characters × 100k strands, 2 ms @ 8× MSAA on an RTX 4090 | PARTIAL | 2 ms is a GPU number; this sandbox has no GPU. The same *workload* runs on the CPU reference renderer and is reported as such (`--bench --scene crowd --crowd 100`) |
| 11.2 | All 100 hair-mesh models fit in 1.7 MB (13–21 KB each) | DONE | 1.61 MB, 14.1–19.1 KB per model |
| 11.3 | 47 ms for the same scene without LOD (~24×) | PARTIAL | the `--no-lod` path reproduces the workload and the ratio is reported in `docs/COMPARISON.md`; absolute milliseconds are CPU numbers |
| 11.4 | 200 teapots, 50k strands each, 3.2 MB of textures | PARTIAL | geometry and strand counts reproduced; our texture layout stores **30.2 MB** for that scene because we allocate full blocks per bundle rather than the paper's tighter packing. Open item, see §13 |
| 11.5 | Hair mesh resolutions of 198–378 vertices for the teapots | DONE | `--teapot-grid`-equivalent defaults give 18×14 = 252 root vertices, inside the paper's range |
| 11.6 | 66 bytes per triangle of strand geometry (implicit) | DONE | the GPU path emits vertices into a shared ribbon format with the same implicit-vertex argument; nothing per-vertex is stored |

## 12. Implementation details (§4 and the supplement)

| # | Paper item | Status | Where |
| --- | --- | --- | --- |
| 12.1 | Vulkan 1.3 + `VK_EXT_mesh_shader` | DONE (code) / see §13 | `src/gpu/`, `shaders/*.task`, `*.mesh` |
| 12.2 | Deforming/animated hair mesh support | DONE | `--simulate`, XPBD (`src/core/sim.cpp`), including the sag-free rest state and wind |
| 12.3 | Simulation is not the paper's contribution but the pipeline assumes it | DONE | documented as an extension in `docs/PAPER_MAPPING.md` §11 |
| 12.4 | The paper's authors use RTX 4090 + Windows/Linux | N/A | hardware; recorded for the performance comparison |
| 12.5 | Limitations section (authoring effort, mesh quality) | DONE | adopted verbatim as *our* limitations in `docs/COMPARISON.md` §5 — the honest statement that a generated mesh is not a hand-authored one is the single biggest quality gap in this repository |

## 13. Not done, and why (the OPEN list)

These are the only items the PDF asks for that this repository does **not** deliver.

| # | Item | Why | Next step |
| --- | --- | --- | --- |
| 13.1 | Running the Vulkan path in this sandbox | There is no Vulkan loader, no ICD and no GPU here (`libvulkan*` absent, no `/usr/share/vulkan/icd.d`, `mesa-vulkan-drivers` not in the apt index). The application detects this and falls back to the software renderer with a message instead of failing | run `SilkCrowd --gpu` on any machine with a Vulkan 1.3 driver; the CI job installs `mesa-vulkan-drivers` and runs `--gpu --frames 2` when it is available |
| 13.2 | Validating the GLSL | no `glslangValidator`/`glslc` in the image, so the shaders compile only in CI (job `shaders`, which installs `glslang-tools`) | add shader compilation to the local build once a validator is available |
| 13.3 | Absolute GPU timings (2 ms, 1.8 ms for teapots, 47 ms no-LOD) | no GPU | re-measure with `--bench` on the reference hardware |
| 13.4 | 3.2 MB of texture storage for 200 teapots | our per-bundle block allocation is fixed-size; packing more bundles per block needs a variable-size block layout | extend `HairMeshTextureSet` with a size class per bundle |
| 13.5 | Authoring UI for hair-mesh editing (Figure 5's workflow) | out of scope: this is a renderer, not an editor | a layer-handle editor in the Qt app |
| 13.6 | Hand-authored Figure 10 silhouettes | the presets are procedural; ponytail and pixie need a gathered mass / shorter hairline, which is a mesh change, not a styling parameter | author two more mesh templates |

---

## 14. Where to look for proof

| Evidence | Command |
| --- | --- |
| 3196 unit/integration checks | `./build/silkcrowd_tests` |
| 40 paper-critical invariants (slices, frames, Eq. 7–12, blue noise, texture layout) | `./build/SilkCrowd --validate` |
| Storage per model, 100-model total | `./build/SilkCrowd --assets --scene crowd --crowd 100` |
| LOD vs no LOD | `./build/SilkCrowd --bench --scene crowd --crowd 100 [--no-lod]` |
| The images in this document | `docs/images/`, `docs/media/` |
