# Paper to code mapping

Reference: Bhokare, Montalvo, Diaz, Yuksel, *Real-Time Hair Rendering with Hair Meshes*,
SIGGRAPH 2024. This document records exactly where each equation and figure appears in the
implementation, and every place where the code had to make a decision the paper leaves implicit.

Notation: `L` = number of hair mesh layers (root slice + `L-1` extrusion layers), `ns` = number
of segments of a strand, `N_i` = the number of strands in bundle `i`. "Slice" means one depth
slice of the hair mesh textures.

---

## 1. Hair mesh textures (Table 1, Section 3.1, Figure 3e)

| Texture | Kind | Dynamic | Built by |
| --- | --- | --- | --- |
| hair mesh texture | 3D, 4 components (3 root-layer values + mask) | **yes** | `HairMeshTextureSet::buildHairMeshTexture()` |
| uv-texture | 2D | **yes** | `HairMeshTextureSet::buildUvTexture()` |
| u-texture | 3D, 3 components | **yes** | `buildUVWTextures()` |
| v-texture | 3D, 3 components | **yes** | `buildUVWTextures()` |
| w-texture | 3D, 1 component | no (static) | `buildUVWTextures()` |

`include/silkcrowd/hair_mesh_texture.h` is the only texture type in the project: nothing else
stores per-strand or per-bundle data, and the GPU path samples exactly these five images
(`shaders/hair_common.glsl`, `set = 4`).

**Figure 3e layout.** Between the root slice and each extrusion layer the paper inserts three
intermediate slices, giving

```
depth = 1 + (numLayers - 1) * 4
```

and each strand segment is evaluated with *two trilinear* fetches (which need those three
intermediate slices) plus *one linear* fetch for the styling weight, exactly as the figure
shows. The interlacing of the four slices per layer lives in `sliceForLayer()` and
`blockToSlice()`; `--validate` checks the arithmetic (`1 + (L-1)*4`), the interpolation weights
and the fact that two trilinear + one linear fetch reproduce a piecewise-cubic hair curve.

**Spatial layout.** Bundles are packed into 2x2 blocks of the texture, following Figure 3e: each
bundle owns a 4x4 texel region per slice, so `blocksX * 2 * 4` is the texture width. The block
coordinates come from the quad (or triangle, Section 3.3) root parameterisation.

**Curve evaluation.** Catmull-Rom (the hair mesh curve) is converted to a cubic Bezier
`(p0, b0+, b5-, p5)`, split into two quadratics (Truong 2020), and evaluated with the intermediate
slices. `src/core/catmull_rom.cpp` implements all three forms so the validation suite can check
that they agree to machine precision.

## 2. Frames (Eq. 1-4)

```
u_i = area-weighted average of the unit edge directions at vertex i     (Eq. 1, 2)
T_i = [u_i, v_i, w_i]^T,  v_i = normal_i,  w_i = normalize(u_i x v_i)  (Eq. 3, 4)
```

`src/core/hair_mesh.cpp::computeFrames()` computes `u` by summing each incident face's edge
directions weighted by the *area* of the face, then orthonormalises to get `T_i` and its inverse.
The frame and its inverse are precomputed per vertex once and stored in the u/v/w textures, so
runtime does no finite differences (which is what produces the seam visible in Figure 4a - the
suite renders that seam through `--finite-differences` for comparison).

## 3. Strand generation (Eq. 5-6, Section 3.4)

Positions come from the texture evaluation above; tangents use the endpoint rules of Eq. 5 and 6
(`tangentFirst()` / `tangentLast()`); styling displaces the curve by `T_i * d(w)` so the
displacement follows the local frame rather than a global axis.

## 4. Root placement (Section 3.3)

Progressive blue-noise sets from sample elimination (Yuksel 2015) are generated once per process
(`generateBlueNoiseSamples`, `(1 - d/r_max)^8` weights, progressive removal order). Any prefix of
the set is blue noise, so a bundle can render any strand count (Eq. 8) with a natural root
distribution.

For triangle bundles a *separate* set lives in the triangular domain `xi_s + xi_t < 1` and the
mapping into the bilinear block is

```
xi_Delta_s = xi_s / (1 - xi_t),   xi_Delta_t = xi_t
```

Note: this is the algebraically equivalent form of the mapping that makes `|xi_Delta|` reproduce
barycentric interpolation with coordinates `(xi_s, xi_t, 1 - xi_s - xi_t)`. The literal
expression printed in the paper (`xi_s (1 - xi_t)`) is exposed as
`TriangleMapping::PaperLiteral` and measured against it in `--validate`; the published literal
form introduces a measurable bias toward the duplicated vertex, so the barycentric form is the
default (`TriangleMapping::Barycentric`). Nothing else about the section changes.

## 5. Strand LOD (Eq. 7, 8)

```
L      = min(1, h * R / (2 d * tan(theta/2)) * alpha)    (Eq. 7)
n_i    = ceil(L * (N_i + delta_i)),  delta_i in [0,1) random per bundle
radius = baseRadius * min(1/L, N_i)                      (Eq. 8)
```

`lod.h::detailLevel()`, `strandCountLod()`, `bundleDelta()`, `strandThicknessScale()`.

The denominator `2 d tan(theta/2)` is the world-space height of the view frustum at distance
`d`. It is what makes the perspective and the orthographic form of Eq. 7 (used by the shadow
pass) agree: the orthographic variant takes that same frustum height directly. Writing
`d tan(theta)` instead would double the projected size at small FOVs and go to zero at
`theta = pi/2`, where the frustum height is finite. The random
`delta_i` is what keeps the per-bundle strand counts from jumping together, which is why the
largest single-bundle change measured by `--validate` is **6 strands** (for `N = 200`) instead of
100 for a plain `ceil(L * N)`.

## 6. Vertex LOD (Eq. 9-12)

```
e   = floor(log2(L * ns))          (Eq. 9)   power-of-two-plus-one vertex counts, 2^e + 1
e~  = e (gamma), blending toward e-1 or e+1                 (Eq. 10)
gamma morphs positions and tangents between the two levels  (Eq. 11, 12)
```

`lod.h::computeVertexLod()` returns `e`, `e~`, the vertex count `2^e + 1` and the gamma weight;
`strand_generator.cpp` (CPU) and `hair.mesh` / `hair_compute.comp` (GPU) apply the morph to both
positions *and* tangents, because morphing only positions makes the shading flicker.

## 7. Parallelism (Section 3.2)

* **Task shader** (`shaders/hair.task`): one task per bundle-chunk, decides the strand count for
  the bundle (Eq. 8) and its vertex LOD (Eq. 9), emits mesh workgroups.
* **Mesh shader** (`shaders/hair.mesh`): one workgroup per strand, one vertex per ribbon vertex,
  `2*ns + 2` vertices and `2*ns` triangles per strand.
* **SIMD lanes per vertex** are implicit in the GPU: each vertex is written by its own lane.
* **Compute fallback (Option A)**: on devices without `VK_EXT_mesh_shader` (Adreno, Intel UHD
  630) the task-shader decision runs in `HairGpuRenderer::prepareComputeWorkItems()` - the CPU
  pre-pass described by the paper - and `hair_compute.comp` writes the same ribbons into storage
  buffers, which `hair_draw.vert` then rasterises with the *same* fragment shader. The ribbon
  buffers are sized for the frame's maximum `ns`, so a single uniform stride works for every
  strand.

  The dispatch itself is recorded by `recordComputeFallback()` *before* the shadow and colour
  passes: a compute dispatch may not be recorded inside a render pass instance. Both passes then
  consume the same buffers, and the shadow pass uses a depth-only twin of the vertex pipeline
  (`hairShadowDrawPipeline_`, `hair_draw.vert` + `shadow.frag`, light orthographic matrix) so the
  fallback casts hair shadows on Adreno / Intel UHD 630 just like the mesh path.

## 8. Shading and image-based lighting

The single-scattering term is the Marschner BCSDF, evaluated per fibre in `shaders/hair.frag`:

| BCSDF factor | Implementation |
| --- | --- |
| longitudinal `M_p(theta_i, theta_o)` | normalised Gaussian in `theta_o + theta_i - alpha_p` (Marschner Eq. 3); the cuticle tilt is `alpha_R = -0.10`, `alpha_TRT = +0.14`, widths `beta_R = 0.10`, `beta_TRT = 0.20` |
| azimuthal `N_p(phi)` | von Mises lobe `exp(v (cos phi - 1)) / (2 pi (1 - e^{-2v}))` on `phi_o + phi_i` (the R lobe mirrors about the normal plane); the TRT lobe is shifted by the refraction offset `delta = 0.35` and is wider (`v = 40` versus `180`) |
| Fresnel | Schlick with the hair dielectric `F0 = 0.046`, applied to `T . H` |
| geometry | `cos theta_i` (the projected area of the fibre), which is the same `sin(T, L)` the Kajiya-Kay diffuse term uses |

`theta_i` and `theta_o` are the true inclinations of the light and view directions from the plane
normal to the fibre (not a half-vector proxy), and `phi_i`/`phi_o` are measured in the `(T, u, v)`
frame of the fibre. The multiple-scattering term is the dual-scattering approximation, and the
environment is the analytic sky sampled along the fibre normal (diffuse) and along the reflection
vector (specular), which is a closed-form stand-in for a prefiltered cubemap.

### Skin, face masks and eyes (both renderers)

The head is not part of the paper, so its shading is documented here to keep the two renderers in
step. `headFaceMasks(localPos)` in `src/core/assets.cpp` inverts the head parameterisation
(`theta = atan2(x, z)`, `phi = acos(y / 0.104)`) and evaluates a band/Gaussian mask for brows,
vermilion, stomion line, nostrils, the nose dorsum and side walls, the nasolabial fold, the lip
seam, the socket, the cheek hollow, the temple and under the jaw. `software_renderer.cpp::
shadeSkin` and `shaders/head.frag::shadeSkin` implement the same model from the same constants:
wrapped diffuse with a red terminator, a two-lobe specular, a warm key / cool fill / rim rig,
0.72x AO from the masks, and a high-frequency streak pattern for the eyebrows. The draw kind is
carried in the sign of the head push constant's `w` (`-1` skin with masks, `-2` eyeball), so the
Vulkan path draws the eyes too. `shadeEye` builds the palpebral fissure analytically (an almond
`|h|/0.575`, `|v - 0.10 h side|/0.255`, exponent 1.7), the iris/limbal ring/pupil on an 11.8 mm
globe, a lash fan with per-lash length jitter on both lids, a crease and two catchlights.

## 9. Anti-aliasing

`8x MSAA` is hardware MSAA on the Vulkan path: `HairGpuRenderer::setSampleCount()` clamps the
request to `framebufferColorSampleCounts` / `framebufferDepthSampleCounts`, then rebuilds the three
main-pass pipelines (head, task/mesh hair, compute-path hair) at that sample count and creates the
multisampled `RGBA16F` colour and `D32` depth attachments. The HDR pass resolves with
`VK_RESOLVE_MODE_AVERAGE` into the single-sampled `hdrImage_`, so the shadow map, bloom chain, TAA
resolve and composite are untouched. `shaders/taa.frag` adds a temporal resolve: Halton(2,3)
sub-pixel jitter of the projection, reprojection with the previous view-projection, and a 3x3
neighbourhood clamp on the history so the moving hair does not smear. MSAA and TAA are
complementary here: MSAA resolves geometric edges but not the alpha-blended strands, TAA
converges the strands too.

## 10. Compositing and depth ordering

PCF shadow map (2048^2) and back-to-front alpha compositing against the opaque
depth buffer. A single depth-tested transparent pass keeps only the front shell, which is exactly
the artifact the paper attributes to naive hair rendering; the software renderer's `opaqueDepth_`
buffer and the GPU path's `depthWriteEnable = VK_FALSE` implement the paper's ordering instead.

## 11. Simulation (Section 3.6)

XPBD (Macklin 2016) with stretch, bending and shape-matching constraints, sag-free rest shape
initialisation, head-sphere collision with friction, and Wu & Yuksel 2016 wind forces expressed as
drag toward a moving air velocity (`a = k (v_air - v)`, not a pure acceleration, which is invisible
under gravity). `--no-sag-free`, `--wind F` and `--simulate` expose all of it; `--validate`
reports sag-free drift (`0.0005 m` versus `0.0091 m`) and the monotone wind response.

## 12. Measurements that are checked in CI

| Quantity | Paper | This repository |
| --- | --- | --- |
| per-model texture storage | 13-21 KB | 14.1-19.1 KB (`--assets`, crowd presets) |
| 100-model total | 1.7 MB | 1.61 MB (`--scene crowd --crowd 100 --bench`) |
| 200 teapots x 50k | 3.2 MB | 1.53 MB (6x5 root grid on our teapots) |
| strands rendered with LOD | depends on R and distance | 6.9 %-51.5 % (crowd at 480x300 ... hero head) |
| triangles, crowd | 100+ M | 40.3 M at 480x300, 67.4 M at 800x460 |
| no-LOD slowdown | ~24x (47 ms vs 2 ms) | workload reproduced by `--no-lod` |
