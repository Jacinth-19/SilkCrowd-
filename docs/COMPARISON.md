# SilkCrowd vs. the paper, and vs. the reference photographs

This is a deliberately unflattering document. It exists because the interesting question is not
"does it run" but "how close is it, and where exactly does it fall short". Every number below was
produced by this checkout; every rating is subjective but justified in the row next to it.

Three things are compared:

1. **Functional parity** with Bhokare, Montalvo, Diaz and Yuksel, *Real-Time Hair Rendering with
   Hair Meshes* (SIGGRAPH 2024) — is each piece of the algorithm actually there, and does it behave
   the way the paper says?
2. **Visual parity** with the paper's own figures and with the three reference photographs
   supplied for this project (a wireframed blonde bob, ash ringlet curls, and a honey-blonde blunt
   bob on a shoulder bust, all shot in profile on black).
3. **Cost**, where the two are on the same footing at all — which they mostly are not, because the
   paper's numbers are RTX 4090 numbers and this machine has no GPU.

## 1. How the images were made

```bash
# the reference framing: profile view, dark studio backdrop, no grid
./build/SilkCrowd --screenshot bob.png --scene head --style 4 --strands 40000 \
    --width 420 --height 480 --ssaa 2 --yaw 78 --studio \
    --head-scan assets/heads/female_head_scan.glb

# the ten Figure 10 presets on the female head
for s in 0 1 2 3 4 5 6 7 8 9; do
  ./build/SilkCrowd --screenshot style_$s.png --scene head --style $s --strands 26000 \
      --width 340 --height 400 --ssaa 2 --head-scan assets/heads/female_head_scan.glb
done
```

`--yaw` and `--studio` were added for this comparison: the reference photographs are shot from
the side against black, and comparing a three-quarter view on a bright grid to a profile on black
compares the backdrop, not the hair.

The side-by-side sheet is `docs/images/compare_reference.png`; the ten-style sheet is
`docs/images/styles_female.png`.

**Caveats that materially affect the judgement**

* Every image here is the **software rasteriser** (no Vulkan driver in this environment). It has no
  MSAA and no TAA, so its edges are `--ssaa 2` supersampled and nothing more. The paper's images
  are 8x MSAA + post-processing on a 4090.
* The reference photographs are **offline renders with resolved individual fibres**, i.e. they are
  closer to the paper's *quality target* than to what any real-time renderer produces. They are a
  good standard to fail against.
* Our head is a **MakeHuman CC0 bust**, not the model the references use. Face shape, skin tone and
  neck length differ, and that difference is visible in the sheet and is not a hair-rendering
  result.

## 2. Functional parity with the paper

| Paper element | Paper | SilkCrowd | Verdict |
| --- | --- | --- | --- |
| Hair mesh as the only hair representation (§3.1) | hair mesh + textures; no strand list is stored | `HairMesh` + `hair_mesh_texture.h`; no strand storage anywhere in the renderer | **parity** |
| Figure 3e's three intermediate slices | root, two intermediates, tip | `HairMesh::sliceCount` reproduces it; `--validate` checks slice indices, spacing and the root copy | **parity** |
| Eq. 1-4, per-vertex frames | frames from the mesh, not finite differences | `hairTextureFrames()`; `--finite-differences` renders Figure 4a's seam on demand to show the difference | **parity** |
| Eq. 5-6, strand interpolation | within a quad, bilinear in the local frame | `generateStrands()` | **parity** |
| Eq. 7-8, strand LOD | strand count from screen-space density | reproduced; 39,780 defined -> 14,488 rendered (36.4 %) | **parity** |
| Eq. 9-12, vertex LOD | vertex count from the same criterion | reproduced; `--no-lod` for Figure 8 comparisons | **parity** |
| §3.2 parallelism | generation and shading distributed over the mesh | compute + task/mesh shaders in `shaders/hair_compute.comp`, `hair.task`, `hair.mesh`, CPU jobs in the software path | **parity in code, unexecuted on a GPU here** |
| §3.3 blue-noise root placement | stratified, blue-noise-ish roots | `root_placement` module + tests | **parity** |
| §3.5 self-shadowing | deep shadowing of hair and skin | one light-space envelope buffer with a penetration distance (§"Deep self-shadowing" in the README); no strand BVH | **partial** - the paper's version can shadow through individual fibres; ours cannot see through the envelope, so a thick clump shadows like a solid |
| Figure 10 styling presets | ten female hairstyles from styling parameters | ten presets, all from `StylingParams` | **partial** - see §4 |
| Figure 1 storage target | 13-21 KB per model, 1.7 MB for 100 | 14-19 KB per model, 166.1 KB for ten crowd presets | **parity** |
| Figure 1 timing | 2 ms for 100 characters x 100k strands | 14.9 s software; Vulkan path never executed here | **not compared** |
| Figure 9 (200 teapots) | 1.8 ms (8x MSAA) / 1.1 ms, 3.2 MB | 12.4 s software, 1.53 MB textures | **not compared** |
| No-LOD workload (§4, Figure 9) | 47 ms / 150 ms | 1,025,711 of 10,000,000 strands survive LOD (10.3 %) | **workload parity** |

The honest summary: **the algorithm is implemented, the numbers that depend on a GPU are not
demonstrated.** Nothing in the tables above is a GPU result.

## 3. Visual comparison

Ratings are 0-10 against the reference photograph for that row. "Ours" is the best of the ten
presets for that criterion, not the average — the question is what the renderer *can* do.

| Criterion | Reference | Ours | Why |
| --- | --- | --- | --- |
| Bob silhouette, volume, hem | 9 | 6 | The hem is blunt and level and the wall is cylindrical, but ours is a smooth helmet: the reference tapers in under the jaw and breaks at the ear. |
| Individual strand resolution | 9 | 4 | The reference shows every fibre and its shadow on its neighbour. Ours reads as brushed ribbons; at 40k strands the ribbon strips are visible as a faint cross-hatch, especially on the curly preset. |
| Curl / ringlet clumping | 9 | 3 | The reference is defined *clumps* with a spiral and a shadowed groove between them. Ours is a coarse waffle at roughly the mesh's quad size — the curl is applied per strand layer, not per clump. |
| Specular glints, angel ring | 9 | 5 | The Marschner lobes are evaluated per fibre so the ring exists and moves with the light, but with ribbon-wide normals it is one broad band instead of a sparkle field. |
| Dye and colour variation | 8 | 7 | Root-to-tip dye, per-strand value jitter and a lightened-flyaway term are all in both renderers. This is the closest row. |
| Ends and flyaways | 8 | 4 | 8 % of strands now get a tip excursion of up to 34 mm in a seeded direction (`StrandStylingState::flyaway`), which is enough to leave the hair mesh's own volume and break the silhouette - visible at the hem and the crown in `docs/images/compare_reference.png`. The reference still has more of them, finer, and they catch the light differently because they are their own fibres rather than styled mesh strands. |
| Skin under the hair | 8 | 6 | The scan's own albedo plus the new subsurface/T-zone/pore model (below) reads as skin at portrait distance; it still goes slightly waxy where the T-zone mask peaks on the nose. |
| Eyes | 8 | 6 | Rebuilt globes with a painted iris, limbal ring and pupil (`docs` note in `tools/prep_female.py`). They read correctly from the front and from profile, but there is no tear line, no wet meniscus and no catchlight geometry in the scan path. |
| Crowd of 100 | 9 | 5 | `docs/images/crowd_100_linux.png` is the paper's Figure 1 shot: 100 heads, 31.4 M triangles, 1.61 MB of textures. At that distance the styles read as silhouettes, and our silhouettes are all bob-like. |

**Overall, in one line:** functionally this is a faithful implementation of the paper's pipeline
(the geometry, the LOD, the textures, the storage); visually it is roughly **half to two thirds of
the reference bar** for a single head, and the gap is concentrated in three places — strand-level
detail, clump-level curl, and flyaways.

## 4. What is still visibly wrong, ranked by how much it costs the image

1. **Curls are mesh-scale, not clump-scale.** `StylingParams::curl` bends each strand by a sinusoid
   along its length. A ringlet is a *clump* that coils as a unit with a groove between clumps; that
   needs a clump partition (which strand belongs to which ringlet) and a shared spiral per clump.
   The clump machinery exists (`numLayers`, per-layer offsets) but the ringlet preset does not use
   it as a clump. This is the single largest visual gap.
2. **Flyaways exist but are coarse.** 8 % of strands are pushed up to 34 mm sideways at the tip,
   which does break the silhouette; they are still generated *inside* the hair mesh and shaded with
   its frames, so they bend with the style and cannot be much finer than a styled strand. A proper
   set of loose fibres would be generated outside the mesh and shaded on their own — the paper does
   not forbid it (it is a detail decision, not a representation one) but it is not a one-line
   change either.
3. **Ribbon visible as cross-hatch.** At 40k strands the strands are narrower than a pixel pair, so
   the alpha-blended ribbon shows its strip pattern. The cure is more strands (compute budget) or
   a better overdraw/coverage model (Q-Brian/alpha-to-coverage style) rather than more geometry.
4. **Ten presets, four silhouettes.** `long_straight`, `wavy_medium`, `curly_short`, `afro`, `bob`,
   `ponytail`, `beach_waves`, `curly_long`, `pixie`, `platinum_bob` differ convincingly in dye,
   length and curl, but `ponytail` and `pixie` do not read as their names: a ponytail needs a
   gathered mass behind the head, and a pixie needs the hairline and the volume profile changed,
   not just a shorter hem.
5. **Hairline.** The anatomical hairline is one curve for every head (63 degrees from the crown at
   the front, 116 at the nape), and every preset now extends its *front* with
   `HairStyleSpec::hairlineDrop` (4-12 degrees) so a fringe actually covers the forehead. That is
   the right place for it - a global hairline change gave the male scan a fringe over its eyes -
   but it is still one number per preset rather than a real hairline shape.
6. **Vulkan head shader does not sample the scan texture.** The GPU path still evaluates the
   analytic face masks for the head. The software path (every image in this repository) is the one
   that samples the 2048^2 scan atlas. This is the one known functional gap in the scan
   integration and it cannot be closed here: the machine has no Vulkan driver, so the descriptor
   bindings could not be validated.
7. **The eyes have no tear line or meniscus**, and the scan's lashes are painted, not geometric.

## 5. Running this on an integrated GPU

The question was: what is the best path-tracing / ray-tracing approach on an IGPU. The measured
answer from this repository is *don't trace anything, and make the raster path cheap*.

### What the raster path costs, measured

| Scene | Strands defined | Strands after LOD | Triangles | Textures | Software frame (480x300, ssaa 1) |
| --- | --- | --- | --- | --- | --- |
| 1 head, bob | 39,780 | 14,488 (36.4 %) | 927,232 | 0.18 MB | 839 ms |
| 100 heads | 9,998,570 | 754,433 (7.5 %) | 31,982,960 | 1.61 MB | 14.9 s |
| 200 teapots | 10,000,000 | 1,025,711 (10.3 %) | 26,262,960 | 1.53 MB | 12.4 s |

The software rasteriser is ~1-2 % of a modern IGPU's fill and geometry throughput, so these are
workload numbers, not predictions. Scale them: an Iris Xe (96 EU) or a Radeon 780M has roughly
1.5-3 TFLOP/s and 80-120 GB/s, which is about 1/20th of an RTX 4090's geometry rate and 1/8th of
its bandwidth, so the paper's 2 ms crowd is *not* reachable — but 30-60 ms is, if the triangle
count is kept to 8-16 M instead of 31 M (the crowd figure above is with a deliberately generous
per-head strand count).

### Why not ray tracing

* The workload is ~10 M hair segments for 100 heads. Even one shadow ray per shaded sample is
  ~10 M rays/frame; an IGPU with no RT units does ~30-100 M incoherent rays/s in the best case, so
  that alone is 100-300 ms — before any shading, and before a BVH built from 10 M segments
  (100 ms+ per frame if it moves).
* Hair is the *worst* case for ray coherence: rays that leave a ribbon immediately diverge, so the
  effective ray rate is a fraction of the peak. This is why offline hair renderers are farms.
* The raster path already gives the thing ray tracing is usually bought for here — correct
  ordering — via the depth pre-pass plus back-to-front composite (or WBOIT on the GPU path), and
  the envelope shadow buffer gives the deep shadowing at a few thousand quads.

### What to do instead, in order

1. **Keep Figure 8's LOD aggressive on IGPU.** It is the whole point of the paper: 31 M -> 8 M
   triangles is a 4x win for free. Raise the screen-space strand threshold on weak devices and let
   the dual-scattering term cover the missing strands (it already does on the software path).
2. **Prefer the compute path's generated ribbons over the mesh-shader path on IGPU.** Task/mesh
   shaders need a mesh-shader-capable driver (Adreno, Arc, RDNA3+); on UHD/Xe/Mali the compute
   fallback in `src/gpu` is the one that runs, and it is already implemented.
3. **Render hair at half resolution and composite.** Hair is high-frequency but low-contrast; the
   head and background stay at full res. This is a 2-4x fill win on an IGPU, which is
   bandwidth-bound, not ALU-bound.
4. **Sort-free transparency via WBOIT**, not per-fragment sorting: the ordering cost of a
   back-to-front composite over ~10 layers is what kills an IGPU.
5. **If some ray tracing is wanted**, spend it on *shadows only*, at 1/4 resolution, with a
   two-level BVH (clumps, then strands), one ray per 4x4 pixel block and a denoiser. That is the
   only tracing shape that fits in a 16 ms frame; full path tracing of hair does not fit at any
   resolution an IGPU can run.

## 6. Assets

* **Female head**: `assets/heads/female_head_scan.glb`, MakeHuman/MPFB base mesh, **CC0** — see
  `assets/heads/LICENSE-female_head.txt`. 5,224 vertices, 9,878 triangles, 2048^2 authored atlas,
  6.1 MB. Built by `tools/prep_female.py`.
* **Male head**: `assets/heads/head_scan.glb`, CC BY 3.0 photogrammetry scan.
* **free3d.com**: not usable from this environment. The sandbox can only reach GitHub and the
  package registries, so no file can be fetched from it, and free3d's terms for its free models are
  per-model and often non-redistributable. The MakeHuman head is CC0, which is why it was chosen
  for a repository that ships assets.
