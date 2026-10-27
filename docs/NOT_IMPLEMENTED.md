# What is not implemented, and why

This is the honest list the team asked for: **everything the paper does that this repository
does not do, cannot verify, or does differently.** It complements the audit in
`docs/PDF_COVERAGE.md` (which is organised by paper section and includes the items that *are*
done) and the CPU analysis in `docs/CPU_2MS.md`.

Legend: **[H]** blocked by hardware available here · **[S]** out of scope for a renderer ·
**[D]** implemented differently on purpose, deviation documented · **[T]** not done yet, with
the work it needs.

Nothing in this list is hidden in the code: where a deviation exists it is a named parameter, a
printed warning or a flag.

---

## 1. Blocked by the hardware in this environment

| # | What the paper has | What this repository has | Why | What it would take |
| --- | --- | --- | --- | --- |
| 1.1 **[H]** | Frames rendered on an RTX 4090 | A Vulkan 1.3 path that is executed in CI on Mesa **lavapipe** (CPU Vulkan 1.4) with the Khronos validation layer | this sandbox has no GPU and no Vulkan driver; the wrapper answers `--gpu-info` with which piece is missing | run `SilkCrowd --gpu` on any machine with a Vulkan 1.3 driver |
| 1.2 **[H]** | **2 ms** for 100 characters, 1.8 ms for 200 teapots, 47 ms without LOD | the same *workloads*, measured on the CPU: 25.5 s, and the LOD-vs-no-LOD ratio at crowd distance | no GPU; see `docs/CPU_2MS.md` for the full analysis. Per parallel lane the CPU is within 8× of the paper's GPU, so the gap is lane count (8,192×) | the reference hardware, or ~1,000 cores |
| 1.3 **[H]** | 8× MSAA | MSAA is implemented (`setSampleCount`, resolve to `RGBA16F`/`D32`) and *clamped to what the device reports*; lavapipe exercises the code, but the visual result is verified only on CI | needs a real 8×-capable device to look at | a GPU run, then compare against `docs/images/` |
| 1.4 **[H]** | A frame on a phone (Adreno, the paper's compute fallback path) | the APK is built and checked in CI (both ABIs + 12 SPIR-V modules) but never executed | hosted runners cannot boot an emulator without KVM; no device here | `adb install` + the command in `docs/GPU.md` |
| 1.5 **[H]** | GPU memory bandwidth / cache behaviour of the texture layout | the layout is validated numerically; its *hardware* cost (texture-unit throughput, cache residency of 13–21 KB per model) is not measured | needs hardware counters | Nsight/RGP capture |
| 1.6 **[H]** | Qt6 editor screenshots | `src/app/qt_app.cpp` exists, is compiled by `cmake -DSILKCROWD_QT=ON`, but **has never been compiled or run here** (no Qt6 in the image); it was audited by hand and two API bugs were fixed | no Qt6 package reachable | build once with Qt6 and click through it |

## 2. Not implemented (paper has it, this repository does not)

| # | Paper item | State here | Why | What it needs |
| --- | --- | --- | --- | --- |
| 2.1 **[S]** | Hand-authored hair meshes on hand-modelled characters (Figures 1, 5, 9, 10) | meshes are **generated** from a scalp parameterisation (`buildHairMesh`), then styled | authoring 100 groomed meshes by hand is an artist task, not a code task; the paper's own Limitations section says the mesh quality dominates the result | a hair-mesh authoring tool and artists |
| 2.2 **[S]** | The interactive hair-mesh *editing* workflow of Figure 5 (layer handles, silhouette sculpting) | the Qt app orbits, switches style, changes LOD and simulates; it does not edit layer vertices | this is a renderer, not a modelling package | a layer-handle editor in the Qt app |
| 2.3 **[T]** | Figure 10's ten silhouettes, hand-authored | ten procedural presets exist; **ponytail and pixie are the two weakest** — a gathered mass and a shorter hairline are mesh changes, not styling parameters | a parameter cannot gather hair into a tail | two more mesh templates |
| 2.4 **[D]** | Per-strand hair self-shadowing (shadow rays / deep opacity maps) | **one light-space envelope depth buffer** with a penetration distance, plus a crown term; bright shell, dark interior and shadowed skin all come from it | per-strand shadow rays at crowd scale are not affordable, and the paper does not specify its own shadow solution in detail | deep opacity maps or per-strand VSSM; 2–3× the shadow-pass cost |
| 2.5 **[D]** | The paper's shading model as published | `hair.frag` has a full longitudinal/azimuthal Marschner BCSDF with Fresnel; the **software path uses Kajiya–Kay**, and the **fast CPU path uses a one-lobe approximation** | the BCSDF is ~5× the cost per vertex in scalar code and the software path is already 600 ms | nothing — it is a deliberate cost/quality split, documented per path |
| 2.6 **[D]** | Multiple scattering between strands | dual-scattering approximation in `hair.frag`; the software path uses a penetration-depth ambient floor | a true multiple-scattering solution is research-grade | none planned |
| 2.7 **[D]** | Tighter texture packing (the paper's 200 teapots fit in 3.2 MB) | **30.2 MB** for the same 200 teapots: every bundle gets a full 2×2 block on every slice | fixed-size blocks are what makes the texel addressing branch-free; a size-class layout is the fix | a variable-size block layout + a per-bundle size class |
| 2.8 **[D]** | The printed triangle→block mapping of Section 3.3 | default is the *barycentric* mapping; the printed expression is available as `TriangleMapping::PaperLiteral` and both are measured in `--validate` (the printed one biases roots toward the duplicated corner) | the printed expression does not reproduce barycentric interpolation; see `docs/PAPER_MAPPING.md` | nothing — it is a parameter |
| 2.9 **[T]** | Temporal LOD stabilisation beyond δᵢ and γ | Eq. 8's per-bundle δ and Eq. 12's γ are implemented; there is **no extra hysteresis state** | the paper describes exactly these two mechanisms; TAA absorbs the residual | optional: a per-bundle hysteresis term |
| 2.10 **[T]** | The progressive blue-noise sets as a precomputed asset | generated at run time by sample elimination, once per process (**~300 ms** for both sets: 1024 samples, 4× oversample) | it is training-time work that a shipped app would bake into an asset | write the sets to `assets/` and load them |
| 2.11 **[T]** | SIMD / bundle-coherent CPU evaluation | the fast CPU path is **scalar**; the single biggest remaining win (§11 of `docs/CPU_2MS.md`) is evaluating 8–16 strands per pass over the *same* 2×2 texel block | time; it is a substantial kernel rewrite | a 16-wide kernel per bundle, with the taps loaded once |
| 2.12 **[T]** | GPU counterpart of the fast path | `--cpu-budget` is CPU only; there is no `--gpu-budget` profiling split of the Vulkan path | the Vulkan path can only be timed in CI, and only as a whole frame | timestamp queries per pass on real hardware |
| 2.13 **[T]** | Mesh-shader workgroup tuning against the paper's numbers | the local size is chosen so that one lane writes one ribbon vertex (32 vertices per workgroup); it is **not** tuned against measured hardware throughput | no device to tune on | a GPU + a sweep of `local_size_x` |
| 2.14 **[D]** | The scanned head's texture in the Vulkan path | the software path samples the scan's diffuse; the Vulkan head pipeline still evaluates analytic face masks | the scan upload is the one piece of the asset pipeline that exists only on the software path | bind the 1K diffuse to the head pipeline (a day's work, needs a GPU to verify) |
| 2.15 **[S]** | Animated characters | characters are static transforms; the *hair mesh* deforms (XPBD simulation, `--simulate`), and everything else follows from that | character rigging is out of scope | a rig + skinning, which the hair mesh would then follow |
| 2.16 **[D]** | The paper's simulation (it uses, but does not contribute, one) | XPBD hair-mesh simulation with sag-free rest state and wind | not the paper's contribution; ours is an extension and is not validated against the authors' | none |
| 2.17 **[S]** | The paper's evaluation/ablation figures (their comparison with strand-based rendering, their artist study) | not reproduced: no competitor implementation, no artists | would need the strands-based baseline built too | a second renderer |

## 3. Quality gaps in the imagery (by decision, not by accident)

The team's instruction was explicit: **a decent AA-level look, not hyper-realism**, because of
the sandbox. The images are what that costs:

| # | Gap | State |
| --- | --- | --- |
| 3.1 | The head is a **photogrammetry scan bust** (CC BY) or a MakeHuman bust (CC0), not a hand-modelled character with a groomed hairstyle | deliberate; a procedural head was rejected earlier and a scan is the best available |
| 3.2 | The three female reference styles are **styling presets that reproduce the silhouettes**, not reconstructions of the photographs | one-to-one in silhouette (bob / wavy / curly), not in every fibre |
| 3.3 | The bob that "looked like a Russian winter hat" | fixed in the previous turn: the crown was being shaded as a closed shell (see `docs/FAQ.md`); the hem taper and the crown AO were the two causes |
| 3.4 | Individual strand glints at 720p | visible in the reference path's renders, but a strand is sub-pixel at crowd distance by construction — the paper has the same property |
| 3.5 | No real-world groom photographs to compare against | the references the team supplied are the only ground truth available |

## 4. Things I deliberately did **not** do

| # | Not done | Why |
| --- | --- | --- |
| 4.1 | Look at or reverse-engineer the authors' code | the instruction was "the paper is the only source of truth" |
| 4.2 | Invent a GPU timing | every millisecond in the docs was produced by a command in the same document, on the machine named next to it |
| 4.3 | Claim parity with the paper's 2 ms | `docs/CPU_2MS.md` states the 12,700× gap and explains it; the only thing that fits in 2 ms is the front end (1.17 ms), and that is labelled as CPU-only work |
| 4.4 | Present the fast CPU path's frame as an image of the renderer | its shading and shadowing are simplified; `docs/images/` is produced by the reference path only |
| 4.5 | Tune the fast path against the reference path's *pixels* | the two differ by design; the guarantee is on the **geometry**, which is asserted bit-identical in `tests/tests.cpp` |

---

## 5. Summary

* **Implemented and validated**: the hair-mesh texture system (Table 1, Figure 3e, 2 trilinear
  + 1 lerp), Eq. 1–12, root placement by progressive blue noise, 11 styling operations, the
  mesh-shader path and its compute fallback, LOD (both types), the scan pipeline, the Qt6 app
  source, the Android APK, and CI on five platforms.
* **Not implemented**: hand-authored meshes and their editing UI, per-strand shadow rays, a
  tighter texture packing, SIMD, GPU-side phase profiling, and (unverifiable here) the paper's
  GPU timings, its 8× MSAA look and its on-device Android frame.
* **Not possible here**: 2 ms on two cores. Measured: 25.5 s for the same scene, 1.17 ms for
  the part of the frame that is CPU work.
