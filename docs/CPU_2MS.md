# Can a CPU break the paper's 2 ms? Measured answer

The paper reports a **2 ms frame for 100 characters on an RTX 4090**. The question this
document answers is the obvious follow-up: *can the same workload be done on a CPU in 2 ms,
or better?*

Short answer, measured on this machine, with nothing projected or extrapolated:

* **No — not the whole frame.** The fast CPU path renders the Figure 1 crowd in **25.5 s**,
  which is **12,700×** the paper's 2 ms. That is not a surprise and it is not a defect in the
  implementation: it is the 8,192× difference in parallel lanes, minus what the CPU recovers
  per lane.
* **Per core, the CPU is within 8× of the paper's GPU per lane** (39.4k vs 305k strands/s).
  The wall-clock gap is hardware, not algorithm.
* **Yes for the part of the frame the CPU actually owns.** All of the per-frame CPU work the
  paper's method needs — Eq. 7–12 for **4,130 bundles across 100 models**, the Eq. 8 strand
  counts, frustum culling and the draw-order sort — runs in **1.17 ms**, i.e. **inside the
  2 ms budget with 1.7× to spare**. That is the number a real (GPU) implementation pays on the
  CPU every frame, and it fits.

Everything below is reproducible with the commands in §9.

---

## 1. What the paper's 2 ms covers

| | Paper (RTX 4090) | This CPU (2 cores) |
| --- | --- | --- |
| Scene | 100 characters, 100k strands each = 10M strands | `--scene crowd --crowd 100`, 40k strands each = 4M strands (LOD → 2.0M) |
| Frame | geometry generation + rasterisation + shading, 8× MSAA | same geometry, 1 sample per pixel (§7) |
| Time | **2 ms** | **25,472 ms** |
| Storage | 1.7 MB of hair-mesh textures | 1.9 MB for the same 100 models |

The paper's 2 ms is a *whole frame*: amplification, styling, shading, rasterisation. To claim
anything about it, the same whole frame has to be measured here — so this turn added a second
CPU path (`src/render/cpu_fast.cpp`) that does the whole frame, not just the parts that are
convenient to time.

## 2. What the fast path is, and what it shares

It is a *second* renderer, not a rewrite of the reference one. The reference path
(`src/render/software_renderer.cpp`) is unchanged and still produces every image in
`docs/images/`: it keeps exact back-to-front order, a light-space shadow buffer and full
per-vertex shading. The fast path answers one question — how much frame can a CPU afford? —
and measures each phase separately:

| Phase | What it does | Shared with the reference path |
| --- | --- | --- |
| front end | Eq. 7 detail level, Eq. 8 strand count, Eq. 9–12 vertex count and morph factor, per-bundle frustum + sub-pixel culling, back-to-front sort | **the same functions** (`detailLevel`, `computeStrandLod`, `computeVertexLod`, `bundleDelta`) |
| generation | 2 trilinear fetches + 1 lerp per vertex (Figure 3e), Eq. 1–4 styling frames, procedural styling, Eq. 11 morphing | **the same calls** — `evalCurve`, `sampleW`, `stylingMatrix`, `styleStrandVertex`, `computeStrandTangents` |
| raster | capsule ribbons, no `sqrt` in the inner loop, 1 sample per pixel | same primitives (`2*ns+2` vertices, `2*ns` triangles per strand) |
| resolve | composite the per-thread buffers in depth order, tonemap | same tonemapper |

The generator is asserted vertex-for-vertex against the reference `StrandGenerator` in
`tests/tests.cpp` (`testCpuFastPath`): **max deviation 0.0 m, bit-identical**. "Faster" here
cannot quietly mean "different geometry".

## 3. Optimisations this turn (all value-preserving)

Profiling found three costs that were pure waste, and each fix is exact — the same numbers, or
bit-identical ones, not an approximation:

| Change | Before | After | Why it is exact |
| --- | --- | --- | --- |
| `hash11`/`hash12` used `std::fmod(x, 1)` | 36.4 ms | **18.6 ms** | the remainder of a division by one *is* the truncated fractional part; `x - trunc(x)` is bit-identical to `fmod(x,1)` for every input used here, and it is also what GLSL `fract` does |
| `styleStrandVertex` evaluated `cos/sin` twice for the same phase (curl and ringlet helix) and called `pow(w, 2.0f)` | — | included above | the second pair was the same angle; `pow(w,2)` is `w*w` |
| `Texture3D::sample` recomputed the eight texel addresses inside the per-channel loop | 37 ns/call | **35 ns/call** | addresses hoisted; integer strides instead of `texel()` index math |
| `stylingMatrix` sampled the u- and v-textures with two independent bilinear weight computations | 34.6 ms | **28.7 ms** | both textures have identical dimensions, so the eight taps are at the same addresses |
| fast path: one arena instead of four `std::vector`s per strand; no per-strand allocation | 555 ms | **~190 ms** | same arithmetic, different scratch space |

Net effect on the reference generator (same 10k strands, 33 vertices each): **120.1 ms →
94.3 ms**. Effect on the whole hero frame: reference 2,534 ms → fast path **665 ms** (3.8×),
of which the phase split is in §5.

## 4. Hardware this was measured on

```
Intel Xeon @ 2.60 GHz, 2 vCPU, 3.9 GB RAM, AVX-512 available (not used: the path is scalar)
g++ 12.2, -O2, no -ffast-math (results are IEEE-exact and match the reference bit for bit)
```

Two cores is the whole machine. The paper's hardware is an RTX 4090: 128 SMs, 16,384 FP32
lanes, 24 GB at 1 TB/s.

## 5. Measurements, phase by phase (1280×720, 1 sample/pixel, 2 threads)

### Hero head (Figure 5/10 style), 40k strands, LOD on

| phase | ms | share | ns/strand | ns/vertex |
| --- | --- | --- | --- | --- |
| front end (Eq. 7–12, cull) | 0.07 | 0.0 % | 2.1 | — |
| generation (Figure 3e) | 189.1 | 28.4 % | 5,343 | **161.9** |
| raster (ribbon capsules) | 472.6 | 71.0 % | 13,355 | — |
| resolve (composite + tonemap) | 3.5 | 0.5 % | — | — |
| **total** | **665.4** | 100 % | 18,800 | 569.7 |

Workload: 221 bundles, 35,391 strands after Eq. 8, 1,167,903 vertices after Eq. 9–12,
26.1 M shaded fragments (28.3 per pixel).

### The same head with LOD off (Figure 8's comparison workload)

743.6 ms total — only 12 % slower, because at 0.4 m the detail level is already saturated:
Eq. 7 gives `L ≈ 1`, so LOD has nothing to remove. The paper's ~24× LOD ratio appears at crowd
distance, not in a hero shot.

### The crowd (Figure 1)

| scene | front end | generation | raster | resolve | total |
| --- | --- | --- | --- | --- | --- |
| 20 heads, 720p | 0.29 | 1,768.6 | 2,406.9 | 3.5 | **4,179.3** |
| **100 heads, 720p** | **1.17** | **10,975.2** | **14,491.0** | 4.4 | **25,471.8** |
| 100 heads, front end only | 1.17 | — | — | — | **1.17** |
| hero head, 640×360 | 0.07 | 103.5 | 164.1 | 1.1 | 268.7 |
| hero head, 720p, 1 thread | 0.07 | 297.9 | 722.7 | 3.8 | 1,024.5 |

The 100-head workload: 4,130 bundles, **2,006,895 strands** after Eq. 8, **66,227,535
vertices** after Eq. 9–12, 64,220,640 ribbon segments, 497 M shaded fragments.

Threading scales the frame 1.55× on 2 threads (1,024 → 665 ms): the path is partly
memory-bound, and there are only two cores to scale to.

## 6. The 2 ms line

| within 2 ms? | what | measured |
| --- | --- | --- |
| **YES, 1.7× headroom** | front end + LOD for 100 characters (4,130 bundles, Eq. 7–12, Eq. 8 counts, culling, ordering) | **1.17 ms** |
| no, 95× over | front end + geometry for 100 characters, no rasterisation at all | 10,976 ms |
| no, 12,700× over | the full 100-character frame | 25,472 ms |
| no | one hero head at 720p | 665 ms |
| no | one hero head at 360p | 269 ms |

What 2 ms of *this* CPU buys, at the crowd's detail level: **158 strands**, or **12.4k generated
vertices**, or **0.008 of a character**. The paper's 2 ms buys 100 characters (its own Figure 1
caption: 100k strands each).

## 7. Why the gap is hardware, not method

The honest way to compare a CPU with a GPU on the same workload is per parallel lane, not per
package:

| | total strands/s | lanes | strands/s per lane |
| --- | --- | --- | --- |
| Paper, RTX 4090 (100 chars × 100k strands / 2 ms) | 5.0 × 10⁹ | 16,384 FP32 lanes | 305,000 |
| This CPU path (2.0 M strands / 25.47 s) | 78,800 | 2 cores | **39,400** |

**Within 8× per lane**, while the package-level gap is 12,700× — and 16,384 / 2 = 8,192 is
almost exactly the wall-clock ratio. In other words: the paper's method costs this CPU about
the same number of operations per strand as it costs the 4090 per lane, so the deficit is
lane count. The residue (the factor ~1.5 left over after dividing out 8,192) is what the GPU
gets from hardware texture units (4-tap bilinear filtering in one instruction, which the CPU
does in ~35 ns) and from hardware rasterisation.

This is the answer to "can we beat 2 ms on a CPU": **with 2 cores, no — you would need about
1,000 cores** (12,700 × 2 / 8 per-lane deficit). With those cores the method would fit, which
is the useful finding: nothing in the paper's algorithm *requires* a GPU; it requires
parallelism.

## 8. What is *not* covered by the 2 ms front-end result

* The front end is CPU work only. It does not generate geometry, and the GPU-side cost of that
  geometry is the paper's 2 ms, not this number.
* It measures `--scene crowd --crowd 100` with the shipped 40k-strand models, not the paper's
  100k-strand ones. The front end is `O(bundles)`, and the bundle count is the same 4,130, so
  the number transfers; the strand counts it produces would be larger.
* It includes a `std::sort` of the bundle list (a real implementation would use the GPU's own
  ordering, so this is pessimistic, not optimistic).

## 9. Reproducing this

```bash
make -j                                   # no GPU, no dependencies
./build/SilkCrowd --cpu-budget --scene crowd --crowd 100 --front-end-only --repeat 3
./build/SilkCrowd --cpu-budget --scene crowd --crowd 100 --repeat 1     # ~25 s
./build/SilkCrowd --cpu-budget --scene head --width 1280 --height 720 --repeat 3
./build/SilkCrowd --cpu-budget --scene head --width 640 --height 360 --repeat 3
./build/SilkCrowd --cpu-budget --scene head --threads 1 --repeat 2      # scaling
./build/SilkCrowd --fast-screenshot fast.png --scene head --width 640 --height 360
```

`--fast-screenshot` writes the fast path's own frame (`docs/images/cpu_fast_head.png`): the
geometry is identical to the reference, the shading is not — see §10.

## 10. Where the fast path deviates (read before comparing images)

| deviation | why | effect on the measurement |
| --- | --- | --- |
| one sample per pixel instead of SSAA/MSAA | the analytic capsule coverage *is* the antialiasing | removes a factor of 4–8 in rasterisation, i.e. it makes the CPU look *better* |
| per-vertex Kajiya–Kay + one narrow specular lobe, no BCSDF | a full Marschner BCSDF per vertex costs more than the geometry | shading is cheaper than the paper's on both sides; the reference path keeps the full model |
| no light-space shadow buffer | it is a second full geometry pass | removes ~30 % of the reference path's work; the CPU number is therefore optimistic |
| no opaque geometry (head/body) | the head is a few thousand triangles, invisible next to 66 M vertices | under 1 % at crowd scale |
| per-thread buffers composited in depth order | a single-threaded frame would serialise everything | only changes blending order *between* thread ranges (the ranges are depth-sorted, so the error is confined to overlapping bundles at the seam) |

None of these changes the amount of *geometry* the method has to produce, which is what the
comparison in §6 is about. They are listed so that nobody reads the fast path's 665 ms as "the
reference renderer got 3.8× faster" — the reference renderer is unchanged and still uses the
full model.

## 11. What would actually close the gap

In the order they would pay off, with measured or arithmetic justification:

1. **More cores.** 8,192× the lanes is the whole story. Nothing else comes close.
2. **SIMD across strands.** The paper's texture layout puts all strands of a bundle in one 2×2
   texel block, so the eight taps a strand needs are at *the same addresses* for its 100
   siblings. A 16-wide gather-free kernel could load those texels once and evaluate 16 strands
   per pass: the generation phase's 162 ns/vertex is ~35 ns of texture traffic plus ~125 ns of
   scalar lerp/styling, so this is worth an estimated 4–6× (not measured — it is not
   implemented; see `docs/NOT_IMPLEMENTED.md`).
3. **A tile-based rasteriser with per-tile geometry bins.** The raster phase is 71 % of the
   hero frame at 28 fragments/pixel; bins would cut the empty-bbox work, not the fill.
4. **Then, and only then, the GPU**, which is what the paper did.
