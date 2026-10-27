# Roadmap: everything that is still left, in one place

This is the single list the team asked for: **every open item from the paper, every item that
`docs/NOT_IMPLEMENTED.md` records as not done, and every upgrade that is not in the paper at
all** — merged, deduplicated, given an ID, an effort, a risk and a way to prove it worked.

Three documents feed it and none of them is replaced by it:

| Source | What it holds |
| --- | --- |
| `docs/PDF_COVERAGE.md` | per-section audit: **58 items — 41 done, 13 partial, 4 N/A, 0 open** for the core method, plus the 6 environment items in its §13 |
| `docs/NOT_IMPLEMENTED.md` | §1 hardware-blocked (6), §2 paper-has-it-we-don't (17), §3 accepted quality gaps (5), §4 deliberately not done (5) |
| `docs/CPU_2MS.md` | the 2 ms analysis, the measured phase split, and §11's ranked list of what would close the gap |

**How to read a row.** `R##` is the ID (stable — quote it in commits and PRs). *Effort* is
S ≈ under a day, M ≈ 1–3 days, L ≈ a week or more, for one person. *Gain* is **measured** where
the repository already has a number and **estimated** where it does not; every estimate is
labelled `est.` because nothing here invents a millisecond (`NOT_IMPLEMENTED.md` §4.2). *Proof*
is the command or CI step that turns the row from "done" into "verified" — a row is not closed
by writing code, it is closed by that line passing.

---

## 0. The short version

| Tier | What it is | Rows | Why it is in this tier |
| --- | --- | --- | --- |
| **P0** | Finish what the paper asks for and we do not do | 9 | the only remaining *paper* gaps; two of the nine (`R01`, `R03`) need hardware, the other seven are code |
| **P1** | Make the CPU path fast for real | 10 | the 2 ms question; the gap is lanes, and the only lane-multiplier available to one core is SIMD (`R10`); `R11` is done |
| **P2** | Prove the GPU path | 9 | none of the Vulkan work can be signed off without timestamps, and this environment has no driver |
| **P3** | Make the renderer better, not just faster | 8 | quality, look, and the two weakest silhouettes |
| **P4** | Engineering hygiene that stops regressions | 12 | every number in the docs should fail the build when it drifts |
| **P5** | Hardware-blocked (a restatement of `R01`/`R03`/`R23`/`R25`/`R26` + one closed item) | 6 | so they are "declined" rather than "forgotten" |

Total: **48 `R` rows — 45 open, 3 already closed** (`R02` GLSL validation, `R11` baked blue-noise
sets, `R50` the sanitizer build; all three stay in the tables as pointers to their gates).
**Five** of the open rows cannot be verified without hardware this sandbox does not have (`R01`,
`R03`, `R23`, `R25`, `R26`); §6 lists them with their unblocking condition. Everything else can be
written and checked here.

**Two rows were implemented in the same turn as this list**, because a roadmap nobody starts on is
just a wish list:

* **`R11` — baked blue-noise sets.** `--bake-noise` writes six sets to `assets/noise/*.scbn`
  (92 KB, 8 192 samples); `blueNoiseSet()` loads them, rejects a file whose parameters or contents
  do not fit the request, and regenerates when the asset is missing. `--validate` asserts the files
  are what this build's generator produces, CI re-bakes and fails if the checked-in bytes are
  stale, and the generator is compiled with `-ffp-contract=off` so the file does not depend on the
  compiler's FMA policy. Measured: cold start of
  `--cpu-budget --scene head --width 640 --height 360` **1,668-1,821 ms → 730-734 ms**, i.e. about
  a second off every run, which is more than a whole front end (1.12 ms is the front end for the
  *crowd*; for this head it is 0.05-0.08 ms).
* **`R50` — sanitizers.** `make sanitize` builds the core library and the tests with
  ASan + UBSan and runs them (**3 351 checks, 0 failures, no leaks**), and a new
  `Sanitizers (ASan + UBSan)` CI job runs it. It paid for itself immediately: it found a genuine
  use-after-free — `resetSampleSetStats()` cleared the sample-set cache while `cpu_fast.cpp` and
  `software_renderer.cpp` held references into it — which is fixed.

That is the template for the rest: code, a check in `--validate` or the test binary, a measured
number in a doc.

---

## 1. P0 — the paper, finished

The paper-side gaps are all in `PDF_COVERAGE.md` §13 and `NOT_IMPLEMENTED.md` §2.1–2.3. They are
few but they are the ones an author would ask about first.

| ID | Item | State today | Gain | Effort | Risk | Proof |
| --- | --- | --- | --- | --- | --- | --- |
| `R01` | Run the Vulkan path on a machine with a real GPU (§13.1) | CI runs it headless on lavapipe; it has never drawn to a window on real hardware | closes 3 hardware rows at once | S (a GPU + an hour) | none | `SilkCrowd --gpu-info` then `--gpu` on any Vulkan 1.3 device |
| `R02` | GLSL validation (§13.2) — **closed** | 12 shaders compile with glslang 16; CI compiles them on 5 platforms | — | — | — | the `shaders` step of the CI `linux` job |
| `R03` | Absolute GPU timings: 2 ms, 1.8 ms teapots, 47 ms no-LOD (§13.3) | measured on the CPU instead: 24 831.53 ms whole frame, 1.12 ms front end | the paper comparison | S (hardware) | none | `--bench` on reference hardware; until then `R20` |
| `R04` | **3.2 MB of textures for 200 teapots** (§13.4, `NOT_IMPLEMENTED` 2.7) | **30.2 MB** — every bundle gets a full 2×2 block on every slice | 9.4× less storage; the paper's 1.7 MB for 100 heads becomes reachable | M | medium — the branch-free texel address is the reason it is fixed-size; a size class adds one indirection | `--assets --scene crowd --crowd 100` prints total MB; assert ≤ 4 MB in `--validate` |
| `R05` | Figure 5's **hair-mesh editing** workflow (§13.5, 2.2) | the Qt app orbits, switches style, changes LOD, simulates; it cannot move a layer vertex | the one interactive feature of the paper we lack | L | medium — never built here (no Qt6 in the image) | build `cmake -DSILKCROWD_WITH_QT=ON`, drag a handle, screenshot in `docs/media/` |
| `R06` | Hand-authored Figure 10 silhouettes (§13.6, 2.3) | ten procedural presets; **ponytail and pixie are the weakest** | the two rows a reviewer would pick out of the style sheet | M | low | `docs/images/styles_female.png` re-rendered with the new templates |
| `R33` | The layer-handle **editor UI** itself (implementation of `R05`) | — | — | L | medium | interactive session + video in `docs/media/` |
| `R34` | **Ponytail** mesh template (a gathered mass behind the head, a tie, a tail sweep) | a styling parameter cannot gather hair into a tail — this is a mesh change | fixes one of the two weakest presets | M | low | side-by-side render against the paper's Figure 10 ponytail |
| `R35` | **Pixie / buzz** template (shorter hairline, no hem, higher root density) | same | fixes the second weakest preset | S | low | as `R34` |

Note on `R01`/`R03`: they are the only two rows in P0 that are *hardware*, and they are mirrored
in P5 so they cannot be lost. Everything else in P0 is code that can be written and reviewed
today, with the caveat that `R05`/`R33` need a Qt6 build to see.

---

## 2. P1 — the CPU path: from 661 ms to something defensible

Measured baseline (this sandbox: 2 cores @ 2.6 GHz, head 1280×720, `--repeat 3`):

```
front end 0.07 ms | generation 187.34 | raster 470.37 | resolve 3.64  →  661.46 ms
```

and crowd-100 720p: front end **1.12 ms (inside 2 ms)**, generation 10 295.10, raster
14 532.17, resolve 3.10 → **24 831.53 ms**.

The lane arithmetic from `CPU_2MS.md` §10: the CPU is within **7.5×** of the paper *per
parallel lane*; it is behind by **8 192×** in lanes. So no single row below can produce 2 ms —
what they can do is multiply one core by 4–8×, which is what the list is ranked by.

| ID | Item | What changes | Gain | Effort | Risk | Proof |
| --- | --- | --- | --- | --- | --- | --- |
| `R10` | **SIMD bundle-coherent generation kernel** (`CPU_2MS.md` §11.2) | `generateStrand` evaluates one strand; the paper's layout puts a bundle's strands in the *same* 2×2 texel block, so 8–16 lanes share every tap. Widen to 8 (AVX2) / 16 (AVX-512) / 4 (NEON) with the taps loaded once, then the styling lerps are vertical ops over the lane vector | **est. 4–6×** on the 187 ms generation phase; ~46 ns/vertex | L | high — this is a kernel rewrite, and the memory notes say address hoisting alone gave only 37→35 ns/call, so the win must come from *sharing* the taps across lanes, not from cheaper scalar code | `testCpuFastPath()` must still report max deviation **0.0** against the scalar reference; `--cpu-budget --json` before/after |
| `R11` | **Bake the progressive blue-noise sets** (`NOT_IMPLEMENTED` 2.10) — **DONE** | `--bake-noise` writes all six sets to `assets/noise/*.scbn` (92 KB, 8,192 samples); `blueNoiseSet()` loads them, rejects a file whose parameters or contents do not match, and regenerates when the asset is missing. `--bake-noise` is the only place elimination still runs | **measured** cold start `--cpu-budget --scene head --width 640 --height 360`: **1,668-1,821 ms → 730-734 ms** (~1 s, more than a whole front end) | S | low | 31 new assertions in `testBakedSampleSets()`; `--validate`: "baked blue-noise assets match the generator (6/6 sets, 8,192 samples, max deviation 0.000000)"; CI re-runs `--bake-noise` and requires `git status assets/noise` to be clean |
| `R12` | **4-wide capsule rasteriser** | the raster phase (470 ms head / 14.5 s crowd) tests 4 pixels of one span per pass with the same bundle's tangent/AO; SIMD coverage + a horizontal mask | est. 2–3× on raster | M | medium — the *branchless scalar* variant was tried and **lost** (609 vs 469 ms); SIMD-over-pixels is a different experiment, not a retry of that one | fragment count and shaded pixels identical; wall clock |
| `R13` | **Frame-coherent bundle cache** | a bundle whose LOD class, style and sim state did not change keeps its generated vertices; a camera move only re-rasterises. Recompute when the LOD result changes | interactive orbits become raster-dominated (470 ms → ~0 generation) | M | medium — invalidation bugs are silent; needs a hash of every input | `--repeat` with a static camera vs a moving one; deviation 0.0 with the cache on |
| `R14` | **Shrink the vertex record** | the composite pass walks every vertex; packing tangent frame + AO into 8 bytes instead of 20+ cuts the resolve/interpolation bandwidth | est. 1.2–1.5× on composite | S | low | `sizeof(Vertex)` in a static_assert; identical pixels |
| `R15` | **Fix the thread scaling** | 2 threads give **1.62×**, not 1.9×: a persistent worker pool, per-thread buffers reserved once, and tile ownership so no false sharing on the counters | est. +15–20 % at 2 threads, more at 8+ | M | low | 1/2/4/8-thread sweep printed by `--cpu-budget` |
| `R16` | **Sort-free depth composite** | per-thread depth-ordered buffers are merged at the end; a radix sort over 32-bit depth keys is O(n) and branch-light | est. 1.5× of the resolve+composite tail | M | medium | ordering test: the sorted and unsorted composites agree to 1 LSB |
| `R17` | **Tap LOD** (the texture analogue of the paper's vertex LOD) | at crowd distance one bilinear tap instead of two + a lerp; the paper's Figure 8 argument, applied to the *texture* axis | est. 25–35 % of generation | M | medium — visible as a slight loss of azimuthal detail; must be a flag with a pixel-diff budget | `--taps 1` vs `--taps 2`, mean abs diff on `docs/images/` renders |
| `R18` | **`--cpu-budget --json`** | machine-readable phases + scene metadata, so CI can assert a budget | turns every row above into a gate | S | low | CI step parses it and fails on >10 % regression |
| `R19` | **Release profile: `-O3`, `-ffast-math`-free, PGO** | current build flags are CMake defaults; a measured `-O3 -march=x86-64-v3` translation unit already exists for the strict check | est. 10–25 % (measure before believing it) | S | low — `-ffast-math` is *out*: it would break the bit-identical test | same output, wall clock; `-Werror` clean |

`R11` is closed (see the note in §0). `R10` is the big one: it is the only row that could
plausibly move the crowd figure by a large factor on one machine, and it is the row to do first
in this tier.

---

## 3. P2 — prove the GPU path (`NOT_IMPLEMENTED` 1.1–1.5, 2.12–2.14)

| ID | Item | State | Effort | Risk | Proof |
| --- | --- | --- | --- | --- | --- |
| `R20` | **`--gpu-budget`**: `vkCmdWriteTimestamp` pairs around front end / amplification / mesh-shader or compute / raster / WBOIT resolve, scaled by `timestampPeriod`, printed per pass and averaged over `--repeat` | does not exist; the Vulkan path is only timed as a whole frame in CI | M | medium — must degrade gracefully when the queue does not support timestamps (lavapipe does, most drivers do) | the CI `linux` job prints a phase table on lavapipe |
| `R21` | **GPU screenshot smoke test**: render the head on lavapipe to PNG, assert it is not black and its mean luma is in range | CI runs `--gpu` but does not inspect the image | S | low | new CI step + the PNG as an artifact |
| `R22` | **Bind the scan diffuse in the Vulkan head pipeline** (`NOT_IMPLEMENTED` 2.14) | the GPU head still evaluates analytic face masks; only the software path samples the 2048² atlas | M | medium — descriptor bindings have never been validated | `--gpu --head-scan` on lavapipe, image compared to the software render |
| `R23` | **Mesh-shader `local_size_x` sweep** (2.13) | 32 vertices per workgroup by construction, never tuned | S (code) + hardware | low | `--tune-mesh-shader` prints a table on a real device |
| `R24` | **MSAA 1×/4×/8× comparison** (1.3) | implemented and clamped to device support; never looked at | S | low | lavapipe renders at each sample count, PSNR against a 4× SSAA reference |
| `R25` | **On-device Android frame** (1.4) | APK + 12 SPIR-V modules build in CI, never executed | M | medium — hosted runners have no KVM | `adb install` + Perfetto trace in `docs/media/` |
| `R26` | **GPU counter capture** (1.5): an Nsight/RGP recipe and a `--gpu-counters` hook for texture-unit throughput and cache residency of the 13–21 KB model | the layout is validated numerically only | S (code) + hardware | low | a capture in `docs/images/` |
| `R27` | **WBOIT vs sorted composite A/B** | the GPU path has WBOIT; the software path sorts. Never compared | S | low | same scene, two images, per-pixel diff |
| `R28` | **Compute-fallback parity test** | the compute path exists for devices without mesh shaders; its output vs the mesh-shader path is not compared | M | medium | lavapipe forces the fallback; image diff |

---

## 4. P3 — make it look better (`NOT_IMPLEMENTED` §3, 2.4–2.9)

| ID | Item | State | Effort | Risk | Proof |
| --- | --- | --- | --- | --- | --- |
| `R30` | **Variable-size texture blocks + a size class per bundle** — the engineering half of `R04` | `HairMeshTextureSet` allocates a fixed 2×2 block per bundle per slice | M | medium | `--validate` storage check; texel addressing still branch-free per class |
| `R31` | **Baked mip chain / anisotropic taps** for the slice textures | no mips: a distant bundle aliases and relies on the dual-scattering term | M | low | image diff at crowd distance; shimmer metric over a camera sweep |
| `R36` | **TAA + LOD hysteresis** (2.9): per-bundle temporal state on top of Eq. 8's δ and Eq. 12's γ | the paper has exactly those two mechanisms; no extra state here | M | medium | popping metric: count of LOD class flips per second over a dolly |
| `R37` | **Deep opacity maps / VSSM** for per-strand shadows (2.4) | one light-space envelope buffer + a crown term | L | high — 2–3× the shadow cost | shadow A/B renders; the dark interior improves |
| `R38` | **Dual scattering in the software path** | `hair.frag` has it; the software path uses a penetration-depth ambient floor | M | low | `docs/images/` re-rendered; interior luma comparison |
| `R39` | **Marschner BCSDF in the fast path** (2.5) | the fast path uses a one-lobe approximation by design | M | medium — the 2 ms path is the last place to add cost | `--fast-screenshot` before/after; a `--shading full` flag |
| `R40` | **Reference-matched grooms instead of presets** (`NOT_IMPLEMENTED` §3.2) | three female styles are reproduced in silhouette, not fibre | L | medium | the one-to-one sheet at `docs/images/reference_floor.png` |
| `R41` | **A real hairline curve per head** (§3.5 note) | one angle per preset (`hairlineDrop` 4–12°) | M | low | styles sheet |

---

## 5. P4 — engineering hygiene (nothing here is in the paper; all of it prevents regressions)

| ID | Item | Why | Effort | Proof |
| --- | --- | --- | --- | --- |
| `R50` | **ASan + UBSan CI job** — **DONE** | `make sanitize` (instrumented core + tests) and a `Sanitizers (ASan + UBSan)` CI job. Found and fixed one real use-after-free on the first run | S | the job is green locally: **3 351 checks, 0 failures, no leaks** in 93 s. Extending it to `SilkCrowd --validate` and the GLB/`--screenshot` paths is the remaining half |
| `R51` | **TSan on the threaded fast path** | `--threads 2` is the first threaded code in the project; the depth-ordered composite assumes per-thread ownership | S | `--cpu-budget --threads 2` under TSan, zero reports |
| `R52` | **Fuzz the loaders** (`.skhm`, GLB, and the new `.scbn`) | three parsers eat untrusted files; a fuzz target with libFuzzer is a day | M | `-DSILKCROWD_FUZZ=ON` target + a seed corpus |
| `R53` | **clang-tidy / cppcheck gate** | `-Werror` is on, but only for warnings the compiler emits | S | a CI step on `linux` |
| `R54` | **`tools/check_docs_numbers.py`** | the stale "1.17 ms / 25.5 s / 12 700×" text in `PDF_COVERAGE.md` §13.3 (corrected to 1.12 ms / 24,831.53 ms / 12,400× this turn) existed because numbers are copied by hand into prose | S | the tool parses every `\d+\.\d+ ms` in README/docs and checks it against `--cpu-budget --json` |
| `R55` | **Deterministic asset rebuild** | `tools/make_assets.sh` writes GLBs/SKHMs; a byte-identical rebuild check catches accidental nondeterminism in the generators | S | rebuild, `git diff --exit-code assets/` |
| `R56` | **CMake presets + release artifacts** | `install()` exists; there is no `CMakePresets.json` and no release binary | S | `ctest --preset linux-release`, a tagged artifact |
| `R57` | **Doxygen for `include/silkcrowd/*.h`** | 15 headers, good prose comments, no generated API docs | M | a `docs` CI job publishing to Pages |
| `R58` | **NEON path for arm64** | `R10` done as AVX2 leaves Apple Silicon and Android on the scalar kernel | M (after `R10`) | the iOS CI job builds it; a Neon harness measures it |
| `R59` | **MSVC warning parity** | `SILKCROWD_STRICT` is off by default and the Windows job uses it; the warning set differs from GCC/Clang | S | both jobs at the same warning level |
| `R60` | **Split `src/app/main.cpp`** | 1 059 lines: argument parsing, scene building, validation and rendering all in one translation unit | S | pure refactor, tests unchanged |
| `R61` | **Golden-image regression** | every image in `docs/images/` is checked in but nothing compares a new render against it | M | a tolerant per-pixel diff (mean abs error) per image |

`R50`–`R55` are the rows with the best ratio of "prevents a bad day" to effort, and none of them
needs a GPU.

---

## 6. P5 — hardware-blocked (kept, not forgotten)

These are `NOT_IMPLEMENTED.md` §1 verbatim; they are duplicated here so that P0 is not read as
"four items and a GPU".

| ID | Item | Unblocks with |
| --- | --- | --- |
| `H1` | The paper's own timings on an RTX 4090 | one GPU |
| `H2` | 8× MSAA as a *look*, not as code | one GPU |
| `H3` | A frame on Adreno (the compute fallback) | a phone, or a runner with KVM |
| `H4` | Texture-unit throughput and cache residency of the 13–21 KB layout | Nsight/RGP |
| `H5` | Qt6 editor screenshots (`R05`/`R33`) | `qt6-base-dev` in the image |
| `H6` | GLSL validation *on this machine* (closed in CI instead, `R02`) | — |

---

## 7. What I would do, in order

Four batches. Each batch is a reviewable unit and ends with the whole suite green plus the
numbers regenerated.

1. **Batch A — cheap, certain wins (1 day).** `R11` (bake the noise — done), `R50` (ASan/UBSan —
   done), `R54` (docs numbers gate), `R51` (TSan on the threaded fast path), `R18` (`--cpu-budget --json`), `R14` (vertex
   record), `R19` (measure the release profile). All S, all verifiable without a GPU, and after
   them every later row has a gate to fail against.
2. **Batch B — the CPU multiplier (1 week).** `R10` (SIMD bundle-coherent kernel) with `R58`
   (NEON) behind it, then `R15` (thread scaling) and `R12` (4-wide raster). This is the batch
   that answers "how fast can one machine be", and the only one that can move the crowd number
   by a fair multiple.
3. **Batch C — finish the paper (1 week).** `R04`/`R30` (3.2 MB texture packing), `R34`/`R35`
   (ponytail, pixie), then `R22` (scan texture in the Vulkan head) and `R20`/`R21` so the GPU
   path finally has per-pass numbers even if they are lavapipe's.
4. **Batch D — the features (open-ended).** `R05`/`R33` (Figure 5 editor, needs Qt6), `R37`
   (deep opacity maps), `R40` (reference-matched grooms), `R13` (frame-coherent cache).

What is *not* on this list, on purpose: any claim that a single-core CPU will reach 2 ms on a
100-character scene. `docs/CPU_2MS.md` §10 puts that at ~3 300 cores, and `NOT_IMPLEMENTED.md`
§4.3 refuses to claim it. Batch B makes the honest claim larger; it does not make it 2 ms.

---

## 8. Pointers

| Question | Answer |
| --- | --- |
| Is every paper item accounted for? | `docs/PDF_COVERAGE.md` — 58 core items + §13's environment list |
| What is missing, exactly? | `docs/NOT_IMPLEMENTED.md` §1–§4 |
| Why is the CPU 12 400× slower than a 4090? | `docs/CPU_2MS.md` §10–§11 |
| What did the first real GPU run find? | `docs/GPU.md` |
| Where are the images and videos? | `docs/images/`, `docs/media/` |
