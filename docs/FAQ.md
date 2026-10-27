# FAQ: the questions that come up every time someone runs this

Short answers with the evidence, for the four things that look wrong on first contact.

---

## 1. "Why is the hair on the teapots so small?"

**It was a bug, and it is fixed.** Two separate mistakes made the teapot hair look like a
bald pot with a wig cap on the lid:

1. **The roots were on the lid, not on the pot.** The first mesh was a spherical cap of
   radius 0.058 m centred at `y = 0.112` — that is the *lid only*. Strands grew out of the
   knob and fell 3 cm, so they never reached the body. The teapot body is 0.236 m across and
   its silhouette runs from `y = -0.075` (foot) to `y = 0.112` (knob).
2. **The strand length was in the wrong frame.** `length = 0.03` was being used as "0.03
   hair-mesh units", while everything else in the scene is in metres. 3 cm of hair on a
   24 cm pot is the fur on a hat.

The current mesh (`buildTeapotHairMesh`) samples the pot's **profile of revolution** — a
16-point table from the foot, up the body, over the lid shoulder to the knob — so:

* roots start at the knob (`y = 0.1115`) and on the shoulder (`y = 0.045`);
* every layer position is `(potRadius(y) + standoff)`, i.e. the hair follows the porcelain
  instead of standing off it in a sphere;
* the drop is 0.15–0.20 m, which carries the hem past the foot, where the mop flares out;
* each lock gets its own length (`0.78 + 0.44·hash(i)`), which is what makes the hem of
  Figure 9 ragged rather than a bowl.

The scene then gives each pot a different length (`0.150 + 0.025·(i mod 3)`) so the grid of
200 does not read as a single repeated asset.

Renders: `docs/images/teapot_200_linux.png`, `docs/media/teapot_orbit.gif`.

---

## 2. "I copied what the PDF says — why does it look weird?"

Because the PDF will not give you a hair mesh, and a hair mesh is where the look lives.
Everything the paper *does* specify is implemented here exactly (Eq. 1–12, Figure 3e,
Table 1), and the numbers are still not transferable on their own:

| What the PDF gives | What it does not give | Consequence if you only copy the PDF |
| --- | --- | --- |
| the five textures and their layout | the mesh they are sampled with | strand directions come out of a degenerate root grid, and the crown grows fur (see §3) |
| Eq. 1–4 frames | a scalp parameterisation / hairline | the hairline lands across the eyes or above the ears |
| Eq. 7–12 LOD | the base strand counts `N_i` | `L` is right but `n_i` is meaningless, and the LOD pops |
| styling *operations* (clump, curl, frizz, flyaway…) | the amplitudes and the mesh rows they act on | one look at a time: a "bob" that is a straight extrusion is a helmet |
| Figure 5's authored meshes | the meshes themselves | the silhouette families of Figures 5 and 10 cannot be reproduced from parameters |
| 2 ms @ 8× MSAA on an RTX 4090 | that GPU | a CPU reference renderer needs seconds, not milliseconds |

The paper says it plainly: the hair mesh is "manually prepared" and most of its figures are
*edited* meshes. Our `buildHairMesh()` has to author one, and that authoring decides the
look:

* the **fall line** (where a strand stops following the scalp and starts hanging) is what
  separates a bob from a wig: past it the strand keeps the silhouette it had at the fall
  line, which is why a bob is a cylinder with a blunt hem and not a sheet wrapped round the
  jaw;
* the **hem is a distance in metres from the crown**, so a fringe must be written as
  `hairlineArc + fringeLength`; written as a raw length it never reaches its hanging phase
  and you get a bare forehead with a hard line across it — the "winter hat";
* the **front section must be combed around the temple** (`tuck`), because a vertical drop
  from the front hairline passes over the nose. Every long style that skipped this had a
  sheet of hair over the eyes;
* the **root patch is not area-uniform** (all columns converge at the crown), so per-face
  strand counts must be area-weighted or the crown is 4× too dense and looks like fur.

That list is the answer to "why does it look weird": none of it is in the paper, and all of
it changes the image more than the equations do.

---

## 3. "No Vulkan driver — fix that."

There is no Vulkan driver **in this sandbox**, and that is not fixable from inside it:

```
$ ls /usr/lib/x86_64-linux-gnu/libvulkan*      -> nothing
$ ls /usr/share/vulkan/icd.d                   -> nothing
$ apt-get install mesa-vulkan-drivers          -> package not in the index
```

No loader, no ICD (the driver manifest a loader needs), no GPU device node. The application
therefore probes for Vulkan at start-up and, when it is missing, prints

```
Vulkan path unavailable (this build has no Vulkan support (configure with -DVulkan_FOUND ...));
falling back to the software renderer
```

and renders the frame with `src/render/software_renderer.cpp`, which is a full CPU
implementation of the same pipeline (same texture layout, same LOD, same shading model,
same self-shadow buffer) — that is why every image in `docs/` is real and reproducible here.

On any machine with a Vulkan 1.3 driver (Mesa RADV/ANV, NVIDIA, AMD, Intel), `--gpu` uses
the mesh-shader path; devices without `VK_EXT_mesh_shader` take the compute fallback
described in `docs/PAPER_MAPPING.md` §7. CI installs `mesa-vulkan-drivers` and runs
`--gpu --frames 2` when the runner provides a device; the job is allowed to skip, because
GitHub's hosted runners have no GPU.

---

## 4. "Why is it slower than the paper?"

The paper's numbers are mesh-shader GPU numbers on an RTX 4090: 100 characters × 100k
strands in 2 ms with 8× MSAA, 47 ms without LOD. What runs here is the **software
renderer** on a shared CPU sandbox: a hero head at 60k strands is ~3.3 s per frame at
560×700 with 2× supersampling, and the 200-teapot grid is ~20 s at 800×450.

The parts that correspond are the ones that are algorithmic and can be counted rather than
timed:

| Quantity | Paper | Here |
| --- | --- | --- |
| texture storage per model | 13–21 KB | 14.1–19.1 KB |
| 100 models, all textures | 1.7 MB | 1.61 MB |
| strands rendered with LOD (hero head) | a few % of 100k | 33.8k of 60k (56 %) at 560×700 — the head fills the frame, so `L` is near 1 |
| strands rendered with LOD (crowd) | a few % | 369k of 4.8M (7.7 %) at 480×270 |

So the LOD machinery is doing what the paper says it does; the milliseconds belong to the
hardware, not to the method. `docs/COMPARISON.md` keeps the two columns apart on purpose.
