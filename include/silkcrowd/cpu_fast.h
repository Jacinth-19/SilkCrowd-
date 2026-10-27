// Fast CPU path: the paper's per-frame work at the lowest cost a general-purpose core can
// reach, with the phases timed separately.
//
// This is *not* a replacement for the software reference renderer (src/render/
// software_renderer.cpp, which is what produces the images in the README): the reference
// path keeps exact back-to-front order, a light-space shadow buffer and full per-vertex
// shading. This one exists to answer a specific question - how much of a frame can a CPU
// afford? - and to answer it with numbers instead of a guess:
//
//   phase 1  front end   Eq. 7-12 for every bundle: detail level, Eq. 8 strand count,
//                        Eq. 9-12 vertex count + morph factor, frustum and sub-pixel cull
//   phase 2  generation  the paper's on-the-fly geometry: 2 trilinear fetches + 1 lerp per
//                        vertex (Figure 3e), procedural styling, Eq. 11 morphing, Eq. 5/6
//                        tangents. Strands are still never stored between frames; the
//                        per-frame arena is scratch space for the rasteriser.
//   phase 3  raster      capsule ribbon rasterisation, no sqrt in the inner loop, one
//                        sample per pixel (the analytic ribbon coverage *is* the
//                        antialiasing; the reference path supersamples instead)
//   phase 4  resolve     composite the per-thread buffers in back-to-front order, tonemap
//
// Parallelism is by depth-sorted bundle ranges, one accumulation buffer per thread, which
// keeps the sorted order intact inside a range and only ever composites whole ranges.
// The deviations from the reference path (shading model, shadowing, one sample per pixel)
// are listed in docs/CPU_2MS.md; they are what the speed comes from, and none of them
// changes the amount of geometry the paper's method has to produce.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "silkcrowd/renderer.h"

namespace silk {

struct CpuFastStats {
    double frontEndMs = 0.0;
    double generateMs = 0.0;
    double rasterMs = 0.0;
    double resolveMs = 0.0;
    double totalMs = 0.0;
    uint64_t bundles = 0;          // bundles that survived culling
    uint64_t bundlesCulled = 0;
    uint64_t strands = 0;          // after Eq. 8
    uint64_t vertices = 0;         // after Eq. 9-12
    uint64_t segments = 0;
    uint64_t fragments = 0;        // pixel/segment overlap tests
    uint64_t pixelsShaded = 0;     // blended hair fragments
    int threads = 1;
    std::string toString() const;
};

// Test hook: generates a single strand with the fast path's code. The unit tests use it to
// assert that the fast path and the reference StrandGenerator produce the same vertices, so
// "faster" can never quietly mean "different".
void generateStrandFast(const HairInstance& hair, uint32_t face, uint32_t strandIndex,
                        int segments, float gamma, std::vector<Vec3>* out);

// The root sample the fast path places strand `strandIndex` of bundle `face` at, i.e. the
// block coordinate it feeds the hair mesh texture (Section 3.3, progressive blue noise).
// Exposed so the tests can hand the reference generator exactly the same root.
Vec2 fastRootCoordinate(const HairMesh& mesh, uint32_t face, uint32_t strandIndex);

class CpuFastRenderer {
public:
    // 0 = use every hardware thread.
    void setThreads(int n) { threads_ = n; }
    int threads() const;

    // Full frame (front end + generation + raster + resolve).
    void render(const Scene& scene, const Camera& camera, const RenderOptions& options,
                Image* out, CpuFastStats* stats);

    // Front end + generation only: the number the paper's 2 ms budget is compared against
    // for the geometry the method has to produce, without any rasterisation at all.
    void generateOnly(const Scene& scene, const Camera& camera, const RenderOptions& options,
                      CpuFastStats* stats);

private:
    int threads_ = 0;
};

}  // namespace silk
