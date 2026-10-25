// Level-of-detail (paper Section 3.6, Eq. 7-12, Figures 6-8).
//
// Two independent LOD types:
//   * strand LOD  - reduces the number of strands per bundle (Eq. 8),
//   * vertex LOD  - reduces the number of vertices per strand (Eq. 9-12),
// both driven by the same detail level L (Eq. 7) but with separate alpha parameters.
#pragma once

#include <cstdint>

#include "silkcrowd/math.h"

namespace silk {

struct LodParams {
    // Two independent alpha parameters so the trade-off of the two LOD types can be
    // tuned separately (Section 3.6).
    // Eq. 7 scale factors. With h ~ 2 cm bundles and R = 1080 rows, alpha ~ 0.03 puts
    // the strand-LOD knee at a few metres, which is where the crowd scenes live.
    float alphaStrand = 0.022f;
    float alphaVertex = 0.030f;
    // Transition window between vertex detail levels (lambda in Eq. 10/12).
    float lambda = 0.35f;
    bool enableStrandLod = true;
    bool enableVertexLod = true;
};

// Eq. 7:  L = min(1, h * R / (2 * d * tan(theta/2)) * alpha)
//          = min(1, h * R / viewHeight * alpha)
// i.e. the ratio between the model's projected pixel size and R, scaled by alpha.
//  h     : world-space size of the hair model (bundle scalp face size for strand LOD,
//          strand length for vertex LOD)
//  R     : vertical image resolution
//  d     : distance to the camera
//  theta : camera (vertical) field of view; d * tan(theta/2) is half the frustum height
float detailLevel(float h, float verticalResolution, float distance, float fovYRadians,
                  float alpha);

// Orthographic variant mentioned in Eq. 7: 2*d*tan(theta/2) is replaced by the view height
// in world space (used by the directional-light shadow pass).
float detailLevelOrthographic(float h, float verticalResolution, float viewHeight, float alpha);

// Eq. 8:  N_LOD_i = ceil(L * (N_i + delta_i))
// delta_i in [0,1) is a per-bundle random increment; it desynchronises the bundles so
// that detail levels do not all drop a strand at the same time. Always returns >= 1
// for L > 0.
uint32_t strandCountLod(uint32_t strandsInBundle, float delta, float L);

// Hair thickness compensation for strand LOD: min(1/L, N_i). Deliberately *not*
// synchronised with the discrete strand-count changes (Section 3.6.1).
float strandThicknessScale(float L, uint32_t strandsInBundle);

// Eq. 9:  e = max(0, ceil(log2(L * ns)))  ->  2^e + 1 vertices.
// `nsMax` is the full-resolution number of strand segments (a power of two).
int vertexLodExponent(float L, int nsMaxSegments);

// Eq. 10:  e~ = max(0, ceil(log2(L * ns) - lambda))
int previousVertexLodExponent(float L, int nsMaxSegments, float lambda);

// Eq. 12:  gamma = 1 - (log2(L*ns) - e~) / lambda, clamped to [0,1]; 0 when e == e~.
float vertexMorphGamma(float L, int nsMaxSegments, float lambda);

// Which lower level each strand morphs towards, and whether morphing is active.
struct VertexLodResult {
    int exponentE = 0;      // 2^e + 1 vertices are generated
    int exponentPrev = 0;   // e~
    float gamma = 0.0f;     // Eq. 12
    int numVertices() const { return (1 << exponentE) + 1; }
    int numSegments() const { return (1 << exponentE); }
    bool morphing() const { return gamma > 0.0f; }
};

VertexLodResult computeVertexLod(float L, int nsMaxSegments, float lambda);

// Per-bundle strand-LOD state.
struct StrandLodResult {
    uint32_t numStrands = 1;
    float thicknessScale = 1.0f;
};

StrandLodResult computeStrandLod(uint32_t strandsInBundle, float delta, float L,
                                 uint32_t minStrands = 1);

// Deterministic per-bundle delta_i in [0,1) for Eq. 8.
float bundleDelta(uint32_t bundleIndex, uint32_t seed = 0u);

// Distance used by Eq. 7 for a bundle/strand: distance to the closest point of the
// model's world-space bounding sphere, clamped away from zero.
float lodDistance(Vec3 cameraPosition, Vec3 center, float radius);

}  // namespace silk
