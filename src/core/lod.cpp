#include "silkcrowd/lod.h"

#include <algorithm>
#include <cmath>

namespace silk {

float detailLevel(float h, float verticalResolution, float distance, float fovYRadians,
                  float alpha) {
    // Eq. 7: L = min(1, h * R / (2 d tan(theta/2)) * alpha).
    // The denominator is the world-space height of the view frustum at distance d, i.e. the
    // same quantity the orthographic variant below takes directly as `viewHeight`. Using
    // d * tan(theta) instead would both overestimate the projection by 2x at small FOVs and
    // diverge at theta = pi/2, where the frustum height is finite (2 d) rather than zero.
    const float halfFov = 0.5f * fovYRadians;
    const float viewHeight = 2.0f * std::max(distance, 1e-4f) * std::max(std::tan(halfFov), 1e-4f);
    const float L = (h * verticalResolution) / viewHeight * alpha;
    return std::min(1.0f, L);
}

float detailLevelOrthographic(float h, float verticalResolution, float viewHeight, float alpha) {
    const float vh = std::max(viewHeight, 1e-4f);
    return std::min(1.0f, (h * verticalResolution) / vh * alpha);
}

uint32_t strandCountLod(uint32_t strandsInBundle, float delta, float L) {
    if (L <= 0.0f) return 0;
    // Eq. 8: N_LOD_i = ceil(L * (N_i + delta_i))
    const float n = static_cast<float>(strandsInBundle) + clampf(delta, 0.0f, 0.9999f);
    const float v = std::ceil(std::min(L, 1.0f) * n);
    // "Equation 8 also ensures that each bundle is assigned at least one hair strand."
    // Upper clamp: Eq. 8 with delta_i > 0 can exceed N_i at L = 1; a bundle can never
    // contain more strands than the model defines, so the result is clamped to N_i.
    return std::min(strandsInBundle, std::max(1u, static_cast<uint32_t>(v)));
}

float strandThicknessScale(float L, uint32_t strandsInBundle) {
    if (L <= 0.0f) return 1.0f;
    // min(1/L, N_i), not synchronised with the discrete count changes (Section 3.6.1).
    return std::min(1.0f / L, static_cast<float>(std::max<uint32_t>(strandsInBundle, 1u)));
}

int vertexLodExponent(float L, int nsMaxSegments) {
    if (L <= 0.0f || nsMaxSegments <= 0) return 0;
    const float x = std::log2(std::max(L * static_cast<float>(nsMaxSegments), 1e-6f));
    const int e = static_cast<int>(std::ceil(x));
    const int eMax = static_cast<int>(std::log2(static_cast<double>(nsMaxSegments)));
    return std::min(std::max(e, 0), eMax);
}

int previousVertexLodExponent(float L, int nsMaxSegments, float lambda) {
    if (L <= 0.0f || nsMaxSegments <= 0) return 0;
    const float x = std::log2(std::max(L * static_cast<float>(nsMaxSegments), 1e-6f));
    const int e = static_cast<int>(std::ceil(x - lambda));
    const int eMax = static_cast<int>(std::log2(static_cast<double>(nsMaxSegments)));
    return std::min(std::max(e, 0), eMax);
}

float vertexMorphGamma(float L, int nsMaxSegments, float lambda) {
    if (L <= 0.0f || nsMaxSegments <= 0 || lambda <= 0.0f) return 0.0f;
    const float x = std::log2(std::max(L * static_cast<float>(nsMaxSegments), 1e-6f));
    const int e = vertexLodExponent(L, nsMaxSegments);
    const int ePrev = previousVertexLodExponent(L, nsMaxSegments, lambda);
    if (e <= ePrev) return 0.0f;  // no transition in progress
    // Eq. 12: gamma = 1 - (log2(L*ns) - e~)/lambda
    return clampf(1.0f - (x - static_cast<float>(ePrev)) / lambda, 0.0f, 1.0f);
}

VertexLodResult computeVertexLod(float L, int nsMaxSegments, float lambda) {
    VertexLodResult r;
    r.exponentE = vertexLodExponent(L, nsMaxSegments);
    r.exponentPrev = previousVertexLodExponent(L, nsMaxSegments, lambda);
    r.gamma = vertexMorphGamma(L, nsMaxSegments, lambda);
    return r;
}

StrandLodResult computeStrandLod(uint32_t strandsInBundle, float delta, float L,
                                 uint32_t minStrands) {
    StrandLodResult r;
    r.numStrands = std::max(minStrands, strandCountLod(strandsInBundle, delta, L));
    r.thicknessScale = strandThicknessScale(L, strandsInBundle);
    return r;
}

float bundleDelta(uint32_t bundleIndex, uint32_t seed) {
    const uint32_t h = pcgHash(bundleIndex * 2654435761u + seed * 40503u + 17u);
    return clampf(hashToFloat(h), 0.0f, 0.9999f);
}

float lodDistance(Vec3 cameraPosition, Vec3 center, float radius) {
    const float d = length(cameraPosition - center) - radius;
    return std::max(d, 1e-3f);
}

}  // namespace silk
