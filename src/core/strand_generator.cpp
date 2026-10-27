#include "silkcrowd/strand_generator.h"

#include <algorithm>
#include <cmath>

namespace silk {

void computeStrandTangents(const Vec3* positions, int count, Vec3* tangents) {
    if (count < 2) {
        if (count == 1) tangents[0] = Vec3(0, 1, 0);
        return;
    }
    // Section 3.5: t_k = p_(k+1) - p_(k-1), with special endpoint rules.
    for (int k = 1; k + 1 < count; ++k) tangents[k] = positions[k + 1] - positions[k - 1];

    const Vec3 qLast = normalize(positions[count - 1] - positions[count - 2]);
    const Vec3 tPrev = normalize(tangents[count - 2]);
    // Eq. 5:  t_hat_ns = 2 q_hat_ns (t_hat_(ns-1) . q_hat_ns) - t_hat_(ns-1)
    tangents[count - 1] = normalize(qLast * (2.0f * dot(tPrev, qLast)) - tPrev);

    const Vec3 qFirst = normalize(positions[1] - positions[0]);
    const Vec3 tNext = normalize(tangents[1]);
    // Eq. 6:  t_hat_0 = 2 q_hat_1 (t_hat_1 . q_hat_1) - t_hat_1
    tangents[0] = normalize(qFirst * (2.0f * dot(tNext, qFirst)) - tNext);
}

void computeEvenTangents(const Vec3* positions, int count, Vec3* tangents) {
    if (count < 2) {
        tangents[0] = Vec3(0, 1, 0);
        return;
    }
    const int coarseCount = (count - 1) / 2 + 1;  // indices 0, 2, 4, ...
    std::vector<Vec3> coarse(static_cast<size_t>(coarseCount));
    for (int i = 0; i < coarseCount; ++i) coarse[i] = positions[i * 2];
    std::vector<Vec3> coarseTangents(static_cast<size_t>(coarseCount));
    computeStrandTangents(coarse.data(), coarseCount, coarseTangents.data());
    for (int k = 0; k < count; ++k) {
        if ((k & 1) == 0) {
            tangents[k] = coarseTangents[k / 2];
        } else {
            // t~_k for odd k = average(t~_(k-1) + t~_(k+1))
            const Vec3 a = (k - 1 >= 0) ? coarseTangents[(k - 1) / 2] : coarseTangents[0];
            const Vec3 b = (k + 1 < count) ? coarseTangents[(k + 1) / 2] : coarseTangents[coarseCount - 1];
            tangents[k] = normalize(a + b);
        }
    }
}

StrandGenerator::StrandGenerator(const HairMeshTextureSet& textures, const StylingParams& styling,
                                 float baseRadius, float strandLengthScale)
    : textures_(&textures),
      styling_(&styling),
      baseRadius_(baseRadius),
      strandLengthScale_(strandLengthScale) {}

void StrandGenerator::generate(const StrandRequest& req, const VertexLodResult& lod,
                               StrandGeometry* out) const {
    const HairMeshTextureSet& tex = *textures_;
    const StylingParams& sp = *styling_;
    const uint32_t layers = tex.numLayers();
    const int segs = lod.numSegments();
    const int count = segs + 1;

    out->vertices.resize(static_cast<size_t>(count));

    const Vec2 st = tex.layout().blockToSlice(req.face, req.blockCoord);
    const Vec2 uv = tex.sampleUV(st);

    StrandStylingState state =
        makeStrandStylingState(sp, uv, req.strandIndex, req.bundleIndex);

    std::vector<Vec3> raw(static_cast<size_t>(count));
    for (int k = 0; k < count; ++k) {
        const float v = static_cast<float>(k) / static_cast<float>(segs);
        // Uniform parameter along the hair mesh extrusion; layer slices are evenly spaced
        // in the hair mesh texture's r* direction.
        const float t = v * static_cast<float>(layers - 1) * state.lengthScale;

        Vec3 p;
        float w;
        if (k == 0) {
            // Root vertex: a single lookup on the root slice, no styling (Section 3.2).
            p = tex.sampleSlice(st, 0.0f);
            w = tex.sampleW(st, 0.0f);
        } else {
            // 2 trilinear fetches + 1 linear interpolation (Figure 3e).
            p = tex.evalCurve(st, t);
            w = tex.sampleW(st, t);
            const Mat3 Mp = tex.stylingMatrix(st, t);
            p = styleStrandVertex(sp, state, p, w, k, segs, Mp);
        }
        raw[static_cast<size_t>(k)] = p;
    }

    // Eq. 11: p'_k = (1-gamma) p_k + gamma (p_(k-1) + p_(k+1))/2 for odd k.
    std::vector<Vec3> morphed = raw;
    if (lod.gamma > 0.0f) {
        for (int k = 1; k + 1 < count; k += 2) {
            const Vec3 avg = (raw[static_cast<size_t>(k - 1)] + raw[static_cast<size_t>(k + 1)]) * 0.5f;
            morphed[static_cast<size_t>(k)] = lerp(raw[static_cast<size_t>(k)], avg, lod.gamma);
        }
    }

    // Tangents from the original positions, and from the even vertices only, then
    // morphed with the same gamma so that shading transitions continuously
    // (Section 3.6.2, Figure 6).
    std::vector<Vec3> tanFull(static_cast<size_t>(count)), tanEven(static_cast<size_t>(count));
    computeStrandTangents(raw.data(), count, tanFull.data());
    computeEvenTangents(raw.data(), count, tanEven.data());

    for (int k = 0; k < count; ++k) {
        Vec3 tangent = tanFull[static_cast<size_t>(k)];
        if (lod.gamma > 0.0f) {
            tangent = normalize(lerp(tangent, tanEven[static_cast<size_t>(k)], lod.gamma));
        }
        StrandVertex& sv = out->vertices[static_cast<size_t>(k)];
        sv.position = morphed[static_cast<size_t>(k)];
        sv.tangent = tangent;
        sv.w = static_cast<float>(k) / static_cast<float>(segs);
    }

    out->radius = baseRadius_ * sp.thickness * state.thicknessScale * req.thicknessScale;
}

void StrandGenerator::generateWithDetailLevel(const StrandRequest& req, float L, LodParams params,
                                              StrandGeometry* out) const {
    (void)params;
    const int nsMax = 32;  // full-resolution segments; 2^5 -> 33 vertices per strand
    VertexLodResult lod = computeVertexLod(L, nsMax, params.lambda);
    generate(req, lod, out);
}

}  // namespace silk
