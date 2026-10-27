// On-the-fly strand generation (paper Section 3.2-3.6).
//
// This is the CPU reference implementation of the exact computation that the task/mesh
// shaders (shaders/hair.task / hair.mesh) perform; the texture fetches, the de Casteljau
// evaluation, the styling and the vertex-LOD morphing are written to mirror the GLSL
// line for line so that validation on the CPU transfers to the GPU path.
#pragma once

#include <cstdint>
#include <vector>

#include "silkcrowd/hair_mesh_texture.h"
#include "silkcrowd/lod.h"
#include "silkcrowd/styling.h"

namespace silk {

struct StrandVertex {
    Vec3 position;
    Vec3 tangent;
    float w = 0.0f;  // w styling coordinate, used for shading (Kajiya-Kay shift etc.)
};

struct StrandRequest {
    uint32_t face = 0;        // bundle (= root face)
    Vec2 blockCoord{0.5f, 0.5f};  // root position inside the 2x2 block
    uint32_t strandIndex = 0; // local strand index inside the bundle
    uint32_t bundleIndex = 0; // global bundle index (for per-bundle randomness)
    float thicknessScale = 1.0f;  // strand-LOD thickness compensation (Eq. 8 discussion)
};

struct StrandGeometry {
    std::vector<StrandVertex> vertices;  // 2^e + 1 entries
    float radius = 0.001f;
};

// Eq. 5 and Eq. 6: tangents from the previous/next vertex, with an endpoint rule that
// keeps shading continuous across vertex-LOD changes.
void computeStrandTangents(const Vec3* positions, int count, Vec3* tangents);

// Tangents computed from the even-index vertices only (t~ in Section 3.6.2): even
// vertices use the coarse tangents, odd vertices the average of their neighbours.
void computeEvenTangents(const Vec3* positions, int count, Vec3* tangents);

class StrandGenerator {
public:
    StrandGenerator(const HairMeshTextureSet& textures, const StylingParams& styling,
                    float baseRadius, float strandLengthScale = 1.0f);

    // Generates one strand at the given vertex detail level, applying the Eq. 11/12
    // morphing when gamma > 0.
    void generate(const StrandRequest& req, const VertexLodResult& lod, StrandGeometry* out) const;

    // Convenience: computes the vertex LOD from a detail level and generates.
    void generateWithDetailLevel(const StrandRequest& req, float detailLevel, LodParams params,
                                 StrandGeometry* out) const;

    const HairMeshTextureSet& textures() const { return *textures_; }
    const StylingParams& styling() const { return *styling_; }
    float baseRadius() const { return baseRadius_; }

private:
    const HairMeshTextureSet* textures_;
    const StylingParams* styling_;
    float baseRadius_;
    float strandLengthScale_;
};

}  // namespace silk
