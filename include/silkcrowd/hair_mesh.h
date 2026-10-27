// Hair mesh data structure (paper Section 2.2, Figure 2).
//
// A hair mesh is a volumetric structure formed by extruding the polygonal faces of a
// scalp model. Each root face becomes one *bundle*. A strand root is a barycentric
// coordinate on a root face; its other control points are the same barycentric
// coordinate on successive extrusion layers of that same face.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "silkcrowd/math.h"

namespace silk {

struct HairMeshFace {
    uint32_t v[4] = {0, 0, 0, 0};
    uint32_t count = 4;  // 3 or 4 (quad-dominant)
    bool isTriangle() const { return count == 3; }
};

class HairMesh {
public:
    HairMesh() = default;

    uint32_t numLayers() const { return numLayers_; }
    uint32_t numVertices() const { return numVertices_; }
    uint32_t numFaces() const { return static_cast<uint32_t>(faces_.size()); }

    void setTopology(uint32_t numVertices, uint32_t numLayers, std::vector<HairMeshFace> faces,
                     std::vector<Vec2> rootUV);

    const HairMeshFace& face(uint32_t f) const { return faces_[f]; }
    const std::vector<HairMeshFace>& faces() const { return faces_; }

    const Vec3& layerVertex(uint32_t layer, uint32_t v) const {
        return positions_[static_cast<size_t>(layer) * numVertices_ + v];
    }
    Vec3& layerVertexMutable(uint32_t layer, uint32_t v) {
        return positions_[static_cast<size_t>(layer) * numVertices_ + v];
    }
    const Vec3& rootVertex(uint32_t v) const { return layerVertex(0, v); }

    // Extrusion layers of one vertex, root -> tip. Used by the curve conversion where
    // "the vertices of the hair mesh on successive extrusions over the same root vertex
    // are connected using the same curve formulation" (Section 2.2).
    std::vector<Vec3> vertexCurve(uint32_t v) const;

    const std::vector<Vec2>& rootUV() const { return rootUV_; }
    std::vector<Vec2>& rootUVMutable() { return rootUV_; }

    // w styling coordinate per layer: w = 0 at the root layer, increasing towards the tip.
    const std::vector<float>& layerW() const { return layerW_; }
    void setLayerW(std::vector<float> w) { layerW_ = std::move(w); }

    // Object-space scalp normal at each root vertex (used to build the initial extrusion
    // and to orient strand roots).
    const std::vector<Vec3>& scalpNormals() const { return scalpNormals_; }
    void setScalpNormals(std::vector<Vec3> n) { scalpNormals_ = std::move(n); }

    // Per-bundle strand counts (uniform across bundles unless set explicitly).
    const std::vector<uint32_t>& bundleStrandCounts() const { return bundleStrandCounts_; }
    void setUniformStrandCount(uint32_t perBundle);
    // Distributes `total` strands over the faces in proportion to root-layer area, so the
    // *density* on the scalp is even. A hair mesh rooted on a spherical patch has enormous
    // over-density at the pole (all columns converge there) and under-density on the big
    // faces further down; uniform per-face counts therefore grow fur on the crown and leave
    // the sides sparse. The paper's mesh is hand-authored with even quads, which is the same
    // property this restores for a generated one.
    void setAreaWeightedStrandCount(uint32_t total);

    void computeScalpNormals();

    // Bounding box over all layers.
    void bounds(Vec3* lo, Vec3* hi) const;

    std::string name;

private:
    uint32_t numLayers_ = 0;
    uint32_t numVertices_ = 0;
    std::vector<Vec3> positions_;  // numLayers * numVertices
    std::vector<Vec2> rootUV_;
    std::vector<HairMeshFace> faces_;
    std::vector<float> layerW_;
    std::vector<Vec3> scalpNormals_;
    std::vector<uint32_t> bundleStrandCounts_;
};

// Generates the default layer w coordinates for L layers: 0, 1/(L-1), ..., 1.
std::vector<float> defaultLayerW(uint32_t numLayers);

}  // namespace silk
