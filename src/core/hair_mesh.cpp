#include "silkcrowd/hair_mesh.h"

#include <algorithm>
#include <cmath>

namespace silk {

void HairMesh::setTopology(uint32_t numVertices, uint32_t numLayers,
                           std::vector<HairMeshFace> faces, std::vector<Vec2> rootUV) {
    numVertices_ = numVertices;
    numLayers_ = numLayers;
    faces_ = std::move(faces);
    rootUV_ = std::move(rootUV);
    rootUV_.resize(numVertices, Vec2(0, 0));
    positions_.assign(static_cast<size_t>(numVertices) * numLayers, Vec3(0, 0, 0));
    layerW_ = defaultLayerW(numLayers);
    scalpNormals_.assign(numVertices, Vec3(0, 1, 0));
    bundleStrandCounts_.assign(faces_.size(), 0);
}

std::vector<Vec3> HairMesh::vertexCurve(uint32_t v) const {
    std::vector<Vec3> out(numLayers_);
    for (uint32_t l = 0; l < numLayers_; ++l) out[l] = layerVertex(l, v);
    return out;
}

void HairMesh::setUniformStrandCount(uint32_t perBundle) {
    bundleStrandCounts_.assign(faces_.size(), perBundle);
}

void HairMesh::computeScalpNormals() {
    scalpNormals_.assign(numVertices_, Vec3(0, 0, 0));
    for (const HairMeshFace& f : faces_) {
        // Newell's method for the root-layer polygon normal.
        Vec3 n(0, 0, 0);
        for (uint32_t i = 0; i < f.count; ++i) {
            const Vec3& a = rootVertex(f.v[i]);
            const Vec3& b = rootVertex(f.v[(i + 1) % f.count]);
            n.x += (a.y - b.y) * (a.z + b.z);
            n.y += (a.z - b.z) * (a.x + b.x);
            n.z += (a.x - b.x) * (a.y + b.y);
        }
        for (uint32_t i = 0; i < f.count; ++i) scalpNormals_[f.v[i]] += n;
    }
    for (Vec3& n : scalpNormals_) {
        if (length2(n) < 1e-20f) n = Vec3(0, 1, 0);
        n = normalize(n);
    }
}

void HairMesh::bounds(Vec3* lo, Vec3* hi) const {
    Vec3 l(1e30f), h(-1e30f);
    for (const Vec3& p : positions_) {
        l = minv(l, p);
        h = maxv(h, p);
    }
    *lo = l;
    *hi = h;
}

std::vector<float> defaultLayerW(uint32_t numLayers) {
    std::vector<float> w(numLayers, 0.0f);
    if (numLayers <= 1) return w;
    for (uint32_t i = 0; i < numLayers; ++i)
        w[i] = static_cast<float>(i) / static_cast<float>(numLayers - 1);
    return w;
}

}  // namespace silk
