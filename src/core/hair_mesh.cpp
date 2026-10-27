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

void HairMesh::setAreaWeightedStrandCount(uint32_t total) {
    const size_t n = faces_.size();
    bundleStrandCounts_.assign(n, 0);
    if (n == 0 || total == 0) return;

    std::vector<double> area(n, 0.0);
    double areaSum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const HairMeshFace& f = faces_[i];
        Vec3 c(0, 0, 0);
        for (uint32_t k = 0; k < f.count; ++k) c = c + rootVertex(f.v[k]);
        c = c * (1.0f / static_cast<float>(f.count));
        double a = 0.0;
        for (uint32_t k = 0; k < f.count; ++k) {
            const Vec3& p0 = rootVertex(f.v[k]);
            const Vec3& p1 = rootVertex(f.v[(k + 1) % f.count]);
            a += 0.5 * static_cast<double>(length(cross(p0 - c, p1 - c)));
        }
        area[i] = a;
        areaSum += a;
    }
    if (areaSum <= 0.0) {
        setUniformStrandCount(std::max(1u, total / static_cast<uint32_t>(n)));
        return;
    }

    // Largest-remainder allocation: exact total, and every face above a twentieth of the
    // mean area gets at least one strand so no hole opens in the surface.
    std::vector<double> exact(n, 0.0);
    std::vector<uint32_t> count(n, 0);
    const double meanArea = areaSum / static_cast<double>(n);
    uint32_t assigned = 0;
    for (size_t i = 0; i < n; ++i) {
        exact[i] = static_cast<double>(total) * area[i] / areaSum;
        count[i] = static_cast<uint32_t>(exact[i]);
        if (count[i] == 0 && area[i] > 0.05 * meanArea) count[i] = 1;
        assigned += count[i];
    }
    struct Remainder { double frac; size_t index; };
    std::vector<Remainder> rem;
    rem.reserve(n);
    for (size_t i = 0; i < n; ++i) rem.push_back({exact[i] - std::floor(exact[i]), i});
    std::sort(rem.begin(), rem.end(), [](const Remainder& a, const Remainder& b) {
        return a.frac > b.frac;
    });
    size_t cursor = 0;
    while (assigned < total && cursor < rem.size()) { ++count[rem[cursor++].index]; ++assigned; }
    while (assigned > total) {
        // Take back from the biggest bundles, which can spare one strand.
        size_t big = 0;
        for (size_t i = 1; i < n; ++i) if (count[i] > count[big]) big = i;
        if (count[big] <= 1) break;
        --count[big];
        --assigned;
    }
    bundleStrandCounts_ = std::move(count);
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
