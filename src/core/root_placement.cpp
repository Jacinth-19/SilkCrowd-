#include "silkcrowd/root_placement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <sstream>

namespace silk {
namespace {

struct Candidate {
    Vec2 p;
    std::vector<int> neighbors;
    float weight = 0.0f;
    bool alive = true;
    uint64_t version = 0;
};

// Uniform grid for neighbour queries inside the unit square.
struct Grid {
    int n = 1;
    float cell = 1.0f;
    std::vector<std::vector<int>> cells;
    explicit Grid(float radius, int count) {
        n = std::max(1, static_cast<int>(std::ceil(1.0f / std::max(radius, 1e-4f))));
        n = std::min(n, 4096);
        cell = 1.0f / static_cast<float>(n);
        cells.resize(static_cast<size_t>(n) * n);
        (void)count;
    }
    int ix(float x) const { return std::min(n - 1, std::max(0, static_cast<int>(x / cell))); }
    void insert(int id, Vec2 p) {
        cells[static_cast<size_t>(ix(p.y)) * n + ix(p.x)].push_back(id);
    }
    template <typename F>
    void query(Vec2 p, float radius, F&& fn) const {
        const int r = std::max(1, static_cast<int>(std::ceil(radius / cell)));
        const int cx = ix(p.x), cy = ix(p.y);
        for (int y = std::max(0, cy - r); y <= std::min(n - 1, cy + r); ++y)
            for (int x = std::max(0, cx - r); x <= std::min(n - 1, cx + r); ++x)
                for (int id : cells[static_cast<size_t>(y) * n + x]) fn(id);
    }
};

inline float smin(float a, float b) { return a < b ? a : b; }

}  // namespace

SampleSet2D generateBlueNoiseSamples(const SampleEliminationParams& params) {
    // ---- 1. random candidates (uniform over the square or the lower triangle) ----
    const size_t M = std::max<size_t>(params.targetCount, 8) * std::max<size_t>(params.oversample, 1);
    std::vector<Candidate> cand(M);
    uint32_t state = params.seed * 2654435761u + 1u;
    auto nextRand = [&state]() {
        state = state * 1664525u + 1013904223u;
        return (state >> 8) * (1.0f / 16777216.0f);
    };
    for (size_t i = 0; i < M; ++i) {
        Vec2 p(nextRand(), nextRand());
        if (params.triangleDomain) {
            // Rejection-sample the triangular half xi_s + xi_t < 1.
            while (p.x + p.y >= 0.999f) {
                p.x = nextRand();
                p.y = nextRand();
            }
        } else {
            p.x = clampf(p.x, 1e-4f, 1.0f - 1e-4f);
            p.y = clampf(p.y, 1e-4f, 1.0f - 1e-4f);
        }
        cand[i].p = p;
    }

    // ---- 2. neighbour lists within r_max ----
    // For a Poisson disk of N points in a unit area the mean radius is
    // r = sqrt(2*A/(sqrt(3)*N)); we use a slightly larger interaction radius.
    const double area = params.triangleDomain ? 0.5 : 1.0;
    const float rMean = static_cast<float>(std::sqrt(2.0 * area / (std::sqrt(3.0) * std::max<size_t>(params.targetCount, 1))));
    const float rMax = std::min(0.5f, rMean * 2.0f * std::sqrt(static_cast<float>(std::max<size_t>(params.oversample, 1))));

    Grid grid(std::max(rMax, 1e-3f), static_cast<int>(M));
    for (size_t i = 0; i < M; ++i) grid.insert(static_cast<int>(i), cand[i].p);
    for (size_t i = 0; i < M; ++i) {
        grid.query(cand[i].p, rMax, [&](int j) {
            if (static_cast<size_t>(j) <= i) return;
            const float d = length(cand[j].p - cand[i].p);
            if (d < rMax && d > 1e-9f) {
                cand[i].neighbors.push_back(j);
                cand[j].neighbors.push_back(static_cast<int>(i));
            }
        });
    }

    auto computeWeight = [&](Candidate& c) {
        float w = 0.0f;
        for (int j : c.neighbors) {
            if (!cand[j].alive) continue;
            const float d = length(cand[j].p - c.p);
            if (d < rMax) {
                const float t = 1.0f - d / rMax;
                if (params.weightExponent == 8.0f) {
                    const float t2 = t * t, t4 = t2 * t2;
                    w += t4 * t4;  // (1 - d/r_max)^8 without a transcendental call
                } else {
                    w += std::pow(t, params.weightExponent);
                }
            }
        }
        c.weight = w;
        ++c.version;
    };
    for (size_t i = 0; i < M; ++i) computeWeight(cand[i]);

    // ---- 3. progressive elimination down to one sample ----
    struct HeapItem {
        float weight;
        int id;
        uint64_t version;
        bool operator<(const HeapItem& o) const { return weight < o.weight; }
    };
    std::priority_queue<HeapItem> heap;
    for (size_t i = 0; i < M; ++i) heap.push({cand[i].weight, static_cast<int>(i), cand[i].version});

    std::vector<int> removalOrder;
    removalOrder.reserve(M);
    int alive = static_cast<int>(M);
    while (alive > 1 && !heap.empty()) {
        const HeapItem top = heap.top();
        heap.pop();
        Candidate& c = cand[top.id];
        if (!c.alive || top.version != c.version) continue;  // stale entry
        c.alive = false;
        removalOrder.push_back(top.id);
        --alive;
        // Only neighbours' weights change.
        for (int j : c.neighbors) {
            if (!cand[j].alive) continue;
            computeWeight(cand[j]);
            heap.push({cand[j].weight, j, cand[j].version});
        }
        // Refresh this candidate's own neighbours lazily by recomputing on demand.
    }

    // ---- 4. progressive set: reverse removal order gives increasing sample counts ----
    SampleSet2D out;
    out.triangleDomain = params.triangleDomain;
    out.samples.reserve(params.targetCount);
    const size_t keep = std::min(params.targetCount, removalOrder.size());
    for (size_t i = 0; i < keep; ++i) {
        const int id = removalOrder[removalOrder.size() - 1 - i];
        out.samples.push_back(cand[id].p);
    }
    return out;
}

BlueNoiseMetrics measureBlueNoise(const std::vector<Vec2>& pts) {
    BlueNoiseMetrics m;
    const size_t n = pts.size();
    if (n < 2) return m;
    double sum = 0.0, sumSq = 0.0;
    double mn = 1e9, mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double best = 1e9;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const double d = length(pts[i] - pts[j]);
            best = std::min(best, d);
        }
        sum += best;
        sumSq += best * best;
        mn = std::min(mn, best);
        mx = std::max(mx, best);
    }
    m.meanNearestNeighbor = sum / static_cast<double>(n);
    m.varianceNN = sumSq / static_cast<double>(n) - m.meanNearestNeighbor * m.meanNearestNeighbor;
    m.minNN = mn;
    m.maxNN = mx;
    // Expected nearest-neighbour distance for white noise in a unit square is ~0.5/sqrt(n).
    m.expectedWhiteNoise = 0.5 / std::sqrt(static_cast<double>(n));
    // Quality: blue noise maximizes the mean NN distance while keeping it regular
    // (low variance). Normalize both into [0,1].
    const double spacingGain = m.meanNearestNeighbor / std::max(m.expectedWhiteNoise, 1e-9);
    const double regularity = 1.0 - std::min(1.0, std::sqrt(std::max(m.varianceNN, 0.0)) /
                                                      std::max(m.meanNearestNeighbor, 1e-9));
    m.radialDistributionScore = clampf(static_cast<float>(0.5 * (spacingGain / 1.5) + 0.5 * regularity), 0.0f, 1.0f);
    return m;
}

std::string BlueNoiseMetrics::toString() const {
    std::ostringstream os;
    os << "mean NN distance " << meanNearestNeighbor << " (white noise " << expectedWhiteNoise
       << "), NN std-dev " << std::sqrt(std::max(varianceNN, 0.0)) << ", range [" << minNN
       << ", " << maxNN << "], score " << radialDistributionScore;
    return os.str();
}

Vec2 rootPlacementTriangle(Vec2 xi, TriangleMapping mapping) {
    // xi lives in the triangular half (xi_s + xi_t < 1).
    const float s = clampf(xi.x, 0.0f, 1.0f);
    const float t = clampf(xi.y, 0.0f, 1.0f);
    const float denom = std::max(1.0f - t, 1e-5f);
    if (mapping == TriangleMapping::PaperLiteral) {
        return Vec2(s * (1.0f - t), t);
    }
    return Vec2(s / denom, t);
}

void triangleBarycentricsFromBlock(Vec2 ab, float* b0, float* b1, float* b2) {
    // Block layout for a triangle: (0,0)=v0, (1,0)=v1, (0,1)=v2, (1,1)=v2 (duplicated).
    // Bilinear weights: v1 -> a(1-b), v2 -> b, v0 -> (1-a)(1-b).
    const float a = saturate(ab.x);
    const float b = saturate(ab.y);
    *b0 = (1.0f - a) * (1.0f - b);
    *b1 = a * (1.0f - b);
    *b2 = b;
}

}  // namespace silk
