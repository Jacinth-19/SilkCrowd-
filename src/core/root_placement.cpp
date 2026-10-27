#include "silkcrowd/root_placement.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
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

// ---------------------------------------------------------------------------
//  Baked sample sets (see the header for the file layout)
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kSampleSetVersion = 1u;
constexpr char kSampleSetMagic[4] = {'S', 'C', 'B', 'N'};
constexpr uint32_t kFlagTriangle = 1u;
constexpr uint32_t kMaxSamples = 1u << 20;

SampleSetStats g_stats;
std::mutex g_cacheMutex;
std::map<uint64_t, SampleSet2D> g_cache;

inline uint32_t floatBits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
inline float bitsFloat(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

void hashBytes(uint64_t& h, const void* data, size_t bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;  // FNV-1a
    }
}

bool sameParams(const SampleEliminationParams& a, const SampleEliminationParams& b) {
    return a.targetCount == b.targetCount && a.oversample == b.oversample &&
           a.weightExponent == b.weightExponent && a.seed == b.seed &&
           a.triangleDomain == b.triangleDomain;
}

// A loaded set is only trusted if it is well formed: the right count, finite coordinates in the
// unit square (and in the triangle when the set says so).
bool validateSamples(const SampleSet2D& set, const SampleEliminationParams& p) {
    if (set.samples.size() != std::min<size_t>(p.targetCount, kMaxSamples)) return false;
    if (set.triangleDomain != p.triangleDomain) return false;
    for (const Vec2& s : set.samples) {
        if (!std::isfinite(s.x) || !std::isfinite(s.y)) return false;
        if (s.x < 0.0f || s.x > 1.0f || s.y < 0.0f || s.y > 1.0f) return false;
        if (p.triangleDomain && s.x + s.y > 1.0001f) return false;
    }
    return true;
}

}  // namespace

uint64_t sampleSetKey(const SampleEliminationParams& p) {
    uint64_t h = 1469598103934665603ull;
    hashBytes(h, &p.targetCount, sizeof(p.targetCount));
    hashBytes(h, &p.oversample, sizeof(p.oversample));
    const uint32_t we = floatBits(p.weightExponent);
    hashBytes(h, &we, sizeof(we));
    hashBytes(h, &p.seed, sizeof(p.seed));
    const unsigned char tri = p.triangleDomain ? 1u : 0u;
    hashBytes(h, &tri, sizeof(tri));
    return h;
}

std::string sampleSetPath(const SampleEliminationParams& p, const std::string& cacheDir) {
    char name[64];
    std::snprintf(name, sizeof(name), "bn_%016llx.scbn",
                  static_cast<unsigned long long>(sampleSetKey(p)));
    if (cacheDir.empty()) return std::string(name);
    std::string dir = cacheDir;
    if (dir.back() != '/') dir.push_back('/');
    return dir + name;
}

bool writeSampleSet(const SampleSet2D& set, const SampleEliminationParams& p,
                    const std::string& path) {
    if (set.samples.size() > kMaxSamples) return false;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good()) return false;
    const uint32_t header[7] = {kSampleSetVersion,
                                static_cast<uint32_t>(p.targetCount),
                                static_cast<uint32_t>(p.oversample),
                                floatBits(p.weightExponent),
                                p.seed,
                                p.triangleDomain ? kFlagTriangle : 0u,
                                static_cast<uint32_t>(set.samples.size())};
    out.write(kSampleSetMagic, sizeof(kSampleSetMagic));
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    // Two float32 per sample, explicitly little-endian so a big-endian host writes a file that
    // still round-trips (it would fail the magic check rather than silently decode).
    for (const Vec2& s : set.samples) {
        const float xy[2] = {s.x, s.y};
        out.write(reinterpret_cast<const char*>(xy), sizeof(xy));
    }
    out.flush();
    return out.good();
}

bool readSampleSet(SampleSet2D& set, const SampleEliminationParams& p, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) return false;
    char magic[4] = {0, 0, 0, 0};
    in.read(magic, sizeof(magic));
    if (!in.good() || std::memcmp(magic, kSampleSetMagic, sizeof(magic)) != 0) {
        ++g_stats.rejectedFiles;
        return false;
    }
    uint32_t header[7] = {0, 0, 0, 0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in.good()) {
        ++g_stats.rejectedFiles;
        return false;
    }
    SampleEliminationParams stored;
    stored.targetCount = header[1];
    stored.oversample = header[2];
    stored.weightExponent = bitsFloat(header[3]);
    stored.seed = header[4];
    stored.triangleDomain = (header[5] & kFlagTriangle) != 0u;
    const uint32_t count = header[6];
    if (header[0] != kSampleSetVersion || count == 0u || count > kMaxSamples ||
        !sameParams(stored, p)) {
        ++g_stats.rejectedFiles;
        return false;
    }
    SampleSet2D loaded;
    loaded.triangleDomain = stored.triangleDomain;
    loaded.samples.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        float xy[2] = {0.0f, 0.0f};
        in.read(reinterpret_cast<char*>(xy), sizeof(xy));
        if (!in.good()) {
            ++g_stats.rejectedFiles;
            return false;
        }
        loaded.samples[i] = Vec2(xy[0], xy[1]);
    }
    if (!validateSamples(loaded, p)) {
        ++g_stats.rejectedFiles;
        return false;
    }
    set = std::move(loaded);
    ++g_stats.loadedFromDisk;
    return true;
}

const SampleSet2D& blueNoiseSet(const SampleEliminationParams& p, const std::string& cacheDir) {
    const uint64_t key = sampleSetKey(p);
    std::lock_guard<std::mutex> lock(g_cacheMutex);
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    SampleSet2D set;
    bool have = false;
    if (!cacheDir.empty()) have = readSampleSet(set, p, sampleSetPath(p, cacheDir));
    if (!have) {
        const auto start = std::chrono::steady_clock::now();
        set = generateBlueNoiseSamples(p);
        const auto end = std::chrono::steady_clock::now();
        g_stats.millisecondsGenerating +=
            std::chrono::duration<double, std::milli>(end - start).count();
        ++g_stats.generated;
        if (!cacheDir.empty() && !writeSampleSet(set, p, sampleSetPath(p, cacheDir))) {
            ++g_stats.writeFailures;  // read-only install: regenerate next time, no error
        }
    }
    return g_cache.emplace(key, std::move(set)).first->second;
}

SampleEliminationParams sampleSetPreset(NoiseSet which) {
    SampleEliminationParams p;
    switch (which) {
        case NoiseSet::FastQuad:
            p.targetCount = 1024;
            break;
        case NoiseSet::FastTriangle:
            p.targetCount = 1024;
            p.triangleDomain = true;
            p.seed = 987654u;
            break;
        case NoiseSet::ReferenceQuad:
            p.targetCount = 1024;
            p.oversample = 3;
            p.seed = 987654321u;
            break;
        case NoiseSet::ReferenceTriangle:
            p.targetCount = 1024;
            p.oversample = 3;
            p.seed = 192837465u;
            p.triangleDomain = true;
            break;
        case NoiseSet::GpuQuad:
            p.targetCount = 2048;
            break;
        case NoiseSet::GpuTriangle:
            p.targetCount = 2048;
            p.triangleDomain = true;
            p.seed = 987654321u;
            break;
        case NoiseSet::Count:
            break;
    }
    return p;
}

const char* sampleSetPresetName(NoiseSet which) {
    switch (which) {
        case NoiseSet::FastQuad: return "fast CPU path, quad domain";
        case NoiseSet::FastTriangle: return "fast CPU path, triangle domain";
        case NoiseSet::ReferenceQuad: return "reference software renderer, quad domain";
        case NoiseSet::ReferenceTriangle: return "reference software renderer, triangle domain";
        case NoiseSet::GpuQuad: return "Vulkan path, quad domain";
        case NoiseSet::GpuTriangle: return "Vulkan path, triangle domain";
        case NoiseSet::Count: break;
    }
    return "unknown";
}

const SampleSet2D& presetSet(NoiseSet which, const std::string& cacheDir) {
    return blueNoiseSet(sampleSetPreset(which), cacheDir);
}

int writeBakedSampleSets(const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    int written = 0;
    for (int i = 0; i < static_cast<int>(NoiseSet::Count); ++i) {
        const NoiseSet which = static_cast<NoiseSet>(i);
        const SampleEliminationParams p = sampleSetPreset(which);
        const SampleSet2D& set = presetSet(which);
        if (writeSampleSet(set, p, sampleSetPath(p, dir))) ++written;
    }
    return written;
}

const SampleSetStats& sampleSetStats() { return g_stats; }

void resetSampleSetStats() {
    // Counters only: see the header. `g_cache` is deliberately left alone.
    std::lock_guard<std::mutex> lock(g_cacheMutex);
    g_stats = SampleSetStats{};
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
