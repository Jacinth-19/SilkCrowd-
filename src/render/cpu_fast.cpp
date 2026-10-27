#include "silkcrowd/cpu_fast.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "silkcrowd/lod.h"
#include "silkcrowd/root_placement.h"
#include "silkcrowd/strand_generator.h"
#include "silkcrowd/styling.h"

namespace silk {
namespace {

using clock = std::chrono::high_resolution_clock;

double msSince(clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// A cheap 3D hash for the per-strand colour variation (no <random>, deterministic, and the
// same construction as the reference path's hash13 so the two agree in spirit).
inline float hash3f(Vec3 p) {
    float v = p.x * 0.1031f + p.y * 0.11369f + p.z * 0.13787f;
    v = fmodOne(v) * 43.7f;
    return fmodOne(v * v * 37.13f);
}

// ---------------------------------------------------------------------------
//  Work description produced by the front end
// ---------------------------------------------------------------------------
struct BundleJob {
    uint32_t instance = 0;
    uint32_t face = 0;
    uint32_t strands = 0;     // Eq. 8
    int segments = 0;         // 2^e
    int vertices = 0;         // 2^e + 1
    float gamma = 0.0f;       // Eq. 12
    float thickness = 1.0f;   // Eq. 8 compensation
    float screenX0 = 0, screenY0 = 0, screenX1 = 0, screenY1 = 0;
    float depth = 0.0f;       // sort key: distance from the camera to the bundle centre
    size_t vertexOffset = 0;  // into the arena, in vertices
};

// Runs fn(begin, end, threadIndex) over [0, n) in `threads` contiguous chunks and joins.
// Contiguous chunks (rather than atomic work stealing) keep the depth-sorted order inside a
// chunk, which is what lets the per-thread buffers be composited in order afterwards.
template <typename Fn>
void parallelRanges(size_t n, int threads, Fn fn) {
    if (n == 0) return;
    const int t = std::max(1, std::min<int>(threads, static_cast<int>(n)));
    if (t == 1) {
        fn(size_t(0), n, 0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(t) - 1);
    const size_t chunk = (n + static_cast<size_t>(t) - 1) / static_cast<size_t>(t);
    for (int i = 0; i < t; ++i) {
        const size_t b = chunk * static_cast<size_t>(i);
        const size_t e = std::min(n, b + chunk);
        if (b >= e) continue;
        if (i + 1 == t) {
            fn(b, e, i);
        } else {
            pool.emplace_back([&fn, b, e, i] { fn(b, e, i); });
        }
    }
    for (auto& th : pool) th.join();
}

// Frustum planes from a row-major view-projection matrix (Gribb/Hartmann), normalised so a
// signed distance compares directly against a sphere radius.
struct Frustum {
    Vec3 n[6];
    float d[6] = {0, 0, 0, 0, 0, 0};

    // Gribb/Hartmann: a clip-space condition a . p + a[3] >= 0 becomes a plane in world
    // space once the matrix rows are folded in. Vulkan clip space is -w <= x,y <= w and
    // 0 <= z <= w, hence the six rows below (m3 = the w row).
    void fromViewProj(const Mat4& m) {
        auto row = [&](int i) { return Vec3(m.m[i][0], m.m[i][1], m.m[i][2]); };
        auto add = [&](Vec3 a, Vec3 b) { return a + b; };
        const Vec3 p[6] = {add(row(3), row(0)),   // x >= -w  (left)
                           add(row(3), row(0) * -1.0f),  // x <=  w  (right)
                           add(row(3), row(1)),   // y >= -w  (bottom)
                           add(row(3), row(1) * -1.0f),  // y <=  w  (top)
                           row(2),                // z >= 0   (near)
                           add(row(3), row(2) * -1.0f)};  // z <= w (far)
        const float dd[6] = {m.m[3][3] + m.m[0][3], m.m[3][3] - m.m[0][3],
                             m.m[3][3] + m.m[1][3], m.m[3][3] - m.m[1][3],
                             m.m[2][3], m.m[3][3] - m.m[2][3]};
        for (int i = 0; i < 6; ++i) {
            const float len = length(p[i]);
            if (len > 1e-20f) {
                n[i] = p[i] * (1.0f / len);
                d[i] = dd[i] / len;
            } else {
                n[i] = p[i];
                d[i] = dd[i];
            }
        }
    }
    bool sphereVisible(Vec3 c, float r) const {
        for (int i = 0; i < 6; ++i)
            if (dot(n[i], c) + d[i] < -r) return false;
        return true;
    }
};

// Integer powers by repeated squaring: the specular exponents here are 8 and 4, and a
// library pow() costs more than the rest of the vertex shading put together.
inline float pow2(float x) { return x * x; }
inline float pow4(float x) { return pow2(pow2(x)); }
inline float pow8(float x) { return pow4(pow2(x)); }

// Cheap per-vertex hair shading for the timing path: Kajiya-Kay diffuse plus a narrow
// specular lobe, an ambient term and a per-bundle occlusion factor. The reference path adds
// the light-space shadow buffer and the full Marschner BCSDF (src/render/
// software_renderer.cpp); docs/CPU_2MS.md records why that difference does not change the
// amount of geometry this path measures.
inline Vec3 shadeFast(const Scene& scene, Vec3 tangent, Vec3 view, const Vec3& albedo, float ao,
                      float w) {
    const Vec3 T = normalize(tangent);
    const Vec3 V = view;
    const Vec3 L = scene.sunDirection;
    const float TdL = dot(T, L);
    const float TdV = dot(T, V);
    const float sinTL = std::sqrt(std::max(0.0f, 1.0f - TdL * TdL));
    const float sinTV = std::sqrt(std::max(0.0f, 1.0f - TdV * TdV));
    const float wrap = sinTL * sinTV - TdL * TdV;  // Kajiya-Kay
    const float diff = std::max(wrap, 0.0f) * scene.sunIntensity * 0.55f;
    const Vec3 Hh = normalize(L + V);
    const float hd = std::max(0.0f, dot(T, Hh));
    const float spec = pow8(pow8(hd));            // exponent 64, cheap
    const float fres = pow4(1.0f - std::max(0.0f, std::abs(TdV)));
    const Vec3 tint = albedo * 1.7f + Vec3(0.10f, 0.09f, 0.075f);
    Vec3 c = albedo * (scene.sunColor * diff) * ao;
    c += tint * (spec * (0.30f + 0.90f * fres) * ao);
    const float up = 0.5f + 0.5f * T.y;
    const Vec3 amb = lerp(scene.groundColor * 0.55f, scene.skyHorizon * 0.80f, up);
    c += albedo * amb * (0.50f * ao + 0.18f);
    c *= 0.80f + 0.30f * w;  // root-to-tip lightening
    return c;
}

// Section 3.3: one progressive blue-noise set per process (any prefix is blue noise), so
// every bundle and every frame reuses the same table.
const SampleSet2D& quadNoise() {
    static SampleSet2D set = [] {
        SampleEliminationParams p;
        p.targetCount = 1024;
        return generateBlueNoiseSamples(p);
    }();
    return set;
}
const SampleSet2D& triangleNoise() {
    static SampleSet2D set = [] {
        SampleEliminationParams p;
        p.targetCount = 1024;
        p.triangleDomain = true;
        p.seed = 987654u;
        return generateBlueNoiseSamples(p);
    }();
    return set;
}

struct FrameShared {
    int width = 1, height = 1;
    Mat4 viewProj;
    Vec3 camPos;
    Frustum frustum;
};

// Projects a world point to pixel coordinates. Returns false when the point is behind the
// camera (the caller skips that vertex).
inline bool projectToScreen(const Mat4& viewProj, Vec3 p, float* x, float* y) {
    Vec3 clip;
    float w;
    viewProj.transformPoint(p, &clip, &w);
    if (w <= 1e-6f) return false;
    const float inv = 1.0f / w;
    *x = clip.x * inv;
    *y = clip.y * inv;
    return true;
}

void frontEnd(const Scene& scene, const Camera& cam, const RenderOptions& options,
              const FrameShared& fs, std::vector<BundleJob>* jobs, CpuFastStats* st) {
    const float height = static_cast<float>(fs.height);
    const float tanHalf = std::tan(cam.fovY * 0.5f);
    (void)tanHalf;
    jobs->clear();
    for (uint32_t inst = 0; inst < scene.hair.size(); ++inst) {
        const HairInstance& hair = *scene.hair[inst];
        const uint32_t faces = hair.mesh.numFaces();
        const std::vector<uint32_t> counts = hair.mesh.bundleStrandCounts();
        Vec3 lo, hi;
        hair.mesh.bounds(&lo, &hi);
        const float strandLength = std::max(hi.y - lo.y, 1e-3f);
        for (uint32_t f = 0; f < faces; ++f) {
            const Vec3 worldCenter = hair.transform.transformPoint(hair.bundleCenter[f]);
            const float dist = lodDistance(fs.camPos, worldCenter, hair.bundleSize[f] * 0.5f);

            BundleJob job;
            job.instance = inst;
            job.face = f;
            job.strands = counts[f];
            job.thickness = 1.0f;
            job.depth = dist;

            float Ls = 1.0f, Lv = 1.0f;
            if (options.enableLod) {
                if (options.strandLod)
                    Ls = detailLevel(hair.bundleSize[f], height, dist * options.lodBias, cam.fovY,
                                     options.lod.alphaStrand);
                if (options.vertexLod)
                    Lv = detailLevel(strandLength, height, dist * options.lodBias, cam.fovY,
                                     options.lod.alphaVertex);
            }
            if (options.enableLod && options.strandLod) {
                const StrandLodResult r =
                    computeStrandLod(counts[f], bundleDelta(f, hair.seed), Ls);
                job.strands = r.numStrands;
                job.thickness = r.thicknessScale;
            }
            VertexLodResult vl;
            if (options.fixedVertexExponent >= 0) {
                vl.exponentE = std::min(options.fixedVertexExponent, 5);
                vl.exponentPrev = vl.exponentE;
                vl.gamma = 0.0f;
            } else if (options.enableLod && options.vertexLod) {
                vl = computeVertexLod(Lv, 32, options.lod.lambda);
            } else {
                vl = computeVertexLod(1.0f, 32, options.lod.lambda);  // full detail
            }
            job.segments = vl.numSegments();
            job.vertices = vl.numVertices();
            job.gamma = vl.gamma;

            // Bounding sphere of the bundle: half its scalp face plus the strand length.
            const float lengthHalf = 0.5f * strandLength;
            const float radius = std::sqrt(hair.bundleSize[f] * hair.bundleSize[f] * 0.25f +
                                           lengthHalf * lengthHalf);
            const Vec3 sphereCenter =
                worldCenter + hair.transform.transformDirection(Vec3(0.0f, lengthHalf, 0.0f));
            if (!fs.frustum.sphereVisible(sphereCenter, radius) || job.strands == 0) {
                ++st->bundlesCulled;
                continue;
            }
            float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
            bool onScreen = false;
            for (int c = 0; c < 8; ++c) {
                const Vec3 corner = sphereCenter + Vec3(((c & 1) ? radius : -radius),
                                                        ((c & 2) ? radius : -radius),
                                                        ((c & 4) ? radius : -radius));
                float sx, sy;
                if (!projectToScreen(fs.viewProj, corner, &sx, &sy)) continue;
                const float px = (sx * 0.5f + 0.5f) * static_cast<float>(fs.width);
                const float py = (sy * 0.5f + 0.5f) * static_cast<float>(fs.height);
                onScreen = true;
                x0 = std::min(x0, px); x1 = std::max(x1, px);
                y0 = std::min(y0, py); y1 = std::max(y1, py);
            }
            if (!onScreen) {
                ++st->bundlesCulled;
                continue;
            }
            job.screenX0 = std::max(0.0f, x0);
            job.screenY0 = std::max(0.0f, y0);
            job.screenX1 = std::min(static_cast<float>(fs.width), x1);
            job.screenY1 = std::min(static_cast<float>(fs.height), y1);
            const float fw = job.screenX1 - job.screenX0;
            const float fh = job.screenY1 - job.screenY0;
            if (fw < 1.0f || fh < 1.0f) {  // entirely outside, or no pixel of its own
                ++st->bundlesCulled;
                continue;
            }
            jobs->push_back(job);
        }
    }
    std::sort(jobs->begin(), jobs->end(),
              [](const BundleJob& a, const BundleJob& b) { return a.depth > b.depth; });
    size_t cursor = 0;
    for (auto& j : *jobs) {
        j.vertexOffset = cursor;
        cursor += static_cast<size_t>(j.strands) * static_cast<size_t>(j.vertices);
        ++st->bundles;
        st->strands += j.strands;
        st->vertices += static_cast<size_t>(j.strands) * static_cast<size_t>(j.vertices);
        st->segments += static_cast<size_t>(j.strands) * static_cast<size_t>(j.segments);
    }
}

// One strand, exactly as StrandGenerator::generate() computes it: same calls, same order,
// same intermediate values - just written into a fixed array instead of four vectors per
// strand. The unit tests assert that the two agree vertex by vertex.
void generateStrand(const HairInstance& hair, uint32_t face, uint32_t strandIndex, int segments,
                    float gamma, const SampleSet2D& quadSet, const SampleSet2D& triSet,
                    std::vector<Vec3>* out) {
    const HairMeshTextureSet& tex = hair.textures;
    const StylingParams& sp = hair.styling;
    const int n = segments + 1;
    const bool triangle = hair.mesh.face(face).isTriangle();
    const uint32_t layers = tex.numLayers();
    out->resize(static_cast<size_t>(n));

    Vec2 xi = quadSet.at(static_cast<size_t>(strandIndex) + face * 131u);
    if (triangle) xi = rootPlacementTriangle(triSet.at(static_cast<size_t>(strandIndex) + face * 97u));
    const Vec2 block = Vec2(clampf(xi.x, 0.002f, 0.998f), clampf(xi.y, 0.002f, 0.998f));
    const Vec2 st = tex.layout().blockToSlice(face, block);
    const Vec2 uv = tex.sampleUV(st);
    const StrandStylingState state = makeStrandStylingState(sp, uv, strandIndex, face);
    for (int k = 0; k < n; ++k) {
        if (k == 0) {
            (*out)[0] = tex.sampleSlice(st, 0.0f);  // Section 3.2: roots are never styled
            continue;
        }
        const float v = static_cast<float>(k) / static_cast<float>(segments);
        const float t = v * static_cast<float>(layers - 1) * state.lengthScale;
        const Vec3 p = tex.evalCurve(st, t);
        const float w = tex.sampleW(st, t);
        const Mat3 Mp = tex.stylingMatrix(st, t);
        (*out)[static_cast<size_t>(k)] = styleStrandVertex(sp, state, p, w, k, segments, Mp);
    }
    if (gamma > 0.0f) {  // Eq. 11
        for (int k = 1; k + 1 < n; k += 2) {
            const Vec3 avg = ((*out)[static_cast<size_t>(k - 1)] + (*out)[static_cast<size_t>(k + 1)]) * 0.5f;
            (*out)[static_cast<size_t>(k)] = lerp((*out)[static_cast<size_t>(k)], avg, gamma);
        }
    }
}

// Phase 2: the paper's on-the-fly geometry, over the whole frame.
void generate(const Scene& scene, const std::vector<BundleJob>& jobs,
              const SampleSet2D& quadSet, const SampleSet2D& triSet, std::vector<float>* arena,
              int threads) {
    parallelRanges(jobs.size(), threads, [&](size_t begin, size_t end, int) {
        std::vector<Vec3> raw;
        for (size_t ji = begin; ji < end; ++ji) {
            const BundleJob& job = jobs[ji];
            const HairInstance& hair = *scene.hair[job.instance];
            const int n = job.vertices;
            float* base = arena->data() + job.vertexOffset * 3u;
            for (uint32_t s = 0; s < job.strands; ++s) {
                generateStrand(hair, job.face, s, job.segments, job.gamma, quadSet, triSet, &raw);
                float* dst = base + static_cast<size_t>(s) * static_cast<size_t>(n) * 3u;
                for (int k = 0; k < n; ++k) {
                    dst[k * 3 + 0] = raw[static_cast<size_t>(k)].x;
                    dst[k * 3 + 1] = raw[static_cast<size_t>(k)].y;
                    dst[k * 3 + 2] = raw[static_cast<size_t>(k)].z;
                }
            }
        }
    });
}

}  // namespace

std::string CpuFastStats::toString() const {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "front end %.2f ms | generation %.2f ms | raster %.2f ms | resolve %.2f ms "
                  "| total %.2f ms\n  %llu bundles (%llu culled), %llu strands, %llu verts, "
                  "%llu segments, %llu shaded fragments, %d threads",
                  frontEndMs, generateMs, rasterMs, resolveMs, totalMs,
                  static_cast<unsigned long long>(bundles),
                  static_cast<unsigned long long>(bundlesCulled),
                  static_cast<unsigned long long>(strands),
                  static_cast<unsigned long long>(vertices),
                  static_cast<unsigned long long>(segments),
                  static_cast<unsigned long long>(pixelsShaded), threads);
    return buf;
}

void generateStrandFast(const HairInstance& hair, uint32_t face, uint32_t strandIndex,
                        int segments, float gamma, std::vector<Vec3>* out) {
    generateStrand(hair, face, strandIndex, segments, gamma, quadNoise(), triangleNoise(), out);
}

Vec2 fastRootCoordinate(const HairMesh& mesh, uint32_t face, uint32_t strandIndex) {
    const SampleSet2D& quad = quadNoise();
    const SampleSet2D& tri = triangleNoise();
    Vec2 xi = quad.at(static_cast<size_t>(strandIndex) + face * 131u);
    if (mesh.face(face).isTriangle())
        xi = rootPlacementTriangle(tri.at(static_cast<size_t>(strandIndex) + face * 97u));
    return Vec2(clampf(xi.x, 0.002f, 0.998f), clampf(xi.y, 0.002f, 0.998f));
}

int CpuFastRenderer::threads() const {
    if (threads_ > 0) return threads_;
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? static_cast<int>(hc) : 1;
}

void CpuFastRenderer::generateOnly(const Scene& scene, const Camera& camera,
                                   const RenderOptions& options, CpuFastStats* stats) {
    CpuFastStats st;
    st.threads = threads();
    const int width = std::max(1, options.width), height = std::max(1, options.height);

    Camera cam = camera;
    cam.aspect = static_cast<float>(width) / static_cast<float>(height);
    FrameShared fs;
    fs.width = width;
    fs.height = height;
    fs.camPos = cam.position;
    fs.viewProj = cam.proj() * cam.view();
    fs.frustum.fromViewProj(fs.viewProj);


    const auto t0 = clock::now();
    std::vector<BundleJob> jobs;
    frontEnd(scene, cam, options, fs, &jobs, &st);
    st.frontEndMs = msSince(t0);

    const auto t1 = clock::now();
    std::vector<float> arena(st.vertices * 3u, 0.0f);
    generate(scene, jobs, quadNoise(), triangleNoise(), &arena, st.threads);
    st.generateMs = msSince(t1);
    st.totalMs = st.frontEndMs + st.generateMs;
    if (stats) *stats = st;
}

void CpuFastRenderer::render(const Scene& scene, const Camera& camera,
                             const RenderOptions& options, Image* out, CpuFastStats* stats) {
    const auto tStart = clock::now();
    const int nThreads = threads();
    CpuFastStats st;
    st.threads = nThreads;

    const int width = std::max(1, options.width);
    const int height = std::max(1, options.height);

    Camera cam = camera;
    cam.aspect = static_cast<float>(width) / static_cast<float>(height);
    FrameShared fs;
    fs.width = width;
    fs.height = height;
    fs.camPos = cam.position;
    fs.viewProj = cam.proj() * cam.view();
    fs.frustum.fromViewProj(fs.viewProj);


    // ---------------- phase 1 ----------------
    const auto tFront0 = clock::now();
    std::vector<BundleJob> jobs;
    frontEnd(scene, cam, options, fs, &jobs, &st);
    st.frontEndMs = msSince(tFront0);

    // ---------------- phase 2 ----------------
    const auto tGen0 = clock::now();
    std::vector<float> arena(st.vertices * 3u, 0.0f);
    generate(scene, jobs, quadNoise(), triangleNoise(), &arena, nThreads);
    st.generateMs = msSince(tGen0);

    // ---------------- phase 3 ----------------
    const auto tRaster0 = clock::now();
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<std::vector<float>> buffers(static_cast<size_t>(nThreads));
    for (auto& b : buffers) b.assign(pixelCount * 4u, 0.0f);
    std::vector<uint64_t> fragmentsPerThread(static_cast<size_t>(nThreads), 0);
    std::vector<uint64_t> shadedPerThread(static_cast<size_t>(nThreads), 0);

    parallelRanges(jobs.size(), nThreads, [&](size_t begin, size_t end, int threadIndex) {
        float* buf = buffers[static_cast<size_t>(threadIndex)].data();
        std::vector<Vec3> pos(65), tan(65), color(65);
        std::vector<float> halfWidth(65);
        std::vector<float> sx(65), sy(65);
        std::vector<uint8_t> visible(65);
        uint64_t fragments = 0, shaded = 0;
        for (size_t ji = begin; ji < end; ++ji) {
            const BundleJob& job = jobs[ji];
            const HairInstance& hair = *scene.hair[job.instance];
            const int n = job.vertices;
            const float* base3 = arena.data() + job.vertexOffset * 3u;
            // Tile rejection: skip the whole bundle when its footprint misses this thread's
            // share of the screen. Each thread covers the full screen here, so only the
            // degeneracy test applies; the bbox is used by the per-segment clamps below.
            for (uint32_t s = 0; s < job.strands; ++s) {
                const float* src = base3 + static_cast<size_t>(s) * static_cast<size_t>(n) * 3u;
                bool anyVisible = false;
                for (int k = 0; k < n; ++k) {
                    const Vec3 local(src[k * 3], src[k * 3 + 1], src[k * 3 + 2]);
                    const Vec3 p = hair.transform.transformPoint(local);
                    pos[static_cast<size_t>(k)] = p;
                    float nx, ny;
                    if (!projectToScreen(fs.viewProj, p, &nx, &ny)) {
                        visible[static_cast<size_t>(k)] = 0;
                        continue;
                    }
                    sx[static_cast<size_t>(k)] = (nx * 0.5f + 0.5f) * static_cast<float>(width);
                    sy[static_cast<size_t>(k)] = (ny * 0.5f + 0.5f) * static_cast<float>(height);
                    visible[static_cast<size_t>(k)] = 1;
                    if (sx[static_cast<size_t>(k)] > -2.0f && sy[static_cast<size_t>(k)] > -2.0f &&
                        sx[static_cast<size_t>(k)] < width + 2.0f &&
                        sy[static_cast<size_t>(k)] < height + 2.0f)
                        anyVisible = true;
                }
                if (!anyVisible) continue;
                computeStrandTangents(pos.data(), n, tan.data());

                const float clumpJit = hash3f(pos[0] * 7.13f);
                const float jit = hash3f(Vec3(static_cast<float>(s) * 0.731f + job.face * 0.113f,
                                              static_cast<float>(job.face) * 0.517f, 3.7f));
                const float value = 1.0f + (0.30f * jit - 0.15f) + 0.40f * (clumpJit - 0.5f);
                const Vec3 albedo = hair.styling.color * value;
                const float ao = 0.72f + 0.28f * clumpJit;
                const float hwBase = clampf(0.55f * std::sqrt(std::max(1.0f, job.thickness)), 1.1f, 1.6f);
                for (int k = 0; k < n; ++k) {
                    const Vec3 p = pos[static_cast<size_t>(k)];
                    const Vec3 V = normalize(fs.camPos - p);
                    const float w = static_cast<float>(k) / static_cast<float>(n - 1);
                    color[static_cast<size_t>(k)] =
                        shadeFast(scene, tan[static_cast<size_t>(k)], V, albedo, ao, w);
                    halfWidth[static_cast<size_t>(k)] = hwBase;
                }

                for (int k = 1; k < n; ++k) {
                    if (!visible[static_cast<size_t>(k - 1)] || !visible[static_cast<size_t>(k)]) continue;
                    const float ax = sx[static_cast<size_t>(k - 1)], ay = sy[static_cast<size_t>(k - 1)];
                    const float bx = sx[static_cast<size_t>(k)], by = sy[static_cast<size_t>(k)];
                    const float hw = 0.5f * (halfWidth[static_cast<size_t>(k - 1)] +
                                             halfWidth[static_cast<size_t>(k)]);
                    const float reach = hw + 1.0f;
                    const int x0 = std::max(0, static_cast<int>(std::floor(std::min(ax, bx) - reach)));
                    const int x1 = std::min(width - 1, static_cast<int>(std::ceil(std::max(ax, bx) + reach)));
                    const int y0 = std::max(0, static_cast<int>(std::floor(std::min(ay, by) - reach)));
                    const int y1 = std::min(height - 1, static_cast<int>(std::ceil(std::max(ay, by) + reach)));
                    if (x1 < x0 || y1 < y0) continue;
                    const float dx = bx - ax, dy = by - ay;
                    const float len2 = dx * dx + dy * dy;
                    if (len2 < 1e-8f) continue;
                    const float invLen2 = 1.0f / len2;
                    // Coverage ramp, hoisted out of the pixel loop: one division per
                    // segment instead of one per fragment.
                    const float band = 1.0f / std::max(0.35f, hw * 0.55f);
                    const float covScale = band / std::max(hw, 1e-3f);
                    const Vec3 c0 = color[static_cast<size_t>(k - 1)];
                    const Vec3 c1 = color[static_cast<size_t>(k)];
                    fragments += static_cast<uint64_t>(x1 - x0 + 1) * static_cast<uint64_t>(y1 - y0 + 1);
                    for (int y = y0; y <= y1; ++y) {
                        const float py = y + 0.5f;
                        float* row = buf + static_cast<size_t>(y) * static_cast<size_t>(width) * 4u;
                        for (int x = x0; x <= x1; ++x) {
                            const float px = x + 0.5f;
                            float tt = ((px - ax) * dx + (py - ay) * dy) * invLen2;
                            tt = tt < 0.0f ? 0.0f : (tt > 1.0f ? 1.0f : tt);
                            const float qx = ax + dx * tt, qy = ay + dy * tt;
                            const float ddx = px - qx, ddy = py - qy;
                            // Coverage without a sqrt: compare d^2 against the squared
                            // radius and fall off linearly across the last pixel band.
                            const float d2 = ddx * ddx + ddy * ddy;
                            const float r2 = hw * hw;
                            float cov;
                            if (d2 <= r2 * 0.55f) {
                                cov = 1.0f;
                            } else {
                                if (d2 >= r2) continue;
                                cov = (r2 - d2) * covScale;
                                if (cov > 1.0f) cov = 1.0f;
                            }
                            const Vec3 c = lerp(c0, c1, tt);
                            const float a = options.hairAlpha * cov;
                            float* px4 = row + static_cast<size_t>(x) * 4u;
                            const float ia = 1.0f - a;
                            px4[0] = px4[0] * ia + c.x * a;
                            px4[1] = px4[1] * ia + c.y * a;
                            px4[2] = px4[2] * ia + c.z * a;
                            px4[3] = px4[3] * ia + a;
                            ++shaded;
                        }
                    }
                }
            }
        }
        fragmentsPerThread[static_cast<size_t>(threadIndex)] = fragments;
        shadedPerThread[static_cast<size_t>(threadIndex)] = shaded;
    });
    for (int i = 0; i < nThreads; ++i) {
        st.fragments += fragmentsPerThread[static_cast<size_t>(i)];
        st.pixelsShaded += shadedPerThread[static_cast<size_t>(i)];
    }
    st.rasterMs = msSince(tRaster0);

    // ---------------- phase 4 ----------------
    const auto tResolve0 = clock::now();
    out->resize(width, height);
    const Vec3 background = options.drawSky ? Vec3(0.30f, 0.38f, 0.50f)
                                            : Vec3(0.015f, 0.016f, 0.02f);
    parallelRanges(static_cast<size_t>(height), nThreads * 4, [&](size_t yBegin, size_t yEnd, int) {
        for (size_t y = yBegin; y < yEnd; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t idx = (y * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
                Vec3 c = background;
                for (const auto& b : buffers) {
                    const float pa = b[idx + 3];
                    if (pa <= 1e-5f) continue;
                    const Vec3 sc(b[idx], b[idx + 1], b[idx + 2]);
                    c = c * (1.0f - pa) + sc;  // sc is already premultiplied by its alpha
                }
                out->at(x, static_cast<int>(y)) = c;
            }
        }
    });
    st.resolveMs = msSince(tResolve0);
    st.totalMs = msSince(tStart);
    if (stats) *stats = st;
}

}  // namespace silk
