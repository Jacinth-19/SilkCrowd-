// Unit tests for the paper-critical parts of SilkCrowd.
// Build:  make test        (no external dependencies)
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "silkcrowd/assets.h"
#include "silkcrowd/catmull_rom.h"
#include "silkcrowd/hair_mesh_texture.h"
#include "silkcrowd/lod.h"
#include "silkcrowd/renderer.h"
#include "silkcrowd/root_placement.h"
#include "silkcrowd/strand_generator.h"
#include "silkcrowd/styling.h"

using namespace silk;

static int gFailures = 0;
static int gChecks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++gChecks;                                                               \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            ++gFailures;                                                         \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                            \
    do {                                                                                 \
        ++gChecks;                                                                       \
        const double va = static_cast<double>(a), vb = static_cast<double>(b);           \
        if (!(std::fabs(va - vb) <= (eps))) {                                            \
            std::printf("  FAIL %s:%d  |%s - %s| = %g > %g\n", __FILE__, __LINE__, #a,   \
                        #b, std::fabs(va - vb), static_cast<double>(eps));                \
            ++gFailures;                                                                 \
        }                                                                                \
    } while (0)

namespace {

HairMesh testMesh(int gu = 6, int gv = 5, int layers = 3) {
    HeadModel head;
    HairStyleSpec spec = HairStyleSpec::fromFigure10(0, 5u);
    spec.gridU = gu;
    spec.gridV = gv;
    spec.numLayers = layers;
    spec.totalStrands = gu * gv * 40;
    return buildHairMesh(head, spec);
}

void testCurves() {
    std::printf("curve conversion\n");
    std::vector<Vec3> pts = {{0, 0, 0}, {0.02f, 0.1f, 0.01f}, {0.01f, 0.2f, 0.03f},
                             {-0.02f, 0.28f, 0.02f}, {-0.01f, 0.35f, 0.0f}};
    const std::vector<float> knots = centripetalKnots(pts);
    CHECK(knots.size() == pts.size());
    for (size_t i = 1; i < knots.size(); ++i) CHECK(knots[i] > knots[i - 1]);

    for (int seg = 0; seg + 1 < static_cast<int>(pts.size()); ++seg) {
        const CubicBezier cb = catmullRomToCubicBezier(knots, pts, seg);
        for (int k = 0; k <= 32; ++k) {
            const float u = static_cast<float>(k) / 32.0f;
            CHECK_NEAR(length(cb.eval(u) - catmullRomEval(knots, pts, seg, u)), 0.0, 1e-6);
        }
        // Figure 3e control points.
        const QuadraticPair q = cubicToTwoQuadratics(cb);
        // Figure 3e / Truong et al. 2020:
        //   q0+ = (9 p0 + 21 b0+ + 3 b5- - p5)/32,  q5- = (9 p5 + 21 b5- + 3 b0+ - p0)/32
        const Vec3 q0p = (cb.p0 * 9.0f + cb.b0plus * 21.0f + cb.b5minus * 3.0f - cb.p5) * (1.0f / 32.0f);
        const Vec3 q5m = (cb.p5 * 9.0f + cb.b5minus * 21.0f + cb.b0plus * 3.0f - cb.p0) * (1.0f / 32.0f);
        CHECK_NEAR(length(q.q0plus - q0p), 0.0, 1e-7);
        CHECK_NEAR(length(q.q5minus - q5m), 0.0, 1e-7);
        // C1 continuity at the joint: both quadratics share the tangent vector.
        const Vec3 tJointA = (q.q05 - q.q0plus) * 2.0f;
        const Vec3 tJointB = (q.q5minus - q.q05) * 2.0f;
        CHECK_NEAR(length(tJointA - tJointB) / std::max(length(tJointA), 1e-9f), 0.0, 1e-4);
        CHECK_NEAR(length(q.q05 - cb.eval(0.5f)), 0.0, 1e-6);
        // Two quadratics reproduce the cubic closely.
        for (int k = 0; k <= 32; ++k) {
            const float u = static_cast<float>(k) / 32.0f;
            CHECK_NEAR(length(q.eval(u) - cb.eval(u)), 0.0, 5e-4);
        }
        // Continuity at the split point.
        CHECK_NEAR(length(q.eval(0.5f - 1e-4f) - q.eval(0.5f + 1e-4f)), 0.0, 1e-4);
    }
}

void testTextureLayout() {
    std::printf("hair mesh texture layout (Figure 3e)\n");
    const HairMesh mesh = testMesh(6, 5, 4);
    const StylingFrames frames = computeStylingFrames(mesh);
    HairMeshTextureSet tex;
    tex.build(mesh, frames, TexelFormat::Float32);

    // depth = 1 + (numLayers - 1) * 4
    CHECK(tex.depth() == 1u + (mesh.numLayers() - 1u) * 4u);
    CHECK(tex.depth() == 1u + 3u * 4u);
    CHECK(tex.wTexture().depth == mesh.numLayers());
    CHECK(tex.uTexture().depth == mesh.numLayers());
    CHECK(tex.vTexture().depth == mesh.numLayers());
    CHECK(tex.uvTexture().components == 2);

    // A 3-layer model: 1 + 2*4 = 9 slices; a 4-layer model: 13.
    CHECK(hairMeshTextureDepth(3) == 9);
    CHECK(hairMeshTextureDepth(4) == 13);
    CHECK(hairMeshTextureDepth(2) == 5);
    CHECK(hairMeshTextureDepth(1) == 1);

    const auto check = tex.validate(mesh, TexelFormat::Float32);
    CHECK(check.depthMatchesFigure3e);
    CHECK(check.attributeDepthsCorrect);
    CHECK(check.rootSliceMatchesMesh);
    CHECK(check.uvTextureMatchesMesh);
    CHECK(check.wTextureMonotonic);
    CHECK(check.intermediateSlicesConsistent);
    CHECK(check.maxCurveReprojectionError < 1e-4f);
    // ~2% of a segment length where the hair mesh bends sharply at the root (segment ~6 cm).
    CHECK(check.maxCatmullRomError < 2e-3f);

    // Bilinear within a bundle must reproduce the bilinear combination of the 4
    // root-layer vertices exactly (hardware filtering equivalence).
    const uint32_t face = 0;
    const HairMeshFace& f = mesh.face(face);
    const Vec2 ab(0.3f, 0.7f);
    const Vec2 st = tex.layout().blockToSlice(face, ab);
    const Vec3 sampled = tex.sampleSlice(st, 0.0f);
    const Vec3 p0 = mesh.rootVertex(f.v[0]);
    const Vec3 p1 = mesh.rootVertex(f.v[1]);
    const Vec3 p2 = mesh.rootVertex(f.v[2]);
    const Vec3 p3 = mesh.rootVertex(f.v[3]);
    const Vec3 bilinear = p0 * ((1 - ab.x) * (1 - ab.y)) + p1 * (ab.x * (1 - ab.y)) +
                          p2 * (ab.x * ab.y) + p3 * ((1 - ab.x) * ab.y);
    CHECK_NEAR(length(sampled - bilinear), 0.0, 1e-6);

    // The curve evaluation must hit the layer slices exactly at integer parameters.
    Vec3 curve[4];
    const std::vector<Vec3> vertCurve = mesh.vertexCurve(f.v[1]);
    for (uint32_t l = 0; l < 4; ++l) {
        curve[l] = tex.evalCurve(tex.layout().blockToSlice(face, Vec2(1.0f, 0.0f)), static_cast<float>(l));
        CHECK_NEAR(length(curve[l] - vertCurve[l]), 0.0, 1e-6);
    }

    // Only three textures are dynamic: rebuilding them must keep the static ones intact.
    HairMesh deformed = mesh;
    for (uint32_t l = 1; l < deformed.numLayers(); ++l)
        for (uint32_t v = 0; v < deformed.numVertices(); ++v)
            deformed.layerVertexMutable(l, v) += Vec3(0.003f, 0.001f, 0.0f);
    tex.updateDynamic(deformed, computeStylingFrames(deformed));
    CHECK(tex.validate(deformed, TexelFormat::Float32).rootSliceMatchesMesh);

    // Size accounting.
    const auto s16 = tex.sizeReport(TexelFormat::Float16);
    const auto s32 = tex.sizeReport(TexelFormat::Float32);
    CHECK_NEAR(static_cast<double>(s32.total()), 2.0 * static_cast<double>(s16.total()), 1.0);
    const uint64_t expected =
        static_cast<uint64_t>(tex.width()) * tex.height() * tex.depth() * 3 * 4ull +
        static_cast<uint64_t>(tex.width()) * tex.height() * 2 * 4ull +
        static_cast<uint64_t>(tex.width()) * tex.height() * mesh.numLayers() * 12ull * 2 +
        static_cast<uint64_t>(tex.width()) * tex.height() * mesh.numLayers() * 4ull;
    CHECK(s32.total() == expected);
}

void testFrames() {
    std::printf("styling frames (Eq. 1-4)\n");
    const HairMesh mesh = testMesh(7, 6, 3);
    const StylingFrames frames = computeStylingFrames(mesh);
    for (uint32_t l = 0; l < mesh.numLayers(); ++l)
        for (uint32_t v = 0; v < mesh.numVertices(); ++v) {
            const size_t k = static_cast<size_t>(l) * mesh.numVertices() + v;
            CHECK_NEAR(length(frames.u[k]), 1.0, 1e-4);
            CHECK_NEAR(length(frames.v[k]), 1.0, 1e-4);
            CHECK_NEAR(length(frames.w[k]), 1.0, 1e-4);
            // w = normalize(u x v)
            CHECK_NEAR(length(frames.w[k] - normalize(cross(frames.u[k], frames.v[k]))), 0.0, 1e-4);
        }

    HairMeshTextureSet tex;
    tex.build(mesh, frames, TexelFormat::Float32);
    const FrameContinuityReport eq = measureFrameContinuity(tex, FrameMode::PerVertexTextures);
    const FrameContinuityReport fd = measureFrameContinuity(tex, FrameMode::FiniteDifferences);
    CHECK(eq.sharedVerticesTested > 0);
    CHECK(eq.maxUVariation < 1e-3f);
    CHECK(fd.maxUVariation > eq.maxUVariation);
}

void testRootPlacement() {
    std::printf("root placement (Section 3.3)\n");
    // Barycentric exactness.
    for (int i = 0; i < 500; ++i) {
        const float a = static_cast<float>(i % 37) / 40.0f;
        const float b = static_cast<float>(i % 23) / 30.0f;
        if (a + b >= 0.99f) continue;
        const Vec2 ab = rootPlacementTriangle(Vec2(a, b));
        float w0, w1, w2;
        triangleBarycentricsFromBlock(ab, &w0, &w1, &w2);
        CHECK_NEAR(w1, a, 1e-5);
        CHECK_NEAR(w2, b, 1e-5);
        CHECK_NEAR(w0, 1.0f - a - b, 1e-5);
        CHECK(w0 >= -1e-6f && w1 >= -1e-6f && w2 >= -1e-6f);
    }
    // Blue noise beats white noise.
    SampleEliminationParams p;
    p.targetCount = 512;
    p.oversample = 2;
    p.seed = 31u;
    const SampleSet2D set = generateBlueNoiseSamples(p);
    CHECK(set.size() == 512);
    const BlueNoiseMetrics m = measureBlueNoise(set.samples);
    std::printf("  blue noise mean NN %.4f vs white noise %.4f (ratio %.2f)\n",
                m.meanNearestNeighbor, m.expectedWhiteNoise,
                m.meanNearestNeighbor / m.expectedWhiteNoise);
    CHECK(m.meanNearestNeighbor > m.expectedWhiteNoise * 1.15);
    // Progressive: every sample is inside the unit square.
    for (const Vec2& s : set.samples) CHECK(s.x >= 0 && s.x <= 1 && s.y >= 0 && s.y <= 1);
}

void testLod() {
    std::printf("level of detail (Eq. 7-12)\n");
    CHECK(detailLevel(0.02f, 1080.0f, 0.1f, 0.7f, 1.0f) <= 1.0f);
    CHECK(detailLevel(0.02f, 1080.0f, 30.0f, 0.7f, 0.5f) <
          detailLevel(0.02f, 1080.0f, 3.0f, 0.7f, 0.5f));
    CHECK_NEAR(detailLevel(1.0f, 1.0f, 1.0f, kPi * 0.5f, 1.0f), 0.5, 1e-5);

    // Eq. 8
    CHECK(strandCountLod(100, 0.0f, 1.0f) == 100);
    CHECK(strandCountLod(100, 0.5f, 0.5f) == 51);
    CHECK(strandCountLod(100, 0.0f, 0.001f) == 1);
    CHECK(strandCountLod(100, 0.0f, 0.0f) == 0);
    CHECK_NEAR(strandThicknessScale(0.25f, 100), 4.0, 1e-6);
    CHECK_NEAR(strandThicknessScale(0.001f, 10), 10.0, 1e-6);

    // Eq. 9-12
    for (int step = 1; step <= 100; ++step) {
        const float L = static_cast<float>(step) / 100.0f;
        const VertexLodResult v = computeVertexLod(L, 32, 0.35f);
        CHECK(v.exponentE >= 0 && v.exponentE <= 5);
        CHECK(v.numVertices() == (1 << v.exponentE) + 1);
        CHECK(v.gamma >= 0.0f && v.gamma <= 1.0f);
        if (v.exponentE == v.exponentPrev) CHECK_NEAR(v.gamma, 0.0f, 1e-6);
    }
    CHECK(computeVertexLod(1.0f, 32, 0.35f).exponentE == 5);
    CHECK(computeVertexLod(0.0001f, 32, 0.35f).exponentE == 0);
    // Just above a level boundary the strand is morphed almost completely.
    // The transition window is [k, k + lambda] in log2(L*ns) space.
    const float Lk = std::pow(2.0f, 4.0f) / 32.0f;              // x = 4
    CHECK(computeVertexLod(Lk * std::pow(2.0f, 1e-3f), 32, 0.35f).gamma > 0.9f);
    CHECK(computeVertexLod(Lk * std::pow(2.0f, 0.34f), 32, 0.35f).gamma < 0.1f);
    // x slightly above 4.35 has no transition in flight at all.
    CHECK_NEAR(computeVertexLod(Lk * std::pow(2.0f, 0.36f), 32, 0.35f).gamma, 0.0, 1e-6);
}

void testStrandGeneration() {
    std::printf("strand generation\n");
    const HairMesh mesh = testMesh(5, 4, 3);
    auto inst = makeHairInstance(mesh, StylingParams::fromPreset(2, 4u), 4u, TexelFormat::Float32);
    StrandGenerator gen(inst->textures, inst->styling, 0.0005f);
    VertexLodResult lod;
    lod.exponentE = 4;
    lod.exponentPrev = 4;
    lod.gamma = 0.0f;
    StrandRequest req;
    req.face = 0;
    req.blockCoord = Vec2(0.25f, 0.75f);
    StrandGeometry g;
    gen.generate(req, lod, &g);
    CHECK(g.vertices.size() == 17);
    // Root vertex equals the single root-slice lookup and is never styled.
    const Vec2 st = inst->textures.layout().blockToSlice(0, req.blockCoord);
    CHECK_NEAR(length(g.vertices[0].position - inst->textures.sampleSlice(st, 0.0f)), 0.0, 1e-6);
    // Section 3.5: interior tangents are p_(k+1) - p_(k-1) (not normalised); the
    // endpoint rules of Eq. 5/6 produce unit vectors. Direction must follow the strand.
    for (size_t k = 1; k + 1 < g.vertices.size(); ++k) {
        const Vec3 expected = g.vertices[k + 1].position - g.vertices[k - 1].position;
        CHECK_NEAR(length(normalize(g.vertices[k].tangent) - normalize(expected)), 0.0, 1e-4);
    }
    CHECK_NEAR(length(g.vertices[0].tangent), 1.0, 1e-4);
    CHECK_NEAR(length(g.vertices.back().tangent), 1.0, 1e-4);
    // Consecutive vertices are ordered along the strand (monotone w).
    for (size_t k = 1; k < g.vertices.size(); ++k) CHECK(g.vertices[k].w > g.vertices[k - 1].w);
    // The strand stays inside a plausible bounding box.
    Vec3 lo, hi;
    mesh.bounds(&lo, &hi);
    for (const StrandVertex& v : g.vertices) {
        CHECK(v.position.x > lo.x - 0.08f && v.position.x < hi.x + 0.08f);
        CHECK(v.position.y > lo.y - 0.08f && v.position.y < hi.y + 0.08f);
    }
    // Morphing pulls odd vertices towards their neighbours (Eq. 11).
    VertexLodResult morphed = lod;
    morphed.exponentPrev = 3;
    morphed.gamma = 1.0f;
    StrandGeometry g2;
    gen.generate(req, morphed, &g2);
    float oddShift = 0.0f;
    for (size_t k = 1; k + 1 < g.vertices.size(); k += 2)
        oddShift = std::max(oddShift, length(g2.vertices[k].position - g.vertices[k].position));
    CHECK(oddShift > 0.0f);
    for (size_t k = 0; k < g.vertices.size(); k += 2)
        CHECK_NEAR(length(g2.vertices[k].position - g.vertices[k].position), 0.0, 1e-6);
}

void testTriangleBundle() {
    std::printf("triangle bundles\n");
    // A single triangle scalp face: the texture block duplicates one vertex.
    HeadModel head;
    HairMesh mesh;
    std::vector<HairMeshFace> faces(1);
    faces[0].count = 3;
    faces[0].v[0] = 0;
    faces[0].v[1] = 1;
    faces[0].v[2] = 2;
    std::vector<Vec2> uv = {{0.1f, 0.1f}, {0.5f, 0.1f}, {0.2f, 0.5f}};
    mesh.setTopology(3, 3, faces, uv);
    for (uint32_t v = 0; v < 3; ++v) {
        mesh.layerVertexMutable(0, v) = Vec3(0.01f * v, 0.0f, 0.0f);
        mesh.layerVertexMutable(1, v) = Vec3(0.01f * v, 0.05f, 0.0f);
        mesh.layerVertexMutable(2, v) = Vec3(0.01f * v, 0.09f, 0.0f);
    }
    mesh.computeScalpNormals();
    mesh.setUniformStrandCount(10);
    const StylingFrames frames = computeStylingFrames(mesh);
    HairMeshTextureSet tex;
    tex.build(mesh, frames, TexelFormat::Float32);
    CHECK(tex.depth() == 1u + 2u * 4u);
    // Every texel of the block holds a valid root vertex; the duplicate is v2.
    for (int corner = 0; corner < 4; ++corner) {
        const Vec3 p = tex.texelPosition(0, corner, 0);
        CHECK_NEAR(p.y, 0.0, 1e-6);
    }
    CHECK(tex.validate(mesh, TexelFormat::Float32).rootSliceMatchesMesh);
}

}  // namespace

int main() {
    std::printf("SilkCrowd unit tests\n====================\n");
    testCurves();
    testTextureLayout();
    testFrames();
    testRootPlacement();
    testLod();
    testStrandGeneration();
    testTriangleBundle();
    std::printf("====================\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
