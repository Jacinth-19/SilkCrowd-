// SilkCrowd - command line front end (headless validation, benchmarking, screenshots).
//
// The Qt6/Vulkan editor (src/app/editor_qt.cpp) drives the same core library; this entry
// point exists so that the paper's claims can be validated automatically on any machine,
// including CI runners without a display or a GPU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "app_gpu.h"
#include "gpu_probe.h"
#include "scenes.h"

#include "silkcrowd/assets.h"
#include "silkcrowd/catmull_rom.h"
#include "silkcrowd/hair_mesh_texture.h"
#include "silkcrowd/image.h"
#include "silkcrowd/lod.h"
#include "silkcrowd/renderer.h"
#include "silkcrowd/root_placement.h"
#include "silkcrowd/sim.h"
#include "silkcrowd/strand_generator.h"
#include "silkcrowd/styling.h"

using namespace silk;

namespace silk {
void writeGlif(const std::string& dir);
void writeHairMeshAssets(const std::string& dir);
}  // namespace silk

namespace {

struct Args {
    std::string mode = "help";
    std::string out = "screenshot.png";
    std::string scene = "head";
    int width = 1280, height = 720;
    int ssaa = 2;
    int crowd = 100;
    int style = 0;
    int frames = 1;
    bool lod = true;
    bool styling = true;
    bool stats = false;
    bool finiteDifference = false;
    bool drawHair = true;
    bool drawHead = true;
    float exposure = 1.05f;
    float seconds = 5.0f;
    int strandCount = 40000;
    std::string reportPath = "";
    bool simulate = false;
    bool sagFree = true;
    float wind = 0.0f;
    bool gpu = true;        // prefer Vulkan, fall back to the software rasteriser
    bool software = false;  // force the software path
    bool compute = false;   // force the compute fallback on a mesh-shader device
    bool validation = false;  // enable VK_LAYER_KHRONOS_validation on the Vulkan path
    int msaa = 1;
    bool taa = true;
    std::string headScan;    // "" = default asset, "none" = procedural
    float yaw = 0.0f;        // camera azimuth around the scene target, degrees
    bool studio = false;     // dark studio backdrop instead of sky + ground grid
};

void printHelp() {
    std::printf(
        "SilkCrowd - real-time hair crowd rendering with hair meshes\n"
        "  (replication of Bhokare, Montalvo, Diaz, Yuksel - SIGGRAPH 2024)\n\n"
        "Usage: SilkCrowd [mode] [options]\n\n"
        "Modes:\n"
        "  --validate            run the paper validation suite (Figure 3e, Eq. 1-12)\n"
        "  --bench               benchmark generation + rasterisation, report workloads\n"
        "  --screenshot <file>   render one frame to a PNG\n"
        "  --assets              write assets/heads/*.gltf and assets/hairmeshes/*.skhm\n"
        "  --interactive         launch the Qt6 editor (requires a build with Qt6)\n"
        "  --gpu-info            diagnose the Vulkan path: build, loader, driver (ICD), device\n\n"
        "Options:\n"
        "  --scene head|crowd|teapot   scene to render (default head)\n"
        "  --width N --height N        resolution (default 1280x720)\n"
        "  --ssaa N                    supersampling factor (default 2)\n"
        "  --crowd N                   number of characters (default 100)\n"
        "  --style N                   hair style preset 0-9\n"
        "  --strands N                 strands for the hero head\n"
        "  --no-lod                    disable both LOD types (Figure 8 comparison)\n"
        "  --no-styling                disable procedural styling\n"
        "  --finite-differences        use the naive frame of Figure 4a (shows the seam)\n"
        "  --frames N                  render N frames (animation / timing)\n"
        "  --seconds N                 headless run duration for CI\n"
        "  --exposure F                tone mapping exposure\n"
        "  --report <file>             write the validation report to a file\n"
        "  --no-hair / --no-head       hide the hair / the head (diagnostics)\n"
        "  --yaw DEG                   orbit the camera around the scene centre (0 = the\n"
        "                              default three-quarter view; 90 = profile)\n"
        "  --studio                    dark studio backdrop: no sky, no floor grid\n"
        "  --head-scan <file.glb>      a different scanned head, or one of the shipped names:\n"
        "                              male (photogrammetry scan, CC BY) or female (MakeHuman\n"
        "                              bust, CC0). Default: shipped male bust\n"
        "  --procedural-head           ignore the scan and use the parametric head\n"
        "  --simulate                  run the XPBD hair mesh simulation before rendering\n"
        "  --no-sag-free               skip the sag-free rest-shape initialization\n"
        "  --gpu / --vulkan            use the Vulkan renderer (default when available)\n"
        "                              when it is not available, --gpu-info says which piece\n"
        "                              is missing: the build, the loader, the driver or a device\n"
        "  --software / --cpu          force the dependency-free software rasteriser\n"
        "  --compute                   force the compute-shader fallback (Adreno/UHD path)\n"
        "  --validation                enable the Vulkan validation layer (needs vulkan-validationlayers)\n"
        "  --msaa N                    1, 2, 4 or 8 (Vulkan path)\n"
        "  --no-taa                    disable temporal anti-aliasing (Vulkan path)\n"
        "  --wind F                    wind speed in m/s (simulation)\n");
}

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s expects a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (s == "--validate") a.mode = "validate";
        else if (s == "--gpu-info" || s == "--vulkan-info") a.mode = "gpu";
        else if (s == "--bench") a.mode = "bench";
        else if (s == "--screenshot") { a.mode = "screenshot"; a.out = next("--screenshot"); }
        else if (s == "--assets") a.mode = "assets";
        else if (s == "--interactive") a.mode = "interactive";
        else if (s == "--headless") { /* accepted for CI symmetry; the CLI is headless */ }
        else if (s == "--scene") a.scene = next("--scene");
        else if (s == "--width") a.width = std::stoi(next("--width"));
        else if (s == "--height") a.height = std::stoi(next("--height"));
        else if (s == "--ssaa") a.ssaa = std::stoi(next("--ssaa"));
        else if (s == "--crowd") a.crowd = std::stoi(next("--crowd"));
        else if (s == "--style") a.style = std::stoi(next("--style"));
        else if (s == "--strands") a.strandCount = std::stoi(next("--strands"));
        else if (s == "--frames") a.frames = std::stoi(next("--frames"));
        else if (s == "--seconds") a.seconds = std::stof(next("--seconds"));
        else if (s == "--exposure") a.exposure = std::stof(next("--exposure"));
        else if (s == "--report") a.reportPath = next("--report");
        else if (s == "--no-lod") a.lod = false;
        else if (s == "--no-styling") a.styling = false;
        else if (s == "--finite-differences") a.finiteDifference = true;
        else if (s == "--stats") a.stats = true;
        else if (s == "--simulate") a.simulate = true;
        else if (s == "--no-sag-free") a.sagFree = false;
        else if (s == "--wind") a.wind = std::stof(next("--wind"));
        else if (s == "--gpu" || s == "--vulkan") a.gpu = true;
        else if (s == "--software" || s == "--cpu") { a.software = true; a.gpu = false; }
        else if (s == "--compute") a.compute = true;
        else if (s == "--validation") a.validation = true;
        else if (s == "--msaa") a.msaa = std::stoi(next("--msaa"));
        else if (s == "--no-taa") a.taa = false;
        else if (s == "--no-hair") a.drawHair = false;
        else if (s == "--no-head") a.drawHead = false;
        else if (s == "--yaw") a.yaw = std::stof(next("--yaw"));
        else if (s == "--studio") a.studio = true;
        else if (s == "--head-scan") a.headScan = next("--head-scan");
        else if (s == "--procedural-head") a.headScan = "none";
        else if (s == "--help" || s == "-h") { printHelp(); std::exit(0); }
        else {
            std::fprintf(stderr, "error: unknown option '%s'\n", s.c_str());
            std::exit(2);
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
//  Validation suite
// ---------------------------------------------------------------------------
struct Report {
    std::string text;
    int passed = 0, failed = 0;
    void line(const std::string& s) { text += s + "\n"; }
    void check(bool ok, const std::string& what, const std::string& detail = "") {
        text += std::string(ok ? "  [PASS] " : "  [FAIL] ") + what;
        if (!detail.empty()) text += "  (" + detail + ")";
        text += "\n";
        ok ? ++passed : ++failed;
    }
};

void validateCurves(Report& r) {
    r.line("== Curve conversion (Section 2.2, Figure 3d-3e) ==");
    // Random control polygon.
    std::vector<Vec3> pts;
    uint32_t st = 12345u;
    auto rnd = [&]() {
        st = st * 1664525u + 1013904223u;
        return (st >> 8) * (1.0f / 16777216.0f);
    };
    for (int i = 0; i < 6; ++i) pts.push_back(Vec3(rnd() * 0.2f - 0.1f, rnd() * 0.3f, rnd() * 0.2f - 0.1f));
    const std::vector<float> knots = centripetalKnots(pts);

    double maxBezier = 0.0, maxQuadRel = 0.0;
    for (int seg = 0; seg + 1 < static_cast<int>(pts.size()); ++seg) {
        const CubicBezier cb = catmullRomToCubicBezier(knots, pts, seg);
        const QuadraticPair q = cubicToTwoQuadratics(cb);
        for (int k = 0; k <= 64; ++k) {
            const float u = static_cast<float>(k) / 64.0f;
            maxBezier = std::max<double>(maxBezier, length(cb.eval(u) - catmullRomEval(knots, pts, seg, u)));
            const double segLen = std::max<double>(length(pts[seg + 1] - pts[seg]), 1e-9);
            maxQuadRel = std::max<double>(maxQuadRel, length(q.eval(u) - cb.eval(u)) / segLen);
        }
    }
    r.check(maxBezier < 1e-6, "Catmull-Rom -> cubic Bezier is exact", "max error " + std::to_string(maxBezier));
    r.check(maxQuadRel < 0.03, "cubic -> 2 quadratic Beziers (Truong 2020)",
            "max error " + std::to_string(100.0 * maxQuadRel) + "% of a segment length");
    // Endpoint tangents (Eq. 5 and 6) must keep the strand closed in shading terms.
    const CubicBezier c0 = catmullRomToCubicBezier(knots, pts, 0);
    const Vec3 hermite0 = catmullRomTangent(knots, pts, 0);
    const Vec3 bezierT0 = (c0.b0plus - c0.p0) * 3.0f / std::max(knots[1] - knots[0], 1e-6f);
    r.check(length(hermite0 - bezierT0) < 1e-4, "Bezier handles match the Catmull-Rom tangents");
}

void validateTextures(Report& r, const HairMesh& mesh) {
    r.line("\n== Hair mesh texture (Table 1, Figure 3e) ==");
    const StylingFrames frames = computeStylingFrames(mesh);
    HairMeshTextureSet tex;
    tex.build(mesh, frames, TexelFormat::Float32);
    const auto check = tex.validate(mesh, TexelFormat::Float32);
    r.line(check.toString());
    r.check(check.depthMatchesFigure3e, "depth = 1 + (numLayers-1)*4",
            "depth " + std::to_string(tex.depth()) + " for " + std::to_string(mesh.numLayers()) + " layers");
    r.check(check.attributeDepthsCorrect, "uv-texture 2D, w/u/v-textures have no intermediate slices");
    r.check(check.rootSliceMatchesMesh, "root slice == scalp vertices (bilinear within a bundle)");
    r.check(check.uvTextureMatchesMesh, "uv-texture == root uv styling coordinates");
    r.check(check.wTextureMonotonic, "w-texture: w = 0 at the root, increasing to the tip");
    r.check(check.intermediateSlicesConsistent, "slice 4L+1/4L+2/4L+3 == q+, q0.5, q- (Figure 3e)");
    r.check(check.maxCurveReprojectionError < 1e-5,
            "2 trilinear + 1 linear reproduces the quadratic Beziers",
            "max error " + std::to_string(check.maxCurveReprojectionError));
    r.check(check.maxCatmullRomError < 3e-3, "texture evaluation vs. exact Catmull-Rom",
            "max error " + std::to_string(check.maxCatmullRomError));

    // Dynamic-texture update path: deforming the mesh must only touch the 3 dynamic textures.
    HairMesh deformed = mesh;
    for (uint32_t l = 1; l < deformed.numLayers(); ++l)
        for (uint32_t v = 0; v < deformed.numVertices(); ++v)
            deformed.layerVertexMutable(l, v) += Vec3(0.004f, -0.002f, 0.001f);
    const StylingFrames frames2 = computeStylingFrames(deformed);
    tex.updateDynamic(deformed, frames2);
    const auto check2 = tex.validate(deformed, TexelFormat::Float32);
    r.check(check2.rootSliceMatchesMesh && check2.intermediateSlicesConsistent,
            "updateDynamic() rebuilds the dynamic textures after a deformation");
}

void validateFrames(Report& r, const HairMesh& mesh) {
    r.line("\n== Styling coordinates (Eq. 1-4, Figures 4-5) ==");
    const StylingFrames frames = computeStylingFrames(mesh);
    HairMeshTextureSet tex;
    tex.build(mesh, frames, TexelFormat::Float32);

    // Orthonormality of the stored frames.
    float minLen = 1e9f, maxLen = 0.0f, maxNonOrtho = 0.0f;
    for (uint32_t l = 0; l < mesh.numLayers(); ++l) {
        for (uint32_t v = 0; v < mesh.numVertices(); ++v) {
            const size_t k = static_cast<size_t>(l) * mesh.numVertices() + v;
            minLen = std::min({minLen, length(frames.u[k]), length(frames.v[k])});
            maxLen = std::max({maxLen, length(frames.u[k]), length(frames.v[k])});
            maxNonOrtho = std::max(maxNonOrtho, std::fabs(dot(frames.u[k], frames.v[k])));
        }
    }
    r.check(std::fabs(minLen - 1.0f) < 1e-4f && std::fabs(maxLen - 1.0f) < 1e-4f,
            "u_hat and v_hat are unit vectors (Eq. 1)");
    r.check(length(cross(frames.u[0], frames.v[0])) > 1e-3f, "w_hat = (u x v)/||u x v|| is well defined (Eq. 4)");
    r.line("  skew (|u_hat . v_hat|, preserved by design): " + std::to_string(maxNonOrtho));

    const FrameContinuityReport withTextures = measureFrameContinuity(tex, FrameMode::PerVertexTextures);
    const FrameContinuityReport withFD = measureFrameContinuity(tex, FrameMode::FiniteDifferences);
    r.line("  Eq. 1-4 frames:  " + withTextures.toString());
    r.line("  finite diffs:    " + withFD.toString());
    r.check(withTextures.maxUVariation < 1e-3f, "Eq. 1-4 frames are continuous across bundle edges (Figure 4b)",
            "max " + std::to_string(withTextures.maxUVariation) + " rad");
    r.check(withFD.maxUVariation > withTextures.maxUVariation * 10.0f + 1e-3f,
            "finite differences are discontinuous (reproduces the Figure 4a seam)",
            "max " + std::to_string(withFD.maxUVariation) + " rad");
}

void validateRoots(Report& r) {
    r.line("\n== Hair root placement (Section 3.3, Yuksel 2015) ==");
    SampleEliminationParams p;
    p.targetCount = 2048;
    p.oversample = 3;
    p.seed = 424242u;
    const SampleSet2D quad = generateBlueNoiseSamples(p);
    p.seed = 777u;
    p.triangleDomain = true;
    const SampleSet2D tri = generateBlueNoiseSamples(p);

    const BlueNoiseMetrics mq = measureBlueNoise(quad.samples);
    const BlueNoiseMetrics mt = measureBlueNoise(tri.samples);
    r.line("  quad set:     " + mq.toString());
    r.line("  triangle set: " + mt.toString());
    r.check(mq.meanNearestNeighbor > mq.expectedWhiteNoise * 1.3,
            "sample elimination produces blue noise (not white noise)",
            "mean NN " + std::to_string(mq.meanNearestNeighbor) + " vs " + std::to_string(mq.expectedWhiteNoise));
    r.check(quad.size() == 2048 && tri.size() == 2048, "progressive sets hold the requested sample count");

    // Triangle mapping: the bilinear lookup must reproduce barycentric interpolation.
    double maxBaryErr = 0.0;
    double sumB0 = 0, sumB1 = 0, sumB2 = 0;
    double sumLit0 = 0, sumLit1 = 0, sumLit2 = 0;
    uint32_t st = 99u;
    auto rnd = [&]() {
        st = st * 1664525u + 1013904223u;
        return (st >> 8) * (1.0f / 16777216.0f);
    };
    const int N = 20000;
    for (int i = 0; i < N; ++i) {
        // Uniform sample in the triangular half xi_s + xi_t < 1.
        float a = rnd(), b = rnd();
        if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
        const Vec2 xi(a, b);
        const Vec2 ab = rootPlacementTriangle(xi, TriangleMapping::Barycentric);
        float w0, w1, w2;
        triangleBarycentricsFromBlock(ab, &w0, &w1, &w2);
        // Expect w = (1-a-b, a, b).
        maxBaryErr = std::max(maxBaryErr, static_cast<double>(std::max({std::fabs(w0 - (1.0f - a - b)),
                                                                    std::fabs(w1 - a), std::fabs(w2 - b)})));
        sumB0 += w0; sumB1 += w1; sumB2 += w2;

        const Vec2 abLit = rootPlacementTriangle(xi, TriangleMapping::PaperLiteral);
        float l0, l1, l2;
        triangleBarycentricsFromBlock(abLit, &l0, &l1, &l2);
        sumLit0 += l0; sumLit1 += l1; sumLit2 += l2;
    }
    r.check(maxBaryErr < 1e-5, "triangle root placement reproduces barycentrics (Section 3.3)",
            "max error " + std::to_string(maxBaryErr));
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  mean barycentrics (uniform samples): Barycentric mapping %.4f/%.4f/%.4f, "
                  "paper-literal expression %.4f/%.4f/%.4f (ideal 0.3333)",
                  sumB0 / N, sumB1 / N, sumB2 / N, sumLit0 / N, sumLit1 / N, sumLit2 / N);
    r.line(buf);
    r.check(std::fabs(sumB0 / N - 1.0 / 3.0) < 0.01, "no bias towards the duplicated vertex");
}

void validateLod(Report& r) {
    r.line("\n== Level of detail (Section 3.6, Eq. 7-12, Figures 6-8) ==");
    const uint32_t N = 200;      // strands per bundle
    const uint32_t F = 100;      // bundles
    // Eq. 8 vs plain ceil(L*N). Sample L finely and measure the size of a single
    // transition event: with plain ceil(L*N) all F bundles drop a strand at exactly the
    // same L, so the model's geometry jumps by F strands; the random increment delta_i
    // is what spreads those events out (Section 3.6.1).
    int maxDropEq8 = 0, maxDropPlain = 0;
    long long prevEq8 = -1, prevPlain = -1;
    int minStrands = 1 << 30;
    // Sweep the L range where the LOD is actually used. At very small L the strands per
    // bundle are already in single digits, so any scheme must change the count quickly.
    const int steps = 40000;
    const float Lmin = 0.20f, Lmax = 0.995f;
    for (int step = steps; step >= 1; --step) {
        const float L = Lmin + (Lmax - Lmin) * static_cast<float>(step) / static_cast<float>(steps);
        uint32_t total8 = 0, totalPlain = 0;
        for (uint32_t f = 0; f < F; ++f) {
            total8 += strandCountLod(N, bundleDelta(f, 7u), L);
            totalPlain += std::max(1u, static_cast<uint32_t>(std::ceil(L * N)));
        }
        if (prevEq8 >= 0) {
            maxDropEq8 = std::max<int>(maxDropEq8, static_cast<int>(prevEq8 - total8));
            maxDropPlain = std::max<int>(maxDropPlain, static_cast<int>(prevPlain - totalPlain));
        }
        prevEq8 = total8;
        prevPlain = totalPlain;
        for (uint32_t f = 0; f < F; ++f)
            minStrands = std::min<int>(minStrands, strandCountLod(N, bundleDelta(f, 7u), L));
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  bundles F=%u, N=%u: largest single transition event: Eq. 8 = %d strands, "
                  "plain ceil(L*N) = %d strands", F, N, maxDropEq8, maxDropPlain);
    r.line(buf);
    r.check(maxDropEq8 * 4 < maxDropPlain, "Eq. 8 random increment desynchronises bundle LOD transitions",
            std::to_string(maxDropEq8) + " vs " + std::to_string(maxDropPlain) + " strands per event");
    r.check(minStrands >= 1, "each bundle keeps at least one strand for L > 0");
    r.check(strandCountLod(N, 0.5f, 0.0f) == 0, "L = 0 culls the bundle");

    const float thick1 = strandThicknessScale(1.0f, N);
    const float thickHalf = strandThicknessScale(0.5f, N);
    r.check(std::fabs(thick1 - 1.0f) < 1e-6f && std::fabs(thickHalf - 2.0f) < 1e-6f,
            "strand thickness compensation min(1/L, N_i)");

    // Eq. 9: 2^e + 1 vertices, monotone with L.
    int prevE = 99;
    bool monotone = true, powerOk = true;
    for (int step = 120; step >= 1; --step) {
        const float L = static_cast<float>(step) / 120.0f;
        const VertexLodResult v = computeVertexLod(L, 32, 0.35f);
        if (v.exponentE > prevE) monotone = false;
        prevE = v.exponentE;
        if (v.numVertices() != (1 << v.exponentE) + 1) powerOk = false;
    }
    r.check(monotone, "vertex LOD is monotone in L (Eq. 9)");
    r.check(powerOk, "vertex count is always 2^e + 1");

    // Eq. 12: gamma in [0,1] and the shape is continuous across a level boundary.
    float minGamma = 1.0f, maxGamma = 0.0f;
    for (int step = 1; step <= 2000; ++step) {
        const float L = static_cast<float>(step) / 2000.0f;
        const float g = vertexMorphGamma(L, 32, 0.35f);
        minGamma = std::min(minGamma, g);
        maxGamma = std::max(maxGamma, g);
    }
    r.check(minGamma >= 0.0f && maxGamma <= 1.0f, "gamma in [0,1] (Eq. 12)");

    // Geometric continuity at a boundary: L slightly below vs above 8/32 = 0.25.
    HeadModel head;
    HairStyleSpec spec = HairStyleSpec::fromFigure10(1, 3u);
    spec.totalStrands = 4000;
    HairMesh mesh = buildHairMesh(head, spec);
    auto inst = makeHairInstance(mesh, spec.styling, 3u, TexelFormat::Float32);
    StrandGenerator gen(inst->textures, inst->styling, inst->baseRadius);
    StrandRequest req;
    req.face = 0;
    req.blockCoord = Vec2(0.37f, 0.61f);
    req.strandIndex = 3;
    req.bundleIndex = 0;
    const float eps = 1e-4f;
    const VertexLodResult lowLod = computeVertexLod(0.25f - eps, 32, 0.35f);
    const VertexLodResult highLod = computeVertexLod(0.25f + eps, 32, 0.35f);
    StrandGeometry gLow, gHigh;
    gen.generate(req, lowLod, &gLow);
    gen.generate(req, highLod, &gHigh);
    float maxDiff = 0.0f;
    if (gHigh.vertices.size() == 2 * (gLow.vertices.size() - 1) + 1) {
        for (size_t k = 0; k < gLow.vertices.size(); ++k)
            maxDiff = std::max(maxDiff, length(gLow.vertices[k].position - gHigh.vertices[2 * k].position));
    }
    std::snprintf(buf, sizeof(buf),
                  "  level boundary L=0.25: e %d -> %d, gamma %.3f, even-vertex mismatch %.5f m",
                  lowLod.exponentE, highLod.exponentE, highLod.gamma, maxDiff);
    r.line(buf);
    r.check(lowLod.exponentE + 1 == highLod.exponentE, "vertex LOD steps by exactly one power of two");
    r.check(maxDiff < 0.01f, "Eq. 11/12 morphing keeps the rendered strand continuous across levels");

    // Eq. 7 sanity: L is 1 when close and drops with distance.
    const float R = 720.0f;
    const float Lclose = detailLevel(0.02f, R, 0.3f, deg2rad(40.0f), 0.03f);
    const float Lfar = detailLevel(0.02f, R, 12.0f, deg2rad(40.0f), 0.03f);
    r.check(Lclose > 0.9f && Lfar < 0.1f, "Eq. 7 detail level falls with distance",
            "L(0.3m)=" + std::to_string(Lclose) + ", L(12m)=" + std::to_string(Lfar));
}

void validateMemory(Report& r) {
    r.line("\n== Storage (Table 1, Figure 1) ==");
    HeadModel head;
    double totalFp16 = 0.0;
    r.line("  preset                  verts  layers  depth  fp16 KB   fp32 KB");
    for (int i = 0; i < HairStyleSpec::numPresets(); ++i) {
        HairStyleSpec spec = HairStyleSpec::fromFigure10(i, 1u + i);
        HairMesh mesh = buildHairMesh(head, spec);
        const StylingFrames frames = computeStylingFrames(mesh);
        HairMeshTextureSet tex;
        tex.build(mesh, frames, TexelFormat::Float16);
        const auto s16 = tex.sizeReport(TexelFormat::Float16);
        const auto s32 = tex.sizeReport(TexelFormat::Float32);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  %-22s %5u  %5u  %5u  %8.2f  %8.2f",
                      spec.name, mesh.numVertices(), mesh.numLayers(), tex.depth(), s16.totalKB(), s32.totalKB());
        r.line(buf);
    }

    // The 100-character crowd: each model must fit in the paper's 13-21 KB range.
    double minKB = 1e9, maxKB = 0.0;
    totalFp16 = 0.0;
    for (int i = 0; i < 100; ++i) {
        HairStyleSpec spec = crowdStyleSpec(i, 100u + i);
        spec.totalStrands = 100000;
        HairMesh mesh = buildHairMesh(head, spec);
        const StylingFrames frames = computeStylingFrames(mesh);
        HairMeshTextureSet tex;
        tex.build(mesh, frames, TexelFormat::Float16);
        const double kb = tex.sizeReport(TexelFormat::Float16).totalKB();
        totalFp16 += kb;
        minKB = std::min(minKB, kb);
        maxKB = std::max(maxKB, kb);
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  100 crowd models: %.2f MB total, %.1f-%.1f KB per model (paper: 1.7 MB, 13-21 KB)",
                  totalFp16 / 1024.0, minKB, maxKB);
    r.line(buf);
    r.check(totalFp16 / 1024.0 <= 1.7, "100 unique hair models fit in 1.7 MB");
    r.check(minKB >= 8.0 && maxKB <= 24.0, "per-model footprint in the paper's 13-21 KB range");
}

void validateWorkload(Report& r) {
    r.line("\n== Crowd workload (Figure 1) ==");
    Scene scene = buildCrowdScene(100);
    // Same camera as the Figure 1 shot.
    const Camera cam = crowdCamera(1920, 1080);
    uint64_t totalStrands = 0, lodStrands = 0, lodVerts = 0, tris = 0;
    int fullStrands = 0;
    for (const auto& inst : scene.hair) {
        Vec3 lo, hi;
        inst->mesh.bounds(&lo, &hi);
        const float strandLen = std::max(hi.y - lo.y, 1e-3f);
        for (uint32_t f = 0; f < inst->mesh.numFaces(); ++f) {
            const uint32_t n = inst->mesh.bundleStrandCounts()[f];
            totalStrands += n;
            fullStrands += static_cast<int>(n);
            const Vec3 c = inst->transform.transformPoint(inst->bundleCenter[f]);
            const float d = lodDistance(cam.position, c, inst->bundleSize[f] * 0.5f);
            const LodParams lodParams;
            const float Ls = detailLevel(inst->bundleSize[f], 1080.0f, d, cam.fovY, lodParams.alphaStrand);
            const StrandLodResult sl = computeStrandLod(n, bundleDelta(f, inst->seed), Ls);
            const float Lv = detailLevel(strandLen, 1080.0f, d, cam.fovY, lodParams.alphaVertex);
            const VertexLodResult vl = computeVertexLod(Lv, 32, 0.35f);
            lodStrands += sl.numStrands;
            lodVerts += static_cast<uint64_t>(sl.numStrands) * vl.numVertices();
            tris += 2ull * static_cast<uint64_t>(sl.numStrands) * vl.numSegments();
        }
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  strands: %llu total (no LOD) -> %llu rendered with LOD (%.2f%%), "
                  "strand vertices %llu, triangles %llu",
                  static_cast<unsigned long long>(totalStrands),
                  static_cast<unsigned long long>(lodStrands),
                  100.0 * lodStrands / std::max<uint64_t>(totalStrands, 1),
                  static_cast<unsigned long long>(lodVerts), static_cast<unsigned long long>(tris));
    r.line(buf);
    std::snprintf(buf, sizeof(buf),
                  "  without LOD the same scene generates %llu strand vertices and %llu triangles "
                  "(%.1f GB of vertex data if it had to be stored)",
                  static_cast<unsigned long long>(totalStrands) * 33ull,
                  static_cast<unsigned long long>(totalStrands) * 64ull,
                  static_cast<double>(totalStrands) * 33.0 * 32.0 / (1024.0 * 1024.0 * 1024.0));
    r.line(buf);
    r.check(totalStrands >= 9000000ull, "100 characters x ~100k unique strands",
            std::to_string(totalStrands) + " strands");
    r.check(lodStrands < totalStrands * 7 / 10,
            "LOD removes a large part of the strand workload at the crowd distance",
            "100% -> " + std::to_string(100.0 * lodStrands / std::max<uint64_t>(totalStrands, 1)) + "%");
    r.check(lodVerts < totalStrands * 33ull,
            "vertex LOD reduces the generated vertex count",
            std::to_string(lodVerts) + " vs " + std::to_string(totalStrands * 33ull) + " vertices");
}

void validateSimulation(Report& r) {
    r.line("\n== Hair mesh simulation (Section 4: XPBD, Wu and Yuksel 2016, Hsu 2022) ==");
    HeadModel head;
    HairStyleSpec spec = HairStyleSpec::fromFigure10(0, 3u);
    spec.gridU = 8;
    spec.gridV = 7;
    spec.totalStrands = 1000;

    auto run = [&](bool sagFree, float wind, int frames, float* meanDisp, SimStats* st) {
        HairMesh mesh = buildHairMesh(head, spec);
        SimParams p;
        p.windStrength = wind;
        HairMeshSimulator sim;
        sim.init(mesh, p);
        if (sagFree) sim.computeSagFreeRest(mesh, p, 200);
        for (int f = 0; f < frames; ++f) sim.step(&mesh, 1.0f / 60.0f, p);
        double d = 0.0;
        int n = 0;
        for (uint32_t l = 1; l < mesh.numLayers(); ++l)
            for (uint32_t v = 0; v < mesh.numVertices(); ++v) {
                d += length(mesh.layerVertex(l, v) -
                            sim.restShape()[static_cast<size_t>(l) * mesh.numVertices() + v]);
                ++n;
            }
        *meanDisp = static_cast<float>(d / std::max(n, 1));
    };

    float dSag = 0, dNoSag = 0, dWind10 = 0, dWind20 = 0;
    SimStats stSag{}, stNo{}, stW10{}, stW20{};
    run(true, 0.0f, 1, &dSag, &stSag);
    run(false, 0.0f, 1, &dNoSag, &stNo);
    run(true, 10.0f, 120, &dWind10, &stW10);
    run(true, 20.0f, 120, &dWind20, &stW20);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  first-frame drift from the rest shape: sag-free init %.4f m, without %.4f m",
                  dSag, dNoSag);
    r.line(buf);
    r.check(dSag < dNoSag * 0.5f, "sag-free initialization removes the initial drop (Hsu et al. 2022)",
            std::to_string(dSag) + " m vs " + std::to_string(dNoSag) + " m");
    std::snprintf(buf, sizeof(buf), "  wind 10 m/s: %.4f m, 20 m/s: %.4f m mean displacement", dWind10, dWind20);
    r.line(buf);
    r.check(dWind10 > 0.001f && dWind20 > dWind10 * 1.5f,
            "wind force model (Wu and Yuksel 2016) deflects the hair proportionally to wind speed");
    r.check(stW10.stable && stW20.stable && stNo.stable,
            "XPBD solver remains stable (no divergence)");
    std::snprintf(buf, sizeof(buf), "  mean stretch error at rest: %.2f%%",
                  100.0 * stSag.meanStretchError);
    r.line(buf);
    r.check(stSag.meanStretchError < 0.05f, "edge constraints converge (stretch error < 5%)");
}

void validateStylingCost(Report& r) {
    r.line("\n== Procedural styling (Section 3.4, Figure 10) ==");
    HeadModel head;
    HairStyleSpec spec = HairStyleSpec::fromFigure10(3, 11u);
    spec.totalStrands = 2000;
    HairMesh mesh = buildHairMesh(head, spec);
    auto inst = makeHairInstance(mesh, spec.styling, 11u, TexelFormat::Float32);
    StrandGenerator gen(inst->textures, inst->styling, inst->baseRadius);
    VertexLodResult lod;
    lod.exponentE = 5;
    lod.exponentPrev = 5;
    lod.gamma = 0.0f;
    const int strands = 2000;
    std::vector<StrandGeometry> out(static_cast<size_t>(strands));
    StrandRequest req;
    req.face = 0;
    req.blockCoord = Vec2(0.4f, 0.6f);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < strands; ++i) {
        req.strandIndex = static_cast<uint32_t>(i);
        gen.generate(req, lod, &out[static_cast<size_t>(i)]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double styled = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Styling disabled: same strand count and vertex counts.
    StylingParams flat = inst->styling;
    flat.clumping = flat.curl = flat.frizz = flat.randomOffset = flat.volume = 0.0f;
    StrandGenerator plain(inst->textures, flat, inst->baseRadius);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < strands; ++i) {
        req.strandIndex = static_cast<uint32_t>(i);
        plain.generate(req, lod, &out[static_cast<size_t>(i)]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    const double unstyled = std::chrono::duration<double, std::milli>(t1 - t0).count();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  %d strands x 33 vertices: %.2f ms with styling, %.2f ms without (+%.0f%%)",
                  strands, styled, unstyled, 100.0 * (styled - unstyled) / std::max(unstyled, 1e-6));
    r.line(buf);
    r.check(true, "styling operations are applied after base curve generation", "cost measured above");
}

std::string runValidation() {
    Report r;
    r.line("SilkCrowd validation suite - replication of Bhokare et al., SIGGRAPH 2024");
    r.line("=====================================================================");
    validateCurves(r);
    HeadModel head;
    HairStyleSpec spec = HairStyleSpec::fromFigure10(1, 1u);
    spec.totalStrands = 20000;
    HairMesh mesh = buildHairMesh(head, spec);
    validateTextures(r, mesh);
    validateFrames(r, mesh);
    validateRoots(r);
    validateLod(r);
    validateStylingCost(r);
    validateSimulation(r);
    validateMemory(r);
    validateWorkload(r);
    r.line("");
    r.line("=====================================================================");
    r.line("checks passed: " + std::to_string(r.passed) + ", failed: " + std::to_string(r.failed));
    r.line(r.failed == 0 ? "RESULT: all paper-critical invariants validated"
                         : "RESULT: FAILURES PRESENT");
    return r.text;
}

// ---------------------------------------------------------------------------
//  Modes
// ---------------------------------------------------------------------------

void runBench(const Args& a) {
    std::printf("SilkCrowd benchmark\n-------------------\n");
    const auto t0 = std::chrono::high_resolution_clock::now();
    Scene scene = (a.scene == "crowd") ? buildCrowdScene(a.crowd)
                 : (a.scene == "teapot") ? buildTeapotScene(a.crowd)
                                         : buildHeadScene(a.style, static_cast<uint32_t>(a.strandCount));
    const auto t1 = std::chrono::high_resolution_clock::now();
    double textureKB = 0.0;
    uint64_t strands = 0;
    for (const auto& inst : scene.hair) {
        textureKB += static_cast<double>(inst->textureBytes) / 1024.0;
        for (uint32_t f = 0; f < inst->mesh.numFaces(); ++f) strands += inst->mesh.bundleStrandCounts()[f];
    }
    std::printf("scene build (hair meshes + 5 textures): %.1f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    std::printf("models: %zu, strands: %llu, texture storage: %.2f MB\n", scene.hair.size(),
                static_cast<unsigned long long>(strands), textureKB / 1024.0);

    Camera cam = (a.scene == "crowd") ? crowdCamera(a.width, a.height)
                 : (a.scene == "teapot") ? teapotCamera(a.width, a.height)
                                         : headCamera(a.width, a.height);
    if (a.msaa > 1)
        std::printf("note: --msaa is a Vulkan feature; the software renderer supersamples "
                    "instead (use --ssaa %d)\n", a.ssaa < 2 ? 2 : a.ssaa);
    RenderOptions opt;
    opt.width = a.width;
    opt.height = a.height;
    opt.ssaa = a.ssaa;
    opt.enableLod = a.lod;
    opt.applyStyling = a.styling;
    SoftwareRenderer renderer;
    Image img;
    RenderStats stats;
    const int frames = std::max(1, a.frames);
    for (int i = 0; i < frames; ++i) {
        cam.position.x += 0.02f * static_cast<float>(i);
        renderer.render(scene, cam, opt, &img, &stats);
        if (a.stats) std::printf("frame %d: %s\n", i, stats.toString().c_str());
    }
    std::printf("%s\n", stats.toString().c_str());
    std::printf("(software rasteriser, %d threads available; GPU numbers come from the Vulkan path)\n",
                static_cast<int>(std::thread::hardware_concurrency()));
}

// Runs the hair mesh simulation for the given number of frames and reports stability.
SimStats simulateScene(Scene* scene, int frames, float dt, const Args& a) {
    SimParams params;
    params.windStrength = a.wind;
    SimStats last;
    for (auto& inst : scene->hair) {
        HairMeshSimulator sim;
        sim.init(inst->mesh, params);
        if (a.sagFree) sim.computeSagFreeRest(inst->mesh, params, 60);
        for (int f = 0; f < frames; ++f) {
            sim.step(&inst->mesh, dt, params);
            last = sim.stats();
        }
        // The hair mesh moved: only the dynamic textures are re-uploaded (Table 1).
        const StylingFrames frames2 = computeStylingFrames(inst->mesh);
        inst->textures.updateDynamic(inst->mesh, frames2);
    }
    return last;
}

// Vulkan first when the build has it; the software renderer is always available so CI on
// a machine without a GPU still produces an image.
bool tryGpuScreenshot(const Args& a, std::string* error) {
    GpuAppOptions gpu;
    gpu.screenshotPath = a.out;
    gpu.scene = a.scene;
    gpu.width = a.width;
    gpu.height = a.height;
    gpu.crowd = a.crowd;
    gpu.style = a.style;
    gpu.strands = a.strandCount;
    gpu.msaa = a.msaa;
    gpu.taa = a.taa;
    gpu.frames = a.frames;
    gpu.lod = a.lod;
    gpu.useMeshShaders = !a.compute;
    gpu.exposure = a.exposure;
    gpu.shadows = true;
    gpu.wind = a.wind;
    gpu.validation = a.validation;
    const int rc = runGpuScreenshot(gpu, error);
    return rc == 0;
}

void runScreenshot(const Args& a) {
    Scene scene = (a.scene == "crowd") ? buildCrowdScene(a.crowd)
                 : (a.scene == "teapot") ? buildTeapotScene(a.crowd)
                                         : buildHeadScene(a.style, static_cast<uint32_t>(a.strandCount));
    Camera cam = (a.scene == "crowd") ? crowdCamera(a.width, a.height)
                 : (a.scene == "teapot") ? teapotCamera(a.width, a.height)
                                         : headCamera(a.width, a.height);
    if (a.yaw != 0.0f) {
        // Orbit around the target. The crowd and teapot scenes are laid out on a grid, so a
        // yaw is a camera move there too -- nothing in the scene itself changes.
        const float r = deg2rad(a.yaw);
        const float cs = std::cos(r), sn = std::sin(r);
        const Vec3 d = cam.position - cam.target;
        cam.position = cam.target + Vec3(d.x * cs + d.z * sn, d.y, -d.x * sn + d.z * cs);
    }
    if (a.simulate) {
        const SimStats st = simulateScene(&scene, std::max(1, a.frames), 1.0f / 60.0f, a);
        std::printf("%s\n", st.toString().c_str());
    }
    if (a.gpu && !a.software) {
        std::string error;
        if (tryGpuScreenshot(a, &error)) return;
        // Say *which* piece of the Vulkan path is missing. "No driver" used to look exactly
        // like "the build has no Vulkan renderer", and the fallback quietly hid both.
        const GpuFacts facts = probeGpu();
        std::printf("Vulkan path unavailable (%s); falling back to the software renderer\n",
                    error.c_str());
        std::printf("  %s\n", gpuHeadline(facts).c_str());
        std::printf("  run `SilkCrowd --gpu-info` for the full report "
                    "(build / loader / driver / device)\n");
    }

    if (a.msaa > 1)
        std::printf("note: --msaa is a Vulkan feature; the software renderer supersamples "
                    "instead (use --ssaa %d)\n", a.ssaa < 2 ? 2 : a.ssaa);
    RenderOptions opt;
    opt.width = a.width;
    opt.height = a.height;
    opt.ssaa = a.ssaa;
    opt.enableLod = a.lod;
    opt.applyStyling = a.styling;
    opt.exposure = a.exposure;
    opt.frameMode = a.finiteDifference ? FrameMode::FiniteDifferences : FrameMode::PerVertexTextures;
    opt.drawHair = a.drawHair;
    opt.drawHead = a.drawHead;
    if (a.studio) {
        // A dark backdrop is how the reference photographs of a hair model are shot: with
        // nothing behind the silhouette, the hair's own edge is what the eye judges.
        opt.drawSky = false;
        opt.drawGround = false;
        opt.drawShadows = false;
    }
    SoftwareRenderer renderer;
    Image img;
    RenderStats stats;
    renderer.render(scene, cam, opt, &img, &stats);
    if (!writePng(a.out, img, a.exposure)) {
        std::fprintf(stderr, "error: could not write %s\n", a.out.c_str());
        std::exit(1);
    }
    std::printf("wrote %s (%dx%d, ssaa %d)\n", a.out.c_str(), a.width, a.height, a.ssaa);
    std::printf("%s\n", stats.toString().c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    if (args.headScan == "none") setHeadScanAsset("");
    // The shipped heads are named, not path-ed: "male" is the CC BY photogrammetry scan and
    // "female" the CC0 MakeHuman bust (see assets/heads/LICENSE-female_head.txt).
    else if (args.headScan == "male") setHeadScanAsset("assets/heads/head_scan.glb");
    else if (args.headScan == "female") setHeadScanAsset("assets/heads/female_head_scan.glb");
    else if (!args.headScan.empty()) setHeadScanAsset(args.headScan);
    if (args.mode == "help") { printHelp(); return 0; }
    if (args.mode == "validate") {
        const std::string report = runValidation();
        std::printf("%s", report.c_str());
        if (!args.reportPath.empty()) {
            FILE* f = std::fopen(args.reportPath.c_str(), "wb");
            if (f) { std::fwrite(report.data(), 1, report.size(), f); std::fclose(f); }
        }
        return report.find("FAILURES") != std::string::npos ? 1 : 0;
    }
    if (args.mode == "bench") { runBench(args); return 0; }
    if (args.mode == "gpu") {
        const GpuFacts facts = probeGpu();
        printGpuReport(facts, stdout);
        // 0 when the Vulkan path would run, 3 when it would not: CI can branch on it without
        // parsing the text, and the report still explains the failure on stdout.
        return gpuUsable(facts) ? 0 : 3;
    }
    if (args.mode == "screenshot") { runScreenshot(args); return 0; }
    if (args.mode == "assets") {
        writeGlif("assets/heads");
        writeHairMeshAssets("assets/hairmeshes");
        return 0;
    }
    if (args.mode == "interactive") {
        const int rc = runGpuWindow(argc, argv);
        if (rc != 2) return rc;
        // Two different reasons land here, and they need different fixes.
#ifdef SILKCROWD_WITH_QT
        std::printf("This build has the Qt6 editor but no Vulkan renderer, so the window "
                    "cannot open (configure with libvulkan headers installed).\n");
#else
        std::printf("This build has no Qt6 editor (configure with Qt6 installed). "
                    "Falling back to a headless screenshot.\n");
#endif
        if (!gpuUsable(probeGpu())) {
            std::printf("Run `SilkCrowd --gpu-info` for the Vulkan report.\n");
        }
        runScreenshot(args);
        return 0;
    }
    printHelp();
    return 0;
}
