#include "silkcrowd/renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "silkcrowd/root_placement.h"

namespace silk {
namespace {

const SampleSet2D& sharedSampleSet(bool triangleDomain) {
    static SampleSet2D quad;
    static SampleSet2D tri;
    SampleSet2D& target = triangleDomain ? tri : quad;
    if (target.size() == 0) {
        SampleEliminationParams p;
        p.targetCount = 1024;
        p.oversample = 3;
        p.seed = triangleDomain ? 192837465u : 987654321u;
        p.triangleDomain = triangleDomain;
        target = generateBlueNoiseSamples(p);
    }
    return target;
}

inline Vec3 transformDir(const Mat4& m, Vec3 v) {
    return Vec3(m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
                m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
                m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z);
}

// Analytic sky: zenith/horizon gradient plus a sun aureole (the GPU path evaluates the
// full Hosek-Wilkie model; this keeps the CPU preview dependency free).
inline Vec3 skyColor(const Scene& s, Vec3 dir) {
    const float h = clampf(dir.y, -1.0f, 1.0f);
    const float t = std::pow(saturate(1.0f - std::fabs(h)), 1.7f);
    Vec3 c = lerp(s.skyZenith, s.skyHorizon, t);
    if (h < 0.0f) c = lerp(s.skyHorizon * 0.62f, s.groundColor * 1.1f, saturate(-h * 2.2f));
    const float mu = saturate(dot(dir, s.sunDirection));
    const float glow = std::pow(mu, 400.0f) * 12.0f + std::pow(mu, 24.0f) * 0.55f +
                       std::pow(mu, 5.0f) * 0.16f;
    return c + s.sunColor * glow;
}

// Kajiya-Kay diffuse plus two Marschner-style longitudinal lobes.
inline Vec3 shadeHair(Vec3 tangent, Vec3 position, Vec3 viewDir, const Scene& scene,
                      const StylingParams& sp, float w, float ao) {
    const Vec3 T = normalize(tangent);
    const Vec3 L = scene.sunDirection;
    const Vec3 V = normalize(viewDir);
    const Vec3 H = normalize(L + V);
    (void)position;

    const Vec3 rootTint = sp.color * 0.45f;
    const Vec3 albedo = lerp(rootTint, sp.color, saturate(w * 1.3f + 0.1f));

    const float sinTL = std::sqrt(saturate(1.0f - dot(T, L) * dot(T, L)));
    Vec3 color = albedo * scene.sunColor * (sinTL * scene.sunIntensity * 0.22f * ao);

    // Shifted-tangent specular lobes: R towards the root, TRT towards the tip.
    auto lobe = [&](float shift, float exponent, float gain) {
        Vec3 ts = normalize(T + shift * normalize(cross(cross(T, V), T) + Vec3(1e-6f, 0, 0)));
        const float sinTH = std::sqrt(saturate(1.0f - dot(ts, H) * dot(ts, H)));
        return std::pow(sinTH, exponent) * gain;
    };
    const float spec = lobe(0.14f, 110.0f, 0.55f) + lobe(-0.18f, 34.0f, 0.22f);
    const Vec3 specColor = lerp(Vec3(1.0f, 0.96f, 0.90f), sp.color * 3.0f + Vec3(0.1f), 0.45f);
    color += specColor * scene.sunColor * (spec * scene.sunIntensity * 0.22f * ao);

    // Sky ambient (approximate IBL).
    Vec3 nApprox = normalize(cross(T, normalize(cross(T, V) + Vec3(1e-5f, 0, 0))));
    if (dot(nApprox, V) < 0.0f) nApprox = -nApprox;
    const float up = saturate(nApprox.y * 0.5f + 0.5f);
    const Vec3 ambient = lerp(scene.skyHorizon * 0.30f, scene.skyZenith * 0.85f, up);
    color += albedo * ambient * (0.30f + 0.35f * ao);
    return color;
}

struct Projected {
    float x = 0, y = 0, z = 0;
    bool visible = false;
};

struct HairVertexOut {
    float x = 0, y = 0, z = 0;
    Vec3 pos;
    float w = 0;
    Vec3 color;
};

}  // namespace

std::string RenderStats::toString() const {
    std::ostringstream os;
    os << "frame " << frameMs << " ms (" << generationMs << " ms generate, " << rasterMs
       << " ms raster) | strands " << strandsRendered << "/" << strandsTotal << " | strand verts "
       << strandVertices << " | tris " << triangles << " | vertex-LOD 2^"
       << (minVertexExponent <= maxVertexExponent ? minVertexExponent : -1) << "..2^"
       << maxVertexExponent;
    return os.str();
}

std::shared_ptr<HairInstance> makeHairInstance(const HairMesh& mesh, const StylingParams& styling,
                                               uint32_t seed, TexelFormat format) {
    auto inst = std::make_shared<HairInstance>();
    inst->mesh = mesh;
    inst->styling = styling;
    inst->seed = seed;
    inst->format = format;
    inst->transform = Mat4::identity();

    const StylingFrames frames = computeStylingFrames(mesh);
    inst->textures.build(mesh, frames, format);
    inst->textureBytes = inst->textures.sizeReport(format).total();

    bundleScalpMetrics(mesh, &inst->bundleSize, &inst->bundleCenter);
    // Per-bundle strand length (h of the vertex-LOD Eq. 7): the distance from the root
    // layer vertex to the tip layer vertex of the bundle.
    Vec3 lo, hi;
    mesh.bounds(&lo, &hi);
    inst->bundleLength.assign(mesh.numFaces(), std::max(hi.y - lo.y, 1e-3f));
    inst->bundleVisible.assign(mesh.numFaces(), 1);

    inst->center = (lo + hi) * 0.5f;
    inst->radius = length(hi - lo) * 0.5f;
    return inst;
}

void SoftwareRenderer::render(const Scene& scene, const Camera& camera,
                              const RenderOptions& options, Image* out, RenderStats* stats) {
    using clock = std::chrono::high_resolution_clock;
    const auto tStart = clock::now();

    // The progressive blue noise root sample sets (Section 3.3) are generated once per
    // process: any prefix of the set is blue noise, so every bundle and every strand
    // count reuses the same table (this is what the paper does to keep it cacheable).
    if (sampleSetQuad_.size() == 0) {
        sampleSetQuad_ = sharedSampleSet(false);
        sampleSetTriangle_ = sharedSampleSet(true);
    }

    const int ss = std::max(1, options.ssaa);
    bw_ = options.width * ss;
    bh_ = options.height * ss;
    buffer_.assign(static_cast<size_t>(bw_) * bh_, Sample{});
    opaqueDepth_.assign(static_cast<size_t>(bw_) * bh_, 1e30f);
    out->resize(options.width, options.height);

    Camera cam = camera;
    cam.aspect = static_cast<float>(options.width) / static_cast<float>(options.height);
    const Mat4 view = cam.view();
    const Mat4 proj = cam.proj();
    const Mat4 viewProj = proj * view;
    const Vec3 camPos = cam.position;
    const Vec3 camRight = normalize(cross(cam.forward(), cam.up));
    const Vec3 camUp = cross(camRight, cam.forward());
    const float tanHalf = std::tan(cam.fovY * 0.5f);

    // ---------------- background + ground ----------------
    for (int y = 0; y < bh_; ++y) {
        const float sy = ((y + 0.5f) / bh_) * 2.0f - 1.0f;  // -1 at the top
        for (int x = 0; x < bw_; ++x) {
            const float sx = ((x + 0.5f) / bw_) * 2.0f - 1.0f;
            Vec3 dir = normalize(cam.forward() + camRight * (sx * tanHalf * cam.aspect) -
                                 camUp * (sy * tanHalf));
            Vec3 color = options.drawSky ? skyColor(scene, dir) : Vec3(0.015f, 0.016f, 0.02f);
            Sample& s = buffer_[static_cast<size_t>(y) * bw_ + x];

            // The floor sits just below the neck: the head model is centred on the origin,
            // so a plane through y = 0 would cut the chin off.
            constexpr float kGroundY = -0.118f;
            if (options.drawGround && dir.y < -1e-4f) {
                const float t = -(camPos.y - kGroundY) / dir.y;
                if (t > 0.0f && t < 200.0f) {
                    const Vec3 hit = camPos + dir * t;
                    const float fade = saturate(1.0f - t / 16.0f);
                    Vec3 g = scene.groundColor * (0.55f + 0.45f * std::exp(-t * 0.25f));
                    auto lineMask = [](float v) {
                        const float f = std::fabs(v - std::round(v));
                        return 1.0f - saturate(f * 30.0f);
                    };
                    const float grid = std::max(lineMask(hit.x / 0.25f), lineMask(hit.z / 0.25f));
                    const float gridBig = std::max(lineMask(hit.x / 1.0f), lineMask(hit.z / 1.0f));
                    g = lerp(g, Vec3(0.40f, 0.44f, 0.50f), grid * 0.28f * fade);
                    g = lerp(g, Vec3(0.55f, 0.60f, 0.66f), gridBig * 0.35f * fade);
                    if (options.drawShadows) {
                        // Soft contact shadow of the model on the floor.
                        const float d = length(Vec2(hit.x, hit.z));
                        const float occ = 1.0f - 0.65f * std::exp(-d * d / 0.035f);
                        g = g * occ;
                    }
                    color = lerp(color, g, saturate(fade + 0.15f));
                    Vec3 clip; float cw;
                    viewProj.transformPoint(hit, &clip, &cw);
                    if (cw > 1e-5f) s.depth = clip.z / cw;
                }
            }
            s.color = color;
        }
    }

    auto project = [&](Vec3 p, Projected* pr) {
        Vec3 clip; float w;
        viewProj.transformPoint(p, &clip, &w);
        if (w <= 1e-6f) { pr->visible = false; return; }
        const float invW = 1.0f / w;
        pr->x = (clip.x * invW * 0.5f + 0.5f) * bw_;
        pr->y = (clip.y * invW * 0.5f + 0.5f) * bh_;
        pr->z = clip.z * invW;
        pr->visible = true;
    };

    // ---------------- opaque geometry ----------------
    auto drawMesh = [&](const Mesh& mesh, Vec3 baseAlbedo, float specPower, float specGain,
                        bool isSkin, bool isEye = false) {
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vec3 p[3], n[3];
            Projected pr[3];
            for (int k = 0; k < 3; ++k) {
                const uint32_t idx = mesh.indices[t + k];
                p[k] = mesh.positions[idx];
                n[k] = mesh.normals.empty() ? Vec3(0, 1, 0) : mesh.normals[idx];
                project(p[k], &pr[k]);
            }
            if (!pr[0].visible && !pr[1].visible && !pr[2].visible) continue;
            const float area = (pr[1].x - pr[0].x) * (pr[2].y - pr[0].y) -
                               (pr[2].x - pr[0].x) * (pr[1].y - pr[0].y);
            if (std::fabs(area) < 1e-9f) continue;
            const float invArea = 1.0f / area;
            const int x0 = std::max(0, static_cast<int>(std::floor(std::min({pr[0].x, pr[1].x, pr[2].x}))));
            const int x1 = std::min(bw_ - 1, static_cast<int>(std::ceil(std::max({pr[0].x, pr[1].x, pr[2].x}))));
            const int y0 = std::max(0, static_cast<int>(std::floor(std::min({pr[0].y, pr[1].y, pr[2].y}))));
            const int y1 = std::min(bh_ - 1, static_cast<int>(std::ceil(std::max({pr[0].y, pr[1].y, pr[2].y}))));
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const float px = x + 0.5f, py = y + 0.5f;
                    const float b0 = ((pr[1].x - px) * (pr[2].y - py) - (pr[2].x - px) * (pr[1].y - py)) * invArea;
                    const float b1 = ((pr[2].x - px) * (pr[0].y - py) - (pr[0].x - px) * (pr[2].y - py)) * invArea;
                    const float b2 = 1.0f - b0 - b1;
                    if (b0 < 0 || b1 < 0 || b2 < 0) continue;
                    const float z = pr[0].z * b0 + pr[1].z * b1 + pr[2].z * b2;
                    Sample& s = buffer_[static_cast<size_t>(y) * bw_ + x];
                    if (z >= s.depth) continue;
                    const Vec3 nrm = normalize(n[0] * b0 + n[1] * b1 + n[2] * b2);
                    const Vec3 pos = p[0] * b0 + p[1] * b1 + p[2] * b2;
                    const Vec3 V = normalize(camPos - pos);
                    Vec3 alb = baseAlbedo;
                    const float ndl = saturate(dot(nrm, scene.sunDirection));
                    float wrap = ndl;
                    if (isEye) {
                        // Sclera -> iris -> pupil, driven by how much the eye normal faces
                        // the camera (the eyes are spheres pointing down +Z).
                        const float f = saturate(dot(nrm, normalize(Vec3(0.0f, 0.0f, 1.0f))));
                        alb = Vec3(0.92f, 0.90f, 0.88f);
                        alb = lerp(alb, Vec3(0.22f, 0.15f, 0.09f), smoothstepf(0.55f, 0.80f, f));
                        alb = lerp(alb, Vec3(0.02f, 0.02f, 0.02f), smoothstepf(0.93f, 0.975f, f));
                        specGain = 1.4f;
                    } else if (isSkin) {
                        alb = Vec3(0.66f, 0.47f, 0.39f);
                        wrap = saturate((dot(nrm, scene.sunDirection) + 0.40f) / 1.40f);
                        // Lips: a soft ellipse rather than a rectangle, with a darker
                        // crease between the upper and lower lip.
                        const float mx = pos.x / 0.0245f;
                        const float my = (pos.y + 0.0305f) / 0.0105f;
                        const float r2 = mx * mx + my * my;
                        if (nrm.z > 0.30f && r2 < 1.0f) {
                            const float t = (1.0f - smoothstepf(0.25f, 1.0f, r2));
                            alb = lerp(alb, Vec3(0.52f, 0.25f, 0.24f), t * 0.85f);
                            if (r2 < 0.10f) alb = lerp(alb, Vec3(0.34f, 0.15f, 0.15f), 0.75f);
                        }
                    }
                    Vec3 color = alb * scene.sunColor * (wrap * scene.sunIntensity * 0.55f);
                    (void)0;
                    const Vec3 amb = lerp(scene.skyHorizon * 0.38f, scene.skyZenith * 0.85f,
                                          saturate(nrm.y * 0.5f + 0.5f));
                    color += alb * amb * 0.9f;
                    const Vec3 H = normalize(scene.sunDirection + V);
                    color += scene.sunColor * (std::pow(saturate(dot(nrm, H)), specPower) * specGain);
                    s.color = color;
                    s.depth = z;
                }
            }
        }
    };

    if (options.drawHead) {
        drawMesh(scene.head, Vec3(0.66f, 0.47f, 0.39f), 26.0f, 0.08f, true);
        drawMesh(scene.body, Vec3(0.56f, 0.41f, 0.34f), 26.0f, 0.08f, true);
        drawMesh(scene.eyes, Vec3(0.16f, 0.11f, 0.07f), 120.0f, 0.9f, false, /*isEye=*/true);
    }
    if (scene.useTeapots) drawMesh(scene.teapot, Vec3(0.86f, 0.84f, 0.80f), 90.0f, 0.45f, false);

    // Freeze the opaque depth buffer. Hair is then blended back-to-front *without*
    // writing depth, so that the many transparent layers of a hairstyle accumulate
    // (a single depth-tested pass would keep only the front shell, like Figure 4a's
    // artifact -- hair needs order-dependent compositing).
    for (size_t i = 0; i < buffer_.size(); ++i) opaqueDepth_[i] = buffer_[i].depth;

    // ---------------- hair ----------------
    const auto tGen0 = clock::now();
    double rasterMs = 0.0;
    uint64_t strandsRendered = 0, strandsTotal = 0, strandVertices = 0, triangles = 0;
    int minE = 99, maxE = -1, bundlesRendered = 0;
    double lodSum = 0.0;
    int lodCount = 0;

    const float ssScale = static_cast<float>(ss);
    const float minWidthPx = 0.70f * ssScale;
    const float maxWidthPx = 2.60f * ssScale;

    std::vector<StrandGeometry> geom;
    std::vector<uint32_t> order;

    for (const auto& instPtr : scene.hair) {
        const HairInstance& inst = *instPtr;
        const uint32_t numFaces = inst.mesh.numFaces();
        for (uint32_t f = 0; f < numFaces; ++f) strandsTotal += inst.mesh.bundleStrandCounts()[f];
        if (!options.drawHair) continue;

        order.resize(numFaces);
        for (uint32_t i = 0; i < numFaces; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const Vec3 wa = inst.transform.transformPoint(inst.bundleCenter[a]);
            const Vec3 wb = inst.transform.transformPoint(inst.bundleCenter[b]);
            return length2(wa - camPos) > length2(wb - camPos);  // back to front
        });

        StrandGenerator generator(inst.textures, inst.styling, inst.baseRadius);

        for (uint32_t f : order) {
            const uint32_t strandsInBundle = inst.mesh.bundleStrandCounts()[f];
            const Vec3 bundleWorld = inst.transform.transformPoint(inst.bundleCenter[f]);
            const float dist = lodDistance(camPos, bundleWorld, inst.bundleSize[f] * 0.5f);

            // --- strand LOD (Eq. 7 + Eq. 8) ---
            float Ls = 1.0f;
            if (options.enableLod && options.strandLod) {
                Ls = detailLevel(inst.bundleSize[f], static_cast<float>(options.height),
                                 dist * options.lodBias, cam.fovY, options.lod.alphaStrand);
            }
            StrandLodResult strandLod;
            if (options.enableLod && options.strandLod) {
                strandLod = computeStrandLod(strandsInBundle, bundleDelta(f, inst.seed), Ls);
            } else {
                strandLod.numStrands = strandsInBundle;
                strandLod.thicknessScale = 1.0f;
            }

            // --- vertex LOD (Eq. 7 + Eq. 9-12) ---
            Vec3 lo, hi;
            inst.mesh.bounds(&lo, &hi);
            const float strandLength = std::max(hi.y - lo.y, 1e-3f);
            float Lv = 1.0f;
            if (options.enableLod && options.vertexLod) {
                Lv = detailLevel(strandLength, static_cast<float>(options.height),
                                 dist * options.lodBias, cam.fovY, options.lod.alphaVertex);
            }
            VertexLodResult vertexLod = computeVertexLod(options.enableLod && options.vertexLod ? Lv : 1.0f,
                                                         32, options.lod.lambda);
            if (!options.enableLod || !options.vertexLod) {
                vertexLod.exponentE = 5;
                vertexLod.exponentPrev = 5;
                vertexLod.gamma = 0.0f;
            }
            if (options.fixedVertexExponent >= 0) {
                vertexLod.exponentE = options.fixedVertexExponent;
                vertexLod.exponentPrev = options.fixedVertexExponent;
                vertexLod.gamma = 0.0f;
            }

            lodSum += Lv;
            ++lodCount;
            minE = std::min(minE, vertexLod.exponentE);
            maxE = std::max(maxE, vertexLod.exponentE);
            ++bundlesRendered;

            geom.resize(strandLod.numStrands);
            for (uint32_t s = 0; s < strandLod.numStrands; ++s) {
                StrandRequest req;
                req.face = f;
                // Progressive blue noise samples; the per-bundle offset gives variety
                // without breaking the blue noise property of each prefix.
                const Vec2 xi = sampleSetQuad_.at(static_cast<size_t>(s) + f * 131u);
                req.blockCoord = inst.mesh.face(f).isTriangle()
                                     ? rootPlacementTriangle(sampleSetTriangle_.at(static_cast<size_t>(s) + f * 97u))
                                     : rootPlacementQuad(xi);
                req.blockCoord.x = clampf(req.blockCoord.x, 0.002f, 0.998f);
                req.blockCoord.y = clampf(req.blockCoord.y, 0.002f, 0.998f);
                req.strandIndex = s;
                req.bundleIndex = f;
                req.thicknessScale = strandLod.thicknessScale;
                generator.generate(req, vertexLod, &geom[s]);
                strandVertices += geom[s].vertices.size();
                triangles += 2ull * static_cast<uint64_t>(vertexLod.numSegments());
                ++strandsRendered;
            }

            const auto tR = clock::now();
            for (uint32_t s = 0; s < strandLod.numStrands; ++s) {
                const StrandGeometry& g = geom[s];
                const int n = static_cast<int>(g.vertices.size());
                if (n < 2) continue;

                // Shading is evaluated once per strand vertex (this is exactly what the
                // mesh shader does: it shades the 2*ns+2 strip vertices it emits) and
                // interpolated across the ribbon, instead of per fragment.
                std::vector<Vec3> vcolor(static_cast<size_t>(n));
                for (int k = 0; k < n; ++k) {
                    const Vec3 p = inst.transform.transformPoint(g.vertices[k].position);
                    const Vec3 t = normalize(transformDir(inst.transform, g.vertices[k].tangent));
                    const Vec3 V = normalize(camPos - p);
                    const float ao = 0.5f + 0.5f * saturate(g.vertices[k].w * 1.5f);
                    vcolor[static_cast<size_t>(k)] =
                        shadeHair(t, p, V, scene, inst.styling, g.vertices[k].w, ao);
                }
                Projected prevA{}, prevB{};
                Vec3 prevPA, prevPB;
                Vec3 prevCA, prevCB;
                Vec3 tangent = transformDir(inst.transform, g.vertices[0].tangent);
                for (int k = 0; k < n; ++k) {
                    const Vec3 p = inst.transform.transformPoint(g.vertices[k].position);
                    const Vec3 t = normalize(transformDir(inst.transform, g.vertices[k].tangent));
                    const Vec3 toCam = normalize(camPos - p);
                    Vec3 side = cross(t, toCam);
                    if (length2(side) < 1e-14f) side = camRight;
                    side = normalize(side);
                    Vec3 offset = side * g.radius;

                    // Screen-space width clamp (keeps far LOD strands ~1-3 px wide).
                    Projected pc{}, po{};
                    project(p, &pc);
                    project(p + offset, &po);
                    if (pc.visible && po.visible) {
                        const float pix = std::sqrt((po.x - pc.x) * (po.x - pc.x) +
                                                    (po.y - pc.y) * (po.y - pc.y));
                        if (pix > maxWidthPx) offset = offset * (maxWidthPx / pix);
                        else if (pix < minWidthPx) offset = offset * (minWidthPx / std::max(pix, 1e-5f));
                    }
                    const Vec3 pa = p - offset, pb = p + offset;
                    Projected pra{}, prb{};
                    project(pa, &pra);
                    project(pb, &prb);

                    if (k > 0 && pra.visible && prb.visible && prevA.visible && prevB.visible) {
                        const Vec3 c0 = vcolor[static_cast<size_t>(k - 1)];
                        const Vec3 c1 = vcolor[static_cast<size_t>(k)];
                        const HairVertexOut quad[4] = {
                            {prevA.x, prevA.y, prevA.z, prevPA, g.vertices[k - 1].w, prevCA},
                            {prevB.x, prevB.y, prevB.z, prevPB, g.vertices[k - 1].w, prevCB},
                            {prb.x, prb.y, prb.z, pb, g.vertices[k].w, c1},
                            {pra.x, pra.y, pra.z, pa, g.vertices[k].w, c0}};
                        static const int triIdx[2][3] = {{0, 1, 2}, {0, 2, 3}};
                        for (int ti = 0; ti < 2; ++ti) {
                            const HairVertexOut& v0 = quad[triIdx[ti][0]];
                            const HairVertexOut& v1 = quad[triIdx[ti][1]];
                            const HairVertexOut& v2 = quad[triIdx[ti][2]];
                            const float area = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
                            if (std::fabs(area) < 1e-9f) continue;
                            const float invArea = 1.0f / area;
                            const int x0 = std::max(0, static_cast<int>(std::floor(std::min({v0.x, v1.x, v2.x}))));
                            const int x1 = std::min(bw_ - 1, static_cast<int>(std::ceil(std::max({v0.x, v1.x, v2.x}))));
                            const int y0 = std::max(0, static_cast<int>(std::floor(std::min({v0.y, v1.y, v2.y}))));
                            const int y1 = std::min(bh_ - 1, static_cast<int>(std::ceil(std::max({v0.y, v1.y, v2.y}))));
                            if (x1 < x0 || y1 < y0) continue;
                            for (int y = y0; y <= y1; ++y) {
                                for (int x = x0; x <= x1; ++x) {
                                    const float px = x + 0.5f, py = y + 0.5f;
                                    const float b0 = ((v1.x - px) * (v2.y - py) - (v2.x - px) * (v1.y - py)) * invArea;
                                    const float b1 = ((v2.x - px) * (v0.y - py) - (v0.x - px) * (v2.y - py)) * invArea;
                                    const float b2 = 1.0f - b0 - b1;
                                    if (b0 < -0.002f || b1 < -0.002f || b2 < -0.002f) continue;
                                    const float z = v0.z * b0 + v1.z * b1 + v2.z * b2;
                                    const size_t pix = static_cast<size_t>(y) * bw_ + x;
                                    if (z > opaqueDepth_[pix]) continue;  // hidden by skin
                                    Sample& sm = buffer_[pix];
                                    const Vec3 c = v0.color * b0 + v1.color * b1 + v2.color * b2;
                                    const float a = options.hairAlpha;
                                    sm.color = sm.color * (1.0f - a) + c * a;
                                }
                            }
                        }
                    }
                    prevA = pra;
                    prevB = prb;
                    prevPA = pa;
                    prevPB = pb;
                    prevCA = prevCB = vcolor[static_cast<size_t>(k)];
                    tangent = t;
                }
            }
            rasterMs += std::chrono::duration<double, std::milli>(clock::now() - tR).count();
        }
    }
    const auto tGen1 = clock::now();

    // ---------------- resolve (supersampling) ----------------
    for (int y = 0; y < options.height; ++y) {
        for (int x = 0; x < options.width; ++x) {
            Vec3 acc(0, 0, 0);
            for (int sy = 0; sy < ss; ++sy)
                for (int sx = 0; sx < ss; ++sx)
                    acc += buffer_[static_cast<size_t>(y * ss + sy) * bw_ + x * ss + sx].color;
            out->at(x, y) = acc * (1.0f / static_cast<float>(ss * ss));
        }
    }

    const auto tEnd = clock::now();
    if (stats) {
        stats->generationMs = std::chrono::duration<double, std::milli>(tGen1 - tGen0).count();
        stats->rasterMs = rasterMs;
        stats->frameMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        stats->strandsRendered = strandsRendered;
        stats->strandsTotal = strandsTotal;
        stats->strandVertices = strandVertices;
        stats->triangles = triangles;
        stats->minVertexExponent = minE;
        stats->maxVertexExponent = maxE;
        stats->averageDetailLevel = lodCount > 0 ? static_cast<float>(lodSum / lodCount) : 0.0f;
        stats->bundlesRendered = bundlesRendered;
    }
}

}  // namespace silk
