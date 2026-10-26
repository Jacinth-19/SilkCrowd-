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

    // Roots read darker than tips (light penetrates the fibre at the ends, and dye is
    // always densest at the root). This root-to-tip ramp is most of what makes dyed hair
    // read as hair rather than as coloured plastic.
    const Vec3 rootTint = sp.color * Vec3(0.52f, 0.50f, 0.50f);
    const Vec3 tipTint = sp.color * Vec3(1.16f, 1.14f, 1.10f);
    const Vec3 albedo = lerp(rootTint, tipTint, saturate(w * 1.15f + 0.15f));

    const float sinTL = std::sqrt(saturate(1.0f - dot(T, L) * dot(T, L)));
    Vec3 color = albedo * scene.sunColor * (sinTL * scene.sunIntensity * 0.22f * ao);

    // Shifted-tangent specular lobes: R towards the root, TRT towards the tip.
    auto lobe = [&](float shift, float exponent, float gain) {
        Vec3 ts = normalize(T + shift * normalize(cross(cross(T, V), T) + Vec3(1e-6f, 0, 0)));
        const float sinTH = std::sqrt(saturate(1.0f - dot(ts, H) * dot(ts, H)));
        return std::pow(sinTH, exponent) * gain;
    };
    // Wide longitudinal lobes. A narrow exponent turns every ribbon into an independent
    // speck of light (which reads as straw); a wide R lobe and a broad TRT lobe instead
    // merge into the continuous highlight band that runs around a glossy hairstyle.
    // The gains are deliberately small: a hairstyle composites dozens of semi-transparent
    // layers, so a lobe that looks fine on one ribbon accumulates into a blown-out patch
    // where the surface faces the key light. The wide lobe supplies the band, the narrow
    // one the glints.
    const float spec = lobe(0.12f, 78.0f, 0.30f) + lobe(-0.16f, 22.0f, 0.13f) +
                       lobe(0.04f, 6.0f, 0.025f);
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

// Cheap value noise, used for skin mottling and hair-colour breakup. Deterministic, so a
// render is reproducible.
inline float hash13(Vec3 p) {
    float v = std::sin(p.x * 127.1f + p.y * 311.7f + p.z * 74.7f) * 43758.5453f;
    return v - std::floor(v);
}
inline float valueNoise(Vec3 p) {
    const Vec3 i(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    const Vec3 f(p.x - i.x, p.y - i.y, p.z - i.z);
    const Vec3 u(f.x * f.x * (3.0f - 2.0f * f.x), f.y * f.y * (3.0f - 2.0f * f.y),
                 f.z * f.z * (3.0f - 2.0f * f.z));
    float acc = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const Vec3 c = i + Vec3(static_cast<float>(dx), static_cast<float>(dy),
                                        static_cast<float>(dz));
                const float w = (dx ? u.x : 1.0f - u.x) * (dy ? u.y : 1.0f - u.y) *
                                (dz ? u.z : 1.0f - u.z);
                acc += w * hash13(c);
            }
    return acc;
}

// Skin: subsurface-ish diffuse (a wrap term plus the red scattering at the terminator),
// a two-lobe specular (the oily sheen of the T-zone and a broad, soft lobe), and the
// analytic face masks for brows, lips, lashes, nostrils and creases.
inline Vec3 shadeSkin(Vec3 localPos, Vec3 nrm, Vec3 viewDir, const Scene& scene,
                      const Vec3& baseAlbedo, float roughPower, float specGain,
                      const FaceMasks* masks) {
    // |masks| is null for the neck and shoulders: the same skin model, no face features.
    const FaceMasks none;
    const FaceMasks& fm = masks ? *masks : none;
    const Vec3 L = scene.sunDirection;
    const Vec3 V = normalize(viewDir);

    // Base complexion with a little mottling so the skin is not a flat plastic colour.
    const float mott = valueNoise(localPos * 900.0f) * 0.6f + valueNoise(localPos * 240.0f) * 0.4f;
    Vec3 alb = baseAlbedo * (0.96f + 0.07f * mott);
    alb = lerp(alb, Vec3(0.76f, 0.42f, 0.36f), fm.blush * 0.42f);
    alb = lerp(alb, Vec3(0.64f, 0.29f, 0.28f), saturate(fm.lipUpper * 1.1f));
    alb = lerp(alb, Vec3(0.60f, 0.24f, 0.25f), saturate(fm.lipLower * 1.1f));
    // Eyebrow hairs are separate strands, not a painted band: a high-frequency streak
    // pattern along the brow gives the feathered, directional look of real brow hair.
        // Two frequencies keep the streaks from looking like a printed comb; the strands run
    // up and outwards, so the pattern is stretched along y and tight along x.
    const float browHair = 0.52f + 0.48f *
        (0.5f + 0.5f * std::sin(localPos.x * 17000.0f + localPos.y * 7600.0f +
                                6.0f * valueNoise(localPos * 2600.0f))) *
        (0.70f + 0.30f * valueNoise(localPos * 700.0f + Vec3(11.0f, 3.0f, 7.0f)));
    alb = lerp(alb, Vec3(0.22f, 0.13f, 0.09f), fm.brow * 0.86f * browHair);
    // The nose: the dorsum is lighter where it faces the key, the side walls darker.
    alb = lerp(alb, alb * 1.10f, fm.noseBridge * 0.55f);
    alb = lerp(alb, alb * Vec3(0.78f, 0.74f, 0.74f), fm.noseSide * 0.85f);
    alb = lerp(alb, Vec3(0.07f, 0.04f, 0.035f), fm.nostril * 0.9f);
    alb = lerp(alb, Vec3(0.26f, 0.11f, 0.12f), fm.mouthLine * 0.95f);
    // Scalp: dark, slightly cooler than the hair it is under, and much rougher.
    alb = lerp(alb, Vec3(0.028f, 0.017f, 0.012f), fm.scalp * 0.97f);

    // ---- studio rig: a warm key (the scene sun), a cool fill from the opposite side and
    // a rim from behind, which is what separates the head from the background.
    const Vec3 keyDir = normalize(L + Vec3(0.0f, 0.15f, 0.0f));
    const Vec3 fillDir = normalize(Vec3(-keyDir.x * 0.9f, 0.25f, -keyDir.z * 0.35f + 0.55f));
    const Vec3 rimDir = normalize(Vec3(-keyDir.x * 0.6f, 0.45f, -keyDir.z));

    // Diffuse: wrap the light around the form (skin scatters) and push red where the
    // surface turns away from the sun, which is the single biggest cue for skin.
    const float ndl = dot(nrm, keyDir);
    const float wrapDiff = saturate((ndl + 0.42f) / 1.42f);
    const float terminator = std::exp(-ndl * ndl * 6.0f);
    Vec3 diffuse = alb * wrapDiff + Vec3(0.34f, 0.09f, 0.06f) * terminator * 0.26f;
    diffuse *= scene.sunIntensity * 0.55f;

    // Cool bounce on the shadow side keeps the form readable instead of crushing to black.
    const float fillNdl = saturate(dot(nrm, fillDir));
    diffuse += alb * scene.skyHorizon * (fillNdl * 0.30f + 0.06f);

    // Specular: the oily T-zone lobe plus a broad sheen; the lips are glossier still.
    const Vec3 H = normalize(keyDir + V);
    const float nh = saturate(dot(nrm, H));
    const float f0 = 0.028f;
    const float fres = f0 + (1.0f - f0) * std::pow(1.0f - saturate(dot(nrm, V)), 5.0f);
    const float gloss = 1.0f + 2.5f * saturate(fm.lipUpper + fm.lipLower);
    const float spec = (std::pow(nh, roughPower) * 0.55f + std::pow(nh, 10.0f) * 0.05f) *
                       specGain * gloss;

    const float ao = 1.0f - 0.72f * fm.ao;
    Vec3 color = diffuse * scene.sunColor * ao + Vec3(spec * fres) * scene.sunColor * ao;

    // Ambient sky term and the rim.
    const Vec3 amb = lerp(scene.skyHorizon * 0.32f, scene.skyZenith * 0.70f,
                          saturate(nrm.y * 0.5f + 0.5f));
    color += alb * amb * (0.52f * ao);
    const float rim = std::pow(1.0f - saturate(dot(nrm, V)), 3.5f) *
                      saturate(dot(nrm, rimDir));
    color += scene.skyZenith * (rim * 0.65f);
    return color;
}

// The eyeball. Everything is measured in the eye's own frame: the globe is an 11.8 mm
// sphere, the iris is 12 mm across and the opening is the almond of a real palpebral
// fissure (30 x 10 mm). Rendering the opening analytically - rather than hoping the socket
// geometry clips the sphere - is what keeps the eye from reading as a googly ball: the
// sclera outside the almond is shaded as lid skin, with the lash line and the lid crease
// drawn along the boundary.
inline Vec3 shadeEye(Vec3 nrm, Vec3 viewDir, const Scene& scene, Vec3 localPos) {
    const Vec3 V = normalize(viewDir);
    const float side = localPos.x >= 0.0f ? 1.0f : -1.0f;
    // Eyes sit at theta = +-0.40 and converge slightly on the viewer.
    const Vec3 outward = normalize(Vec3(side * 0.39f, 0.0f, 0.92f));
    const Vec3 gaze = normalize(outward * 0.45f + V * 0.55f);
    const Vec3 right = normalize(cross(Vec3(0.0f, 1.0f, 0.0f), gaze));
    const Vec3 up = cross(gaze, right);

    const float f = dot(nrm, gaze);   // 1 on the optical axis
    const float h = dot(nrm, right);
    const float v = dot(nrm, up);

    // Palpebral fissure: an almond in (h, v), slightly tilted up at the outer corner.
    const float tilt = 0.10f * h * side;
    const float ax = std::fabs(h) / 0.575f;
    const float ay = std::fabs(v - tilt) / 0.255f;
    const float r = std::pow(ax, 1.7f) + ay * ay;
    const float outside = smoothstepf(1.0f, 1.22f, r);   // 1 = lid, 0 = open eye

    // Sclera, faintly veined and cooler towards the corners.
    Vec3 sclera = Vec3(0.85f, 0.81f, 0.775f);
    sclera = lerp(sclera, Vec3(0.72f, 0.66f, 0.64f), saturate(std::fabs(h) * 1.4f - 0.35f));
    // Iris (6 mm radius -> cos(asin(6/11.8)) = 0.87), limbal ring, pupil.
    const float iris = smoothstepf(0.835f, 0.872f, f);
    Vec3 irisColor = Vec3(0.36f, 0.22f, 0.11f);
    irisColor *= 0.75f + 0.55f * saturate(-v * 1.5f + 0.35f);   // light through the stroma
    // Radial fibres: a little variation around the iris.
    const float ang = std::atan2(v, h);
    const float fibre = 0.85f + 0.15f * std::sin(ang * 26.0f);
    irisColor *= fibre;
    irisColor = lerp(irisColor, Vec3(0.15f, 0.09f, 0.05f), smoothstepf(0.90f, 0.95f, f));
    Vec3 c = lerp(sclera, irisColor, iris);
    c = lerp(c, Vec3(0.02f, 0.02f, 0.02f), smoothstepf(0.968f, 0.982f, f));
    const float limbal = std::exp(-std::pow((f - 0.872f) / 0.016f, 2.0f));
    c = lerp(c, Vec3(0.04f, 0.03f, 0.03f), limbal * 0.75f);

    // Lid skin outside the opening, with a lash line right on the boundary.
    const Vec3 lid = Vec3(0.66f, 0.45f, 0.39f);
    c = lerp(c, lid, outside);
    const float lash = std::exp(-std::pow((r - 1.03f) / 0.055f, 2.0f));
    const float lashUp = smoothstepf(-0.28f, 0.05f, v - tilt);   // thicker on the upper lid
    c = lerp(c, Vec3(0.045f, 0.035f, 0.030f), lash * (0.12f + 0.55f * lashUp));
    // Crease above the lid and the shadow the lid casts on the eyeball.
    const float crease = std::exp(-std::pow((v - tilt - 0.42f) / 0.10f, 2.0f));
    c = lerp(c, lid * 0.72f, saturate(crease) * 0.55f * smoothstepf(0.9f, 1.15f, r));
    c *= 1.0f - 0.34f * saturate(smoothstepf(-0.05f, 0.55f, v)) * (1.0f - outside);

    // Eyelashes. A lash line alone reads as eyeliner; real lashes are individual tapered
    // strands that leave the lid and curve up over the eyeball, and their silhouette on
    // the sclera is most of what makes an eye look alive at portrait distance.
    {
        const float vTop = tilt + 0.255f * std::sqrt(std::max(0.0f, 1.0f - std::pow(ax, 1.7f)));
        const float arc = saturate((h + 0.60f) / 1.20f);
        // Two beats: long lashes over the middle of the lid, shorter ones at the corners.
        // Individual lashes are not all the same length: the slow beat gives the fan its
        // shape and the fast one keeps neighbouring lashes from lining up.
        const float lenUp = (0.26f + 0.13f * std::sin(arc * 3.1f)) *
                            (1.0f - 0.55f * std::pow(std::fabs(h) / 0.62f, 3.0f)) *
                            (0.78f + 0.42f * (0.5f + 0.5f * std::sin(arc * 233.0f + 1.9f)));
        const float comb = 0.5f + 0.5f * std::sin(arc * 118.0f + 0.8f);
        const float above = v - vTop;
        if (above > 0.0f) {
            const float t = above / std::max(1e-3f, lenUp);
            if (t > -0.2f && t < 1.0f) {
                // Taper along the lash and separate the individual hairs in the comb.
                const float taper = smoothstepf(1.0f, 0.55f, saturate(t));
                const float hair = smoothstepf(0.42f, 0.86f, comb);
                c = lerp(c, Vec3(0.05f, 0.038f, 0.032f), taper * hair * 0.80f);
            }
        }
        // Lower lashes: sparse, half the length, angled down.
        const float lenLo = 0.13f * (1.0f - 0.5f * std::pow(std::fabs(h) / 0.62f, 2.0f));
        const float combLo = 0.5f + 0.5f * std::sin(arc * 74.0f + 2.1f);
        const float vBot = tilt - 0.255f * std::sqrt(std::max(0.0f, 1.0f - std::pow(ax, 1.7f)));
        const float below = vBot - v;
        if (below > -0.04f) {
            const float t = below / std::max(1e-3f, lenLo);
            if (t > -0.2f && t < 1.0f) {
                const float taper = smoothstepf(1.0f, 0.5f, saturate(t));
                c = lerp(c, Vec3(0.09f, 0.07f, 0.06f),
                         taper * smoothstepf(0.55f, 0.95f, combLo) * 0.62f);
            }
        }
    }

    // Catchlight: two lights, because a single dot looks dead.
    const Vec3 H = normalize(scene.sunDirection + V);
    c += scene.sunColor * (std::pow(saturate(dot(nrm, H)), 400.0f) * 2.6f);
    c += scene.skyZenith * (std::pow(saturate(dot(nrm, normalize(V + Vec3(-0.3f, 0.5f, 0.4f)))),
                                     260.0f) * 0.8f);
    return c * (0.72f + 0.28f * saturate(dot(nrm, scene.sunDirection)));
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
                        bool isSkin, bool isEye = false, const Mat4* xform = nullptr,
                        bool faceMasks = false) {
        const Mat4 invXform = xform ? xform->inverseAffine() : Mat4::identity();
        const Vec3 headCenter(0.0f, 0.02f, 0.0f);   // HeadModel::center()
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vec3 p[3], n[3];
            Projected pr[3];
            for (int k = 0; k < 3; ++k) {
                const uint32_t idx = mesh.indices[t + k];
                const Vec3 lp = mesh.positions[idx];
                const Vec3 ln = mesh.normals.empty() ? Vec3(0, 1, 0) : mesh.normals[idx];
                p[k] = xform ? xform->transformPoint(lp) : lp;
                n[k] = xform ? normalize(xform->transformDirection(ln)) : ln;
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
                    Vec3 nrm = normalize(n[0] * b0 + n[1] * b1 + n[2] * b2);
                    const Vec3 pos = p[0] * b0 + p[1] * b1 + p[2] * b2;
                    const Vec3 V = normalize(camPos - pos);
                    // Two-sided shading: procedurally generated tubes (the neck) and the
                    // teapot's spout/handle are not guaranteed to be wound outward, and an
                    // inward normal makes the surface read as an unlit gray silhouette.
                    if (dot(nrm, V) < 0.0f) nrm = -nrm;
                    Vec3 alb = baseAlbedo;
                    const float ndl = saturate(dot(nrm, scene.sunDirection));
                    float wrap = ndl;
                    if (isSkin) {
                        // The face masks are authored in the head's own space, so undo the
                        // instance transform before evaluating them. The neck/shoulders are
                        // shaded with the same skin model but without the face masks.
                        const Vec3 local = invXform.transformPoint(pos) - headCenter;
                        const FaceMasks fm = faceMasks ? headFaceMasks(local) : FaceMasks{};
                        s.color = shadeSkin(local, nrm, V, scene, baseAlbedo, specPower, specGain,
                                            faceMasks ? &fm : nullptr);
                        s.depth = z;
                        continue;
                    }
                    if (isEye) {
                        const Vec3 local = invXform.transformPoint(pos) - headCenter;
                        s.color = shadeEye(nrm, V, scene, local);
                        s.depth = z;
                        continue;
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
        // A crowd draws one head/body per hair instance, using that instance's transform
        // (every model is unique, exactly as in Figure 1). A single character uses the
        // mesh as-is.
        if (scene.hair.size() > 1 && !scene.head.positions.empty()) {
            for (const auto& inst : scene.hair) {
                drawMesh(scene.head, Vec3(0.655f, 0.45f, 0.37f), 60.0f, 0.55f, true, false,
                         &inst->transform, true);
                drawMesh(scene.body, Vec3(0.635f, 0.435f, 0.36f), 40.0f, 0.35f, true, false,
                         &inst->transform);
                drawMesh(scene.eyes, Vec3(0.16f, 0.11f, 0.07f), 120.0f, 0.9f, false, true,
                         &inst->transform);
            }
        } else {
            drawMesh(scene.head, Vec3(0.655f, 0.45f, 0.37f), 60.0f, 0.55f, true, false, nullptr,
                     true);
            drawMesh(scene.body, Vec3(0.635f, 0.435f, 0.36f), 40.0f, 0.35f, true);
            drawMesh(scene.eyes, Vec3(0.16f, 0.11f, 0.07f), 120.0f, 0.9f, false, /*isEye=*/true);
        }
    }
    if (scene.useTeapots) {
        // Figure 9: one teapot per hair instance, each under its own hair layer.
        if (scene.hair.size() > 1 && !scene.teapot.positions.empty()) {
            for (const auto& inst : scene.hair)
                drawMesh(scene.teapot, Vec3(0.86f, 0.84f, 0.80f), 90.0f, 0.45f, false, false,
                         &inst->transform);
        } else {
            drawMesh(scene.teapot, Vec3(0.86f, 0.84f, 0.80f), 90.0f, 0.45f, false);
        }
    }

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
    const float minWidthPx = 0.80f * ssScale;
    const float maxWidthPx = 1.45f * ssScale;

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

                // Per-strand colour. The dominant variation is *coherent per bundle*
                // (one clump of hair was dyed together), with only a few percent of
                // strand-to-strand jitter on top: strong per-strand jitter destroys the
                // coherent highlight and turns a hairstyle into straw. A small number of
                // strands are lifted well above the rest to read as flyaways.
                const float clumpJit = hash13(Vec3(static_cast<float>(f) * 0.517f, 11.3f, 3.7f));
                const float jit = hash13(Vec3(static_cast<float>(s) * 0.731f + f * 0.113f,
                                              static_cast<float>(f) * 0.517f, 3.7f));
                const float flyaway = jit > 0.972f ? 1.0f : 0.0f;
                StylingParams spj = inst.styling;
                const float value = 0.975f + 0.05f * jit + 0.055f * clumpJit + 0.20f * flyaway;
                spj.color = Vec3(inst.styling.color.x * value * (0.99f + 0.03f * jit),
                                 inst.styling.color.y * value,
                                 inst.styling.color.z * value * (1.01f - 0.02f * jit));

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
                        shadeHair(t, p, V, scene, spj, g.vertices[k].w, ao);
                }
                // ---- ribbon rasterisation -------------------------------------------
                // Each strand vertex strip is rasterised as a soft-edged capsule along
                // its screen-space spine instead of as two hard-edged triangles. The
                // coverage falls off across the ribbon width, so the silhouette of a
                // hairstyle is smooth before the strand density is high enough for
                // geometric coverage (which is what makes the difference between straw
                // and hair), and it tapers at the root and the tip where real fibres are
                // thinner. A capsule is also cheaper: one distance test instead of two
                // triangle setups and three barycentric coordinates per fragment.
                Projected prevC{};
                Vec3 prevColor(0, 0, 0);
                float prevHalfWidth = 0.0f;
                for (int k = 0; k < n; ++k) {
                    const Vec3 p = inst.transform.transformPoint(g.vertices[k].position);
                    const Vec3 t = normalize(transformDir(inst.transform, g.vertices[k].tangent));
                    const Vec3 toCam = normalize(camPos - p);
                    Vec3 side = cross(t, toCam);
                    if (length2(side) < 1e-14f) side = camRight;
                    side = normalize(side);
                    Vec3 offset = side * g.radius;

                    // Width clamp: never thinner than about a pixel (a sub-pixel ribbon
                    // would flicker away), never wider than a few (LOD strands stand in
                    // for many finer ones, so they are drawn with a proportional width).
                    Projected pc{}, po{};
                    project(p, &pc);
                    project(p + offset, &po);
                    float halfWidth = 0.9f * ssScale;
                    if (pc.visible && po.visible) {
                        const float pix = std::sqrt((po.x - pc.x) * (po.x - pc.x) +
                                                    (po.y - pc.y) * (po.y - pc.y));
                        const float clamped = clampf(pix, minWidthPx, maxWidthPx);
                        if (pix > 1e-6f) offset = offset * (clamped / pix);
                        halfWidth = clamped;
                    }

                    if (k > 0 && pc.visible && prevC.visible) {
                        const float ax = prevC.x, ay = prevC.y;
                        const float dx = pc.x - ax, dy = pc.y - ay;
                        const float len2 = dx * dx + dy * dy;
                        const float maxHw = std::max(prevHalfWidth, halfWidth) + ssScale;
                        const int x0 = std::max(0, static_cast<int>(std::floor(std::min(ax, pc.x) - maxHw)));
                        const int x1 = std::min(bw_ - 1, static_cast<int>(std::ceil(std::max(ax, pc.x) + maxHw)));
                        const int y0 = std::max(0, static_cast<int>(std::floor(std::min(ay, pc.y) - maxHw)));
                        const int y1 = std::min(bh_ - 1, static_cast<int>(std::ceil(std::max(ay, pc.y) + maxHw)));
                        if (len2 > 1e-8f && x1 >= x0 && y1 >= y0) {
                            const float invLen2 = 1.0f / len2;
                            const float st0 = static_cast<float>(k - 1) / static_cast<float>(n - 1);
                            const float st1 = static_cast<float>(k) / static_cast<float>(n - 1);
                            const Vec3 c1 = vcolor[static_cast<size_t>(k)];
                            for (int y = y0; y <= y1; ++y) {
                                for (int x = x0; x <= x1; ++x) {
                                    const float px = x + 0.5f, py = y + 0.5f;
                                    float tt = ((px - ax) * dx + (py - ay) * dy) * invLen2;
                                    tt = clampf(tt, 0.0f, 1.0f);
                                    const float qx = ax + dx * tt, qy = ay + dy * tt;
                                    const float dist = std::sqrt((px - qx) * (px - qx) +
                                                                 (py - qy) * (py - qy));
                                    const float hw = prevHalfWidth + (halfWidth - prevHalfWidth) * tt;
                                    float cov = 1.0f - smoothstepf(hw * 0.40f, hw, dist);
                                    if (cov <= 0.004f) continue;
                                    // Root and tip taper: fibres are thinner at both ends.
                                    const float st = st0 + (st1 - st0) * tt;
                                    cov *= smoothstepf(0.0f, 0.05f, st) *
                                           (1.0f - 0.85f * smoothstepf(0.70f, 1.0f, st));
                                    if (cov <= 0.004f) continue;
                                    const size_t pixIndex = static_cast<size_t>(y) * bw_ + x;
                                    const float z = prevC.z + (pc.z - prevC.z) * tt;
                                    if (z > opaqueDepth_[pixIndex]) continue;   // behind skin
                                    const Vec3 c = prevColor + (c1 - prevColor) * tt;
                                    const float a = options.hairAlpha * cov;
                                    Sample& sm = buffer_[pixIndex];
                                    sm.color = sm.color * (1.0f - a) + c * a;
                                }
                            }
                        }
                    }
                    prevC = pc;
                    prevColor = vcolor[static_cast<size_t>(k)];
                    prevHalfWidth = halfWidth;
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
