#include "silkcrowd/renderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "silkcrowd/assets.h"
#include "silkcrowd/root_placement.h"

namespace silk {
namespace {

// One progressive set per domain, shared by every strand and every frame. The sets are baked
// into `assets/noise/` so a cold run does not pay for the elimination (see root_placement.h).
const SampleSet2D& sharedSampleSet(bool triangleDomain) {
    static const SampleSet2D& quad = presetSet(NoiseSet::ReferenceQuad, assetPath("noise"));
    static const SampleSet2D& tri = presetSet(NoiseSet::ReferenceTriangle, assetPath("noise"));
    return triangleDomain ? tri : quad;
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
                      const StylingParams& sp, float w, float ao, float sunVisibility = 1.0f) {
    const Vec3 T = normalize(tangent);
    const Vec3 L = scene.sunDirection;
    const Vec3 V = normalize(viewDir);
    const Vec3 H = normalize(L + V);
    (void)position;

    // Roots read darker than tips (light penetrates the fibre at the ends, and dye is
    // always densest at the root). This root-to-tip ramp is most of what makes dyed hair
    // read as hair rather than as coloured plastic.
    const Vec3 rootTint = sp.color * Vec3(0.56f, 0.54f, 0.54f);
    const Vec3 tipTint = sp.color * Vec3(1.16f, 1.14f, 1.10f);
    const Vec3 albedo = lerp(rootTint, tipTint, saturate(w * 1.15f + 0.15f));

    const float sinTL = std::sqrt(saturate(1.0f - dot(T, L) * dot(T, L)));
    Vec3 color = albedo * scene.sunColor *
                 (sinTL * scene.sunIntensity * 0.22f * ao * sunVisibility);

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
    const float spec = lobe(0.12f, 78.0f, 0.62f) + lobe(-0.16f, 22.0f, 0.20f) +
                       lobe(0.04f, 6.0f, 0.035f);
    const Vec3 specColor = lerp(Vec3(1.0f, 0.96f, 0.90f), sp.color * 3.0f + Vec3(0.1f), 0.45f);
    color += specColor * scene.sunColor * (spec * scene.sunIntensity * 0.30f * ao * sunVisibility);

    // Sky ambient (approximate IBL).
    Vec3 nApprox = normalize(cross(T, normalize(cross(T, V) + Vec3(1e-5f, 0, 0))));
    if (dot(nApprox, V) < 0.0f) nApprox = -nApprox;
    const float up = saturate(nApprox.y * 0.5f + 0.5f);
    // Ambient is what fills the *inside* of a hairstyle, and it is dim: most of a head of
    // hair is in its own shadow, which is why hair reads dark with brilliant streaks and
    // not as a uniformly lit pale mass (the failure mode of a flat ambient term -- it looks
    // like felt). The occlusion weight matters more than the sky does.
    const Vec3 ambient = lerp(scene.skyHorizon * 0.30f, scene.skyZenith * 0.85f, up);
    color += albedo * ambient * (0.14f + 0.42f * ao);
    return color;
}

// =============================================================================
//  Hair self-shadowing
//
//  Hair is not an opaque surface: light dies a few millimetres into it, which is why a
//  real hairstyle has a bright outer shell, dark clump interiors and a hard shadow thrown
//  onto the forehead, the cheek and the neck. The renderer reproduces that with one
//  light-space depth buffer of the *hair mesh* (the styled volume's envelope, not the
//  strands), sampled with a penetration distance: a sample is lit if it lies within
//  `penetration` of the envelope along the light, and everything deeper is occluded.
//  That single rule gives all three effects -- shell lit, interior dark, face shadowed --
//  for a few thousand rasterised quads per model.
// =============================================================================
struct HairShadow {
    int size = 0;
    Vec3 center{0, 0, 0};
    Vec3 axisU{1, 0, 0}, axisV{0, 1, 0}, axisW{0, 0, 1};   // axisW points towards the light
    float extent = 0.1f;                                   // half-size of the light-space box
    float depthLo = 0.0f, depthHi = 1.0f;
    std::vector<float> depth;                              // depth of the nearest caster
    bool valid = false;

    void build(const Scene& scene, int resolution, bool includeHead);
    void addHairMesh(const HairMesh& mesh, const Mat4* xform, uint32_t firstLayer);
    void addMesh(const Mesh& mesh, const Mat4* xform);
    void addTriangle(Vec3 a, Vec3 b, Vec3 c);
    // 1 where the point sees the sun, 0 deep inside the hair.
    float visibility(Vec3 p, float penetration) const;
};

void HairShadow::build(const Scene& scene, int resolution, bool includeHead) {
    valid = false;
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    auto grow = [&](const Vec3& p) {
        lo = Vec3(std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z));
        hi = Vec3(std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z));
    };
    bool any = false;
    for (const auto& inst : scene.hair) {
        Vec3 mlo, mhi;
        inst->mesh.bounds(&mlo, &mhi);
        for (int c = 0; c < 8; ++c) {
            const Vec3 corner((c & 1) ? mhi.x : mlo.x, (c & 2) ? mhi.y : mlo.y,
                              (c & 4) ? mhi.z : mlo.z);
            grow(inst->transform.transformPoint(corner));
        }
        any = true;
    }
    if (!any) return;
    // The head is included in the caster set so that the hair also shadows the collar and
    // the shoulders below it; the hair envelope itself is what shadows the face.
    if (includeHead && !scene.head.positions.empty())
        for (const Vec3& p : scene.head.positions) grow(p);
    for (const auto& inst : scene.hair)
        if (!scene.head.positions.empty())
            for (const Vec3& p : scene.head.positions) grow(inst->transform.transformPoint(p));

    center = (lo + hi) * 0.5f;
    axisW = normalize(scene.sunDirection);
    const Vec3 reference = std::fabs(axisW.y) > 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    axisU = normalize(cross(reference, axisW));
    axisV = cross(axisW, axisU);
    extent = 0.0f;
    for (const Vec3& corner : {Vec3(lo.x, lo.y, lo.z), Vec3(hi.x, hi.y, hi.z),
                               Vec3(lo.x, hi.y, lo.z), Vec3(hi.x, lo.y, hi.z),
                               Vec3(lo.x, lo.y, hi.z), Vec3(hi.x, hi.y, lo.z)}) {
        const Vec3 d = corner - center;
        extent = std::max({extent, std::fabs(dot(d, axisU)), std::fabs(dot(d, axisV))});
    }
    extent *= 1.06f;
    if (extent <= 0.0f) return;
    size = resolution;
    depth.assign(static_cast<size_t>(size) * size, -1e30f);
    depthLo = 1e30f;
    depthHi = -1e30f;
    for (const Vec3& corner : {Vec3(lo.x, lo.y, lo.z), Vec3(hi.x, hi.y, hi.z)}) {
        const float d = dot(corner - center, axisW);
        depthLo = std::min(depthLo, d);
        depthHi = std::max(depthHi, d);
    }
    valid = true;
}

void HairShadow::addTriangle(Vec3 a, Vec3 b, Vec3 c) {
    const float scale = static_cast<float>(size) / (2.0f * extent);
    float x[3], y[3], d[3];
    const Vec3 p[3] = {a, b, c};
    for (int k = 0; k < 3; ++k) {
        const Vec3 q = p[k] - center;
        x[k] = (dot(q, axisU) + extent) * scale;
        y[k] = (dot(q, axisV) + extent) * scale;
        d[k] = dot(q, axisW);
    }
    const float area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if (std::fabs(area) < 1e-7f) return;
    const float inv = 1.0f / area;
    const int x0 = std::max(0, static_cast<int>(std::floor(std::min({x[0], x[1], x[2]}))));
    const int x1 = std::min(size - 1, static_cast<int>(std::ceil(std::max({x[0], x[1], x[2]}))));
    const int y0 = std::max(0, static_cast<int>(std::floor(std::min({y[0], y[1], y[2]}))));
    const int y1 = std::min(size - 1, static_cast<int>(std::ceil(std::max({y[0], y[1], y[2]}))));
    for (int py = y0; py <= y1; ++py) {
        for (int ix = x0; ix <= x1; ++ix) {
            const float sx = static_cast<float>(ix) + 0.5f;
            const float sy = static_cast<float>(py) + 0.5f;
            const float b0 = ((x[1] - sx) * (y[2] - sy) - (x[2] - sx) * (y[1] - sy)) * inv;
            const float b1 = ((x[2] - sx) * (y[0] - sy) - (x[0] - sx) * (y[2] - sy)) * inv;
            if (b0 < 0.0f || b1 < 0.0f || b0 + b1 > 1.0f) continue;
            float& slot = depth[static_cast<size_t>(py) * size + ix];
            slot = std::max(slot, b0 * d[0] + b1 * d[1] + (1.0f - b0 - b1) * d[2]);
        }
    }
}

void HairShadow::addMesh(const Mesh& mesh, const Mat4* xform) {
    if (!valid || mesh.indices.empty()) return;
    auto world = [&](uint32_t index) {
        const Vec3 p = mesh.positions[index];
        return xform ? xform->transformPoint(p) : p;
    };
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3)
        addTriangle(world(mesh.indices[t]), world(mesh.indices[t + 1]),
                    world(mesh.indices[t + 2]));
}

void HairShadow::addHairMesh(const HairMesh& mesh, const Mat4* xform, uint32_t firstLayer) {
    if (!valid) return;
    auto world = [&](uint32_t layer, uint32_t v) {
        const Vec3 p = mesh.layerVertex(layer, v);
        return xform ? xform->transformPoint(p) : p;
    };
    const uint32_t layers = mesh.numLayers();
    for (uint32_t l = firstLayer + 1; l < layers; ++l) {
        for (uint32_t f = 0; f < mesh.numFaces(); ++f) {
            const HairMeshFace& face = mesh.face(f);
            const uint32_t n = face.count;
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t a = face.v[k];
                const uint32_t b = face.v[(k + 1) % n];
                addTriangle(world(l - 1, a), world(l - 1, b), world(l, b));
                addTriangle(world(l - 1, a), world(l, b), world(l, a));
            }
        }
    }
}

float HairShadow::visibility(Vec3 p, float penetration) const {
    if (!valid) return 1.0f;
    // 3x3 taps, so the shadow edge is soft at the resolution of a strand clump rather than
    // stair-stepped at the resolution of the buffer.
    const Vec3 q = p - center;
    const float scale = static_cast<float>(size) / (2.0f * extent);
    const float fx = (dot(q, axisU) + extent) * scale - 0.5f;
    const float fy = (dot(q, axisV) + extent) * scale - 0.5f;
    const int cx = static_cast<int>(std::floor(fx));
    const int cy = static_cast<int>(std::floor(fy));
    const int reach = penetration > 0.02f ? 2 : 1;
    float sum = 0.0f;
    float darkest = 1.0f;
    int taps = 0;
    for (int dy = -reach; dy <= reach; ++dy) {
        for (int dx = -reach; dx <= reach; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= size || y >= size) {
                sum += 1.0f;
                ++taps;
                continue;
            }
            const float caster = depth[static_cast<size_t>(y) * size + x];
            if (caster < -1e20f) {
                sum += 1.0f;
                ++taps;
                continue;
            }
            // How far behind the envelope this sample is, measured along the light.
            const float delta = caster - dot(q, axisW);
            const float vis = 1.0f - smoothstepf(0.0f, penetration, delta);
            sum += vis;
            darkest = std::min(darkest, vis);
            ++taps;
        }
    }
    if (taps == 0) return 1.0f;
    const float mean = sum / static_cast<float>(taps);
    // The buffer stores the *outermost* caster per cell, so a cell with no hair in it reads
    // as fully lit. Between two strands that is true, but at skin depth it paints bright
    // stripes across a shadowed cheek -- which is exactly what it did before this blend.
    // Weighting towards the darkest tap closes those leaks without hardening the penumbra.
    return darkest + (mean - darkest) * 0.35f;
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
                      const FaceMasks* masks, float sunVisibility = 1.0f) {
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
    diffuse *= scene.sunIntensity * 0.55f * sunVisibility;

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
                       specGain * gloss * sunVisibility;

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

// Bilinear texture fetch with wrap-around. The scan's UV layout is a single chart, so a
// clamp would smear the edge texels of the neck across the whole bust; wrapping cannot be
// seen, because the chart never samples across its own border.
inline Vec3 texelFetch(const Image& image, Vec2 uv) {
    if (image.width <= 0 || image.height <= 0 || image.pixels.empty()) return Vec3(0.5f);
    float u = uv.x - std::floor(uv.x);
    float v = uv.y - std::floor(uv.y);
    const float fx = u * static_cast<float>(image.width) - 0.5f;
    const float fy = v * static_cast<float>(image.height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    const float tx = fx - std::floor(fx);
    const float ty = fy - std::floor(fy);
    const int x1 = (x0 + 1) % image.width;
    const int y1 = (y0 + 1) % image.height;
    const int xa = ((x0 % image.width) + image.width) % image.width;
    const int ya = ((y0 % image.height) + image.height) % image.height;
    const Vec3 a = lerp(image.at(xa, ya), image.at(x1, ya), tx);
    const Vec3 b = lerp(image.at(xa, y1), image.at(x1, y1), tx);
    return lerp(a, b, ty);
}

// Scanned skin: the albedo of every pixel comes from the scan's own 1K texture (pores,
// lips, lashes, brows, the fine red of the ears) instead of from painted masks, so the
// only things left for the shader are the studio rig, the specular and the scalp.
// Scanned skin.
//
// The scan brings its own photographic albedo -- pores, lips, brows, the red of the
// capillaries -- so the shader must not paint a face on top of it. What it adds is what a
// photograph cannot carry: the light. Skin is a layered dielectric, and three things in it
// are visible at portrait distance and are what separate a lit face from a grey one:
//
//   1. subcutaneous scattering. Light that enters the skin is scattered a few millimetres
//      and leaves warm, so the terminator of the face is red rather than grey, the nostrils
//      and the ear rims glow when backlit, and the shadow side never goes neutral;
//   2. a *roughness* split in the specular. Skin is not one gloss: the T-zone (forehead,
//      nose bridge, chin) is oilier and carries a tight highlight, the cheeks and the jaw
//      are drier and only take a broad sheen. The two lobes are weighted by a T-zone mask
//      evaluated in head space;
//   3. pores. At portrait distance a perfectly smooth highlight reads as plastic, so the
//      specular normal is jittered at texel scale with a hash of the UV -- a perturbation,
//      not a texture, so it survives the atlas being re-authored and costs two multiplies.
//
// Everything is energy-scaled against the scene's sun and sky, the same lights the hair uses.
inline float hash01(Vec2 p) {
    const float h = std::sin(p.x * 127.1f + p.y * 311.7f) * 43758.5453f;
    return h - std::floor(h);
}

inline Vec3 shadeScannedSkin(Vec3 nrm, Vec3 viewDir, const Scene& scene, Vec3 albedo,
                             float specGain, float scalp, float sunVisibility, Vec2 uv,
                             Vec3 local) {
    const Vec3 V = normalize(viewDir);
    const Vec3 L = scene.sunDirection;
    // The scan's albedo is photographic: it already carries the mottling, the subsurface
    // red and the pores the procedural path has to fake. It is only lifted a little to
    // compensate for the flat daylight it was captured under.
    albedo = albedo * 1.35f;
    albedo = lerp(albedo, Vec3(0.020f, 0.013f, 0.010f), saturate(scalp));
    const float skin = 1.0f - saturate(scalp);

    const Vec3 keyDir = normalize(L + Vec3(0.0f, 0.15f, 0.0f));
    const Vec3 fillDir = normalize(Vec3(-keyDir.x * 0.9f, 0.25f, -keyDir.z * 0.35f + 0.55f));
    const Vec3 rimDir = normalize(Vec3(-keyDir.x * 0.6f, 0.45f, -keyDir.z));

    // T-zone: forehead, nose bridge and chin carry more oil than the cheeks. `local` is in
    // head space (crown at y = +0.128, face toward +z, midline at x = 0).
    const float forehead = smoothstepf(0.020f, 0.060f, local.y) *
                           smoothstepf(-0.010f, 0.030f, local.z);
    const float noseBridge = smoothstepf(0.016f, 0.006f, std::fabs(local.x)) *
                             smoothstepf(0.055f, 0.075f, local.z) *
                             smoothstepf(-0.030f, -0.010f, local.y);
    const float chin = smoothstepf(-0.045f, -0.065f, local.y) *
                       smoothstepf(0.040f, 0.060f, local.z);
    const float oily = saturate(forehead * 0.85f + noseBridge + chin * 0.7f) * skin;

    // --- pores -------------------------------------------------------------------
    // A hash of the atlas UV breaks the highlight up at pore scale. It scales the specular
    // rather than tilting the normal: tilting was the first attempt and it banded -- the
    // sine hash correlates along diagonals in UV, which on a cheek reads as a smear.
    const float pore = hash01(uv * 812.0f);
    const float poreWide = hash01(uv * 214.0f + Vec2(11.7f, 29.3f));
    const float breakup = 0.80f + 0.26f * pore + 0.14f * poreWide;

    // --- diffuse, with a warm terminator ------------------------------------------
    const float ndl = dot(nrm, keyDir);
    const float wrapDiff = saturate((ndl + 0.42f) / 1.42f);
    Vec3 color = albedo * scene.sunColor * (wrapDiff * scene.sunIntensity * 0.55f) *
                 sunVisibility;
    // Subcutaneous scattering: strongest where the surface turns away from the key, and
    // strongest on thin parts (ears, nostrils) through the backlight term.
    const float terminator = std::exp(-ndl * ndl * 6.0f);
    const float back = saturate(-ndl) * 0.5f + terminator * 0.5f;
    color += albedo * Vec3(1.00f, 0.30f, 0.21f) * (back * 0.42f) * scene.sunColor *
             (scene.sunIntensity * 0.18f) * skin * sunVisibility;
    // Bounce off the shoulders and hair onto the jaw: warm, from below.
    const float bounce = saturate(-nrm.y) * saturate(nrm.z * 0.5f + 0.5f);
    color += albedo * Vec3(0.55f, 0.30f, 0.26f) * (bounce * 0.10f) * skin;

    const float fill = saturate(dot(nrm, fillDir));
    color += albedo * scene.skyHorizon * (fill * 0.30f + 0.05f);
    const Vec3 amb = lerp(scene.skyHorizon * 0.32f, scene.skyZenith * 0.70f,
                          saturate(nrm.y * 0.5f + 0.5f));
    // Occlusion dims the sky light as well as the sun, or deep creases read as painted-on.
    color += albedo * amb * (0.34f + 0.21f * sunVisibility);

    // --- specular: two skin lobes, Fresnel, pores ----------------------------------
    const Vec3 H = normalize(keyDir + V);
    const float nh = saturate(dot(nrm, H));
    const float ndv = saturate(dot(nrm, V));
    const float f0 = 0.028f;
    const float fres = f0 + (1.0f - f0) * std::pow(1.0f - ndv, 5.0f);
    // Dry skin: a broad sheen. Oily skin: a tight, bright highlight on top of it.
    const float broad = std::pow(nh, 14.0f + 26.0f * (1.0f - oily));
    const float tight = std::pow(nh, 120.0f + 460.0f * oily);
    const float oilGain = 0.30f + 0.85f * oily;
    color += scene.sunColor * (specGain * fres * sunVisibility * breakup) *
             (broad * 0.10f + tight * 0.50f * oilGain);

    // --- fuzz ---------------------------------------------------------------------
    // Peach fuzz catches the key at grazing angles and gives the silhouette its softness.
    const float fuzz = std::pow(1.0f - ndv, 3.5f) * saturate(ndl * 0.6f + 0.35f);
    color += scene.skyZenith * (fuzz * 0.30f * skin);
    const float rim = std::pow(1.0f - ndv, 3.5f) * saturate(dot(nrm, rimDir));
    color += scene.skyZenith * (rim * 0.40f);
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

    // ---------------- hair self-shadowing ----------------
    // One light-space depth buffer of the styled volumes (see HairShadow). The hero is
    // rendered at 1024^2 over a 0.4 m box -- roughly 0.4 mm per texel, finer than a strand
    // clump -- while a crowd shares 512^2 over the whole grid, which is still finer than the
    // characters are on screen.
    HairShadow shadow;
    if (options.drawShadows && options.drawHair && !scene.hair.empty()) {
        shadow.build(scene, scene.hair.size() > 8 ? 512 : 1024, options.drawHead);
        for (const auto& inst : scene.hair) {
            const uint32_t layers = inst->mesh.numLayers();
            shadow.addHairMesh(inst->mesh, &inst->transform, layers > 2 ? layers - 2 : 0);
            if (options.drawHead && scene.hair.size() > 1 && !scene.head.positions.empty())
                shadow.addMesh(scene.head, &inst->transform);
        }
        if (options.drawHead && scene.hair.size() <= 1 && !scene.head.positions.empty())
            shadow.addMesh(scene.head, nullptr);
    }

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
    const bool texturedSkin = scene.useSkinTexture && !scene.skinAlbedo.pixels.empty();
    auto drawMesh = [&](const Mesh& mesh, Vec3 baseAlbedo, float specPower, float specGain,
                        bool isSkin, bool isEye = false, const Mat4* xform = nullptr,
                        bool faceMasks = false) {
        const Mat4 invXform = xform ? xform->inverseAffine() : Mat4::identity();
        const Vec3 headCenter(0.0f, 0.02f, 0.0f);   // HeadModel::center()
        const bool hasUv = mesh.uvs.size() == mesh.positions.size();
        const bool hasScalp = mesh.scalpMask.size() == mesh.positions.size();
        const bool useTexture = texturedSkin && hasUv && isSkin;
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vec3 p[3], n[3];
            Vec2 uv[3];
            float scalp[3] = {0.0f, 0.0f, 0.0f};
            Projected pr[3];
            for (int k = 0; k < 3; ++k) {
                const uint32_t idx = mesh.indices[t + k];
                const Vec3 lp = mesh.positions[idx];
                const Vec3 ln = mesh.normals.empty() ? Vec3(0, 1, 0) : mesh.normals[idx];
                p[k] = xform ? xform->transformPoint(lp) : lp;
                n[k] = xform ? normalize(xform->transformDirection(ln)) : ln;
                if (hasUv) uv[k] = mesh.uvs[idx];
                if (hasScalp) scalp[k] = mesh.scalpMask[idx];
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
                    if (useTexture) {
                        // A scanned head: the face is the scan's own photograph, so the
                        // shader only has to light it. The scalp under the hair still goes
                        // dark, or a bald scan would show a bald head at the parting.
                        const Vec2 uvHere = uv[0] * b0 + uv[1] * b1 + uv[2] * b2;
                        const float scalpHere = scalp[0] * b0 + scalp[1] * b1 + scalp[2] * b2;
                        // Hair shadows the face: the same light-space buffer, with a
                        // shorter penetration (skin is opaque, so only the depth gap to
                        // the hair above it matters).
                        const float lit = shadow.visibility(pos, 0.004f);
                        const Vec3 localHere = invXform.transformPoint(pos) - headCenter;
                        s.color = shadeScannedSkin(nrm, V, scene, texelFetch(scene.skinAlbedo, uvHere),
                                                   specGain, scalpHere, lit, uvHere, localHere);
                        s.depth = z;
                        continue;
                    }
                    if (isSkin) {
                        // The face masks are authored in the head's own space, so undo the
                        // instance transform before evaluating them. The neck/shoulders are
                        // shaded with the same skin model but without the face masks.
                        const Vec3 local = invXform.transformPoint(pos) - headCenter;
                        const FaceMasks fm = faceMasks ? headFaceMasks(local) : FaceMasks{};
                        s.color = shadeSkin(local, nrm, V, scene, baseAlbedo, specPower, specGain,
                                            faceMasks ? &fm : nullptr,
                                            shadow.visibility(pos, 0.006f));
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
    const float minWidthPx = 1.05f * ssScale;   // a sub-pixel ribbon vanishes at a grazing silhouette
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
                // Positional, at lock scale: hashing the *face index* (what this did) makes
                // every strand of a face the same colour, and a crown face covers a huge
                // part of the screen, so the hairstyle showed rectangular blocks of tone.
                const Vec3 rootWorld = inst.transform.transformPoint(g.vertices[0].position);
                const float clumpJit = hash13(Vec3(std::floor(rootWorld.x / 0.018f) * 0.517f,
                                                   std::floor(rootWorld.y / 0.018f) * 0.913f,
                                                   std::floor(rootWorld.z / 0.018f) * 1.391f));
                const float jit = hash13(Vec3(static_cast<float>(s) * 0.731f + f * 0.113f,
                                              static_cast<float>(f) * 0.517f, 3.7f));
                const float flyaway = jit > 0.972f ? 1.0f : 0.0f;
                StylingParams spj = inst.styling;
                // Coherent per-clump variation is what makes a hairstyle read as many locks
                // rather than one sheet of paint; per-strand jitter on top is deliberately
                // small (strong per-strand jitter is straw).
                // A head of hair is thousands of fibres of visibly different brightness: the
                // gaps between them are what the eye reads as "hair" at any distance. Averaging
                // that variation away (what 4.5 % jitter did) leaves a smooth, chalky sheet --
                // felt, not hair.
                // Coherent at lock scale, not per-fibre: a per-fibre jitter of this size is
                // salt-and-pepper noise at one pixel per strand, and it turns the crown --
                // where the root patch is densest -- into fur. The eye reads *locks*.
                // ...but only below the crown. On top of the head the fibres lie flat,
                // parallel and *close together* -- the scalp is where a hairstyle is
                // combed flat, not where it breaks up -- so at the crown the variation is
                // damped. Keeping the full lock contrast up there, where the ribbons are
                // seen end-on and overlap, turns the top into brush fur.
                const float crownFlat = 1.0f - smoothstepf(0.085f, 0.125f, rootWorld.y);
                const float value = 1.0f + (0.30f * jit - 0.15f) * (1.0f - 0.75f * crownFlat) +
                                    0.40f * (clumpJit - 0.5f) * (1.0f - 0.90f * crownFlat) +
                                    0.25f * flyaway;
                spj.color = Vec3(inst.styling.color.x * value * (0.99f + 0.03f * jit),
                                 inst.styling.color.y * value,
                                 inst.styling.color.z * value * (1.01f - 0.02f * jit));

                // Shading is evaluated once per strand vertex (this is exactly what the
                // mesh shader does: it shades the 2*ns+2 strip vertices it emits) and
                // interpolated across the ribbon, instead of per fragment.
                //
                // Per-lock tangent tilt. At this resolution a fibre is a fraction of a pixel,
                // so the reader sees the *locks*, not the fibres -- and a lock that shares
                // the exact tangent of its neighbour produces one continuous highlight, i.e.
                // a painted sheet. Tilting the shading tangent a few degrees per lock moves
                // each highlight off its neighbour's and breaks the surface into streaks,
                // which is what a real head of hair looks like at any distance. (Kajiya-Kay
                // from Section 3.4 is kept: this only perturbs the tangent handed to it.)
                const float lockTilt = (0.30f * (clumpJit - 0.5f) + 0.16f * (jit - 0.5f)) *
                                       (1.0f - 0.80f * crownFlat);
                std::vector<Vec3> vcolor(static_cast<size_t>(n));
                for (int k = 0; k < n; ++k) {
                    const Vec3 p = inst.transform.transformPoint(g.vertices[k].position);
                    Vec3 t = normalize(transformDir(inst.transform, g.vertices[k].tangent));
                    const Vec3 V = normalize(camPos - p);
                    {
                        Vec3 sideAxis = cross(t, V);
                        if (length2(sideAxis) > 1e-12f) {
                            t = normalize(t + normalize(sideAxis) * lockTilt);
                        }
                    }
                    // Root-to-tip ramp (light penetrates a fibre at its tip) times the
                    // volumetric self-shadow: a sample 12 mm inside the hair envelope is
                    // in the dark of the hairstyle, the outer shell sees the sun.
                    const float ramp = 0.55f + 0.45f * saturate(g.vertices[k].w * 1.6f);
                    // Deep shadowing, but not a black hole: a sample deep inside the
                    // envelope is in the dark of the hairstyle, yet light still diffuses
                    // through hair. With a hard 0 the fringe -- a thin sheet *inside* the
                    // hair volume, in front of the forehead -- came out as a black band
                    // across the face, because from the sun's side the layer above it
                    // always wins the depth test. 30 % of the sun survives everywhere.
                    const float vis = shadow.visibility(p, 0.012f);
                    // The crown is the outermost shell of the whole hairstyle: nothing is
                    // above it, so it is the one place where the envelope depth buffer
                    // under-reports the light. Without this term the top of every head sits
                    // in a dark band between the bright side and the bright fringe, which
                    // reads as a cap seam.
                    // (head space: the scans are normalised so the crown sits at y ~ 0.128)
                    const float crownLit = smoothstepf(0.055f, 0.100f, p.y);
                    const float lit = 0.30f + 0.70f * std::max(vis, 0.55f * crownLit);
                    // Ambient (sky) light diffuses much further into hair than the sun
                    // does, so it keeps a floor as well: without it the front sheet of a
                    // fringe, which the crown shadows from above, went almost black and
                    // read as a hat brim.
                    const float sky = 0.45f + 0.55f * shadow.visibility(p, 0.055f);
                    // Lock-scale occlusion: the seams between locks are dark and the crowns
                    // of the locks catch the light, so the surface breaks into vertical
                    // streaks of tone at 1-2 cm scale instead of one flat sheet.
                    const float lockAO = 1.0f + (0.76f * clumpJit - 0.38f) *
                                                   (1.0f - 0.90f * crownFlat);
                    // (the AO modulation is applied as given; the crown is flattened above
                    // by damping the tangent tilt and the per-lock value instead)
                    const float ao = ramp * (0.35f + 0.65f * sky) * lockAO;
                    vcolor[static_cast<size_t>(k)] =
                        shadeHair(t, p, V, scene, spj, g.vertices[k].w, ao, lit);
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

                    // Foreshortening. A fibre seen end-on covers no area, but the capsule
                    // rasteriser below enforces a one-pixel minimum width, so every strand
                    // that points at the camera -- which, at the crown, is most of them --
                    // stamps a full-brightness pixel. That is what makes the top of the head
                    // sparkle like a scrubbing brush. Scaling the coverage by the projected
                    // width of the fibre (|sin| of the angle between tangent and view) is
                    // both the physical answer and the fix.
                    const float along = dot(t, toCam);
                    const float foreshorten = clampf(std::sqrt(std::max(0.0f, 1.0f - along * along)),
                                                     0.18f, 1.0f);

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
                                    // Root and tip taper. A blunt bob is cut, not faded, so
                                    // the tip keeps more of its width than a wispy end
                                    // would (this was 0.85, which turned every hem into a
                                    // soft fade and made the fringe read as a brim).
                                    cov *= smoothstepf(0.0f, 0.05f, st) *
                                           (1.0f - 0.80f * smoothstepf(0.82f, 1.0f, st));
                                    cov *= foreshorten;
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
