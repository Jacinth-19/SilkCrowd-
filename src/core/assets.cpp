#include "silkcrowd/assets.h"

#include <algorithm>
#include <cmath>

namespace silk {
namespace {

inline Vec3 gaussBump(Vec3 dir, Vec3 axis, float sharpness) {
    return dir * 1.0f;  // placeholder (unused)
}

// Rotates a direction towards -Y (world down) by `amount`.
inline Vec3 bendDown(Vec3 d, float amount) {
    const Vec3 down(0, -1, 0);
    return normalize(d * (1.0f - amount) + down * amount);
}

}  // namespace

void Mesh::computeNormals() {
    normals.assign(positions.size(), Vec3(0, 0, 0));
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vec3& a = positions[indices[i]];
        const Vec3& b = positions[indices[i + 1]];
        const Vec3& c = positions[indices[i + 2]];
        const Vec3 n = cross(b - a, c - a);
        for (int k = 0; k < 3; ++k) normals[indices[i + k]] += n;
    }
    for (Vec3& n : normals) n = length2(n) > 1e-20f ? normalize(n) : Vec3(0, 1, 0);
}

// =============================================================================
//  Head
// =============================================================================
HeadModel::HeadModel() {
    buildHead();
    buildHeadLow();
    buildBody();
    buildEyes();
}

void HeadModel::buildEyes() {
    // Two spheres sitting in the sockets, slightly in front of the skull surface.
    eyes_.positions.clear();
    eyes_.normals.clear();
    eyes_.uvs.clear();
    eyes_.indices.clear();
    for (int side = -1; side <= 1; side += 2) {
        const float theta = 0.40f * static_cast<float>(side);
        const float phi = 1.55f;
        // Seated in the eye socket: the socket is 7.5 mm deep, so an eyeball of radius
        // 11.5 mm placed 6 mm behind the (deformed) surface leaves the cornea proud of the
        // socket without looking like a bug eye.
        const Vec3 c = surfacePoint(theta, phi) - surfaceNormal(theta, phi) * 0.0042f;
        const float r = 0.0118f;
        const int nu = 20, nv = 14;
        const uint32_t base = static_cast<uint32_t>(eyes_.positions.size());
        for (int j = 0; j <= nv; ++j) {
            const float v = static_cast<float>(j) / nv;
            for (int i = 0; i <= nu; ++i) {
                const float u = static_cast<float>(i) / nu;
                const float ph = kPi * v, th = 2.0f * kPi * u;
                const Vec3 n(std::sin(ph) * std::cos(th), std::cos(ph), std::sin(ph) * std::sin(th));
                eyes_.positions.push_back(c + n * r);
                eyes_.normals.push_back(n);
                eyes_.uvs.push_back(Vec2(u, v));
            }
        }
        for (int j = 0; j < nv; ++j)
            for (int i = 0; i < nu; ++i) {
                const uint32_t a = base + static_cast<uint32_t>(j * (nu + 1) + i);
                const uint32_t b = a + 1, cc = a + nu + 1, d = cc + 1;
                eyes_.indices.insert(eyes_.indices.end(), {a, cc, b, b, cc, d});
            }
    }
    eyes_.name = "eyes";
    eyes_.materialIds.assign(eyes_.triangleCount(), 1);
}

namespace {

inline float sq(float x) { return x * x; }

// A raised (positive) or recessed (negative) bump on the surface: a Gaussian of the
// angular distance to a landmark, with independent azimuth/extent in theta and phi.
inline float bump(float theta, float phi, float th0, float ph0, float thSigma, float phSigma) {
    const float dt = std::atan2(std::sin(theta - th0), std::cos(theta - th0));
    return std::exp(-0.5f * (sq(dt / thSigma) + sq((phi - ph0) / phSigma)));
}

// A crease: the negative of a bump, but narrow across and long along one axis.
inline float crease(float theta, float phi, float th0, float ph0, float thSigma, float phSigma) {
    return -bump(theta, phi, th0, ph0, thSigma, phSigma);
}

}  // namespace

// The female head of the editor: an anatomical parametric surface rather than a displaced
// sphere. theta = azimuth (0 = face/+Z, +pi/2 = right ear, pi = nape) and phi = polar
// distance from the crown, so the same parameterization drives the scalp patch, the hair
// mesh root layer and the hairline (Section 3.1). Every feature below is placed at a real
// landmark of a ~22 cm female head (glabella, nasion, subnasale, stomion, menton, ...),
// and the whole surface stays inside the silhouette the hair meshes were authored against.
Vec3 HeadModel::surfacePoint(float theta, float phi) const {
    const float sp = std::sin(phi), cp = std::cos(phi);
    const float dx = sp * std::sin(theta);   // +x = her right
    const float dy = cp;                     // +y = up
    const float dz = sp * std::cos(theta);   // +z = face
    const float front = saturate(std::cos(theta));       // 1 on the face, 0 at the ears
    const float fw = smoothstepf(0.02f, 0.72f, front);   // soft face mask

    // ---------------------------------------------------------------- cranium -----
    // 14.8 cm wide, 20.6 cm tall, 19.0 cm deep, with the frontal bone flatter than a
    // sphere, a fuller occiput and the temples pinched in above the cheekbones.
    float rx = 0.0805f, ry = 0.1040f, rz = 0.0960f;
    rx *= 1.0f - 0.050f * bump(theta, phi, 1.15f, 0.95f, 0.40f, 0.30f);   // temple pinch
    rx *= 1.0f - 0.050f * bump(theta, phi, -1.15f, 0.95f, 0.40f, 0.30f);
    Vec3 p(rx * dx, ry * dy, rz * dz);
    if (p.z < 0.0f) p.z *= 1.035f;                   // occiput
    if (p.y > 0.0f) p.y *= 1.012f;                   // crown
    // Parietal width: the widest part of the skull is just above the ears.
    p.x *= 1.0f + 0.030f * bump(std::fabs(theta), phi, 1.45f, 0.80f, 0.55f, 0.45f);

    // ---------------------------------------------------------------- jaw and chin -
    // The mandible runs from the gonial angle (below the ear) forward to the menton. In
    // polar terms that is everything below phi ~1.75, so the taper is driven by how far
    // below the cheek line the point is.
    const float below = saturate((p.y + 0.004f) / -0.098f);     // 0 at the cheek line
    const float jaw = below * below;                            // 1 at the chin
    p.x *= 1.0f - 0.31f * jaw;
    p.z *= 1.0f - 0.07f * jaw;
    // Gonial angle: a soft corner where the jaw turns up towards the ear.
    p.x *= 1.0f + 0.028f * bump(std::fabs(theta), phi, 1.32f, 2.02f, 0.45f, 0.20f) * fw;
    // Chin: comes forward and down, with a shallow mental crease above it.
    p.z += fw * (0.020f * jaw * jaw - 0.004f * bump(theta, phi, 0.0f, 2.26f, 0.20f, 0.050f));
    p.y -= 0.008f * jaw * jaw;
    // Soft underside of the jaw, so the chin is not a razor edge.
    if (below > 0.55f) p.z *= 1.0f - 0.10f * saturate((below - 0.55f) / 0.45f) * (1.0f - fw);

    // ---------------------------------------------------------------- face plane -----
    // The mid-face is flatter than the skull: pull the face back slightly around the
    // cheeks so the nose and lips can read as separate forms.
    p.z -= fw * 0.006f * bump(theta, phi, 0.85f, 1.92f, 0.55f, 0.28f);
    p.z -= fw * 0.005f * bump(theta, phi, -0.85f, 1.92f, 0.55f, 0.28f);

    // ---------------------------------------------------------------- cheekbones ----
    p.z += fw * 0.0065f * bump(std::fabs(theta), phi, 0.68f, 1.74f, 0.34f, 0.15f);
    p.x += fw * (theta >= 0.0f ? 1.0f : -1.0f) * 0.0045f *
           bump(std::fabs(theta), phi, 0.72f, 1.72f, 0.30f, 0.13f);
    // The hollow under the cheekbone (this is what gives the face its planes).
    p.z -= fw * 0.0040f * bump(std::fabs(theta), phi, 0.62f, 2.02f, 0.30f, 0.15f);

    // ---------------------------------------------------------------- brow ----------
    // Glabella and supraorbital ridge, with a slight notch between the brows.
    p.z += fw * 0.0075f * bump(theta, phi, 0.0f, 1.36f, 0.30f, 0.09f);
    p.z += fw * 0.0035f * bump(theta, phi, 0.0f, 1.44f, 0.05f, 0.05f);   // glabella notch
    p.z += fw * 0.0030f * bump(std::fabs(theta), phi, 0.42f, 1.35f, 0.26f, 0.070f);
    // Frontal bossing: the forehead is a smooth convex plane, not a sphere section.
    p.z += fw * 0.0060f * bump(theta, phi, 0.0f, 1.06f, 0.75f, 0.24f);

    // ---------------------------------------------------------------- eyes ----------
    // Socket, then the lids: an upper lid that folds over the eyeball and a lower lid
    // with a soft bag under it. The opening itself is left open for the eyeball mesh.
    const float socket = bump(std::fabs(theta), phi, 0.40f, 1.55f, 0.175f, 0.070f);
    p.z -= fw * 0.0105f * socket;
    const float upperLid = bump(std::fabs(theta), phi, 0.40f, 1.475f, 0.20f, 0.028f);
    p.z += fw * 0.0032f * upperLid;
    const float lowerLid = bump(std::fabs(theta), phi, 0.40f, 1.635f, 0.185f, 0.026f);
    p.z += fw * 0.0026f * lowerLid;
    p.z += fw * 0.0022f * bump(std::fabs(theta), phi, 0.42f, 1.695f, 0.16f, 0.042f);  // tear trough
    // Brow-to-eye plane: the orbital rim.
    p.z += fw * 0.0022f * bump(std::fabs(theta), phi, 0.40f, 1.34f, 0.20f, 0.028f);

    // ---------------------------------------------------------------- nose ----------
    // Bridge (nasion -> tip), then the tip, the alae and the nostrils. The bridge is
    // only ~1 cm wide, so it is expressed as a narrow phi-elongated ridge.
    const float bridge = bump(theta, phi, 0.0f, 1.55f, 0.075f, 0.20f);
    p.z += fw * 0.0130f * bridge;
    const float tip = bump(theta, phi, 0.0f, 1.755f, 0.105f, 0.070f);
    p.z += fw * 0.0125f * tip;
    p.y -= fw * 0.0030f * tip;                                  // the tip droops a little
    // Alae (wings of the nose), and the nostrils recessed beside the columella.
    for (int sgn = -1; sgn <= 1; sgn += 2) {
        const float s = static_cast<float>(sgn);
        p.z += fw * 0.0050f * bump(theta, phi, 0.145f * s, 1.790f, 0.085f, 0.048f);
        p.x += fw * s * 0.0035f * bump(theta, phi, 0.130f * s, 1.795f, 0.075f, 0.044f);
        p.z -= fw * 0.0075f * bump(theta, phi, 0.055f * s, 1.820f, 0.040f, 0.026f);  // nostril
    }
    // Columella / subnasale, then the philtrum groove.
    p.z += fw * 0.0040f * bump(theta, phi, 0.0f, 1.830f, 0.035f, 0.028f);
    p.z -= fw * 0.0022f * bump(theta, phi, 0.0f, 1.915f, 0.030f, 0.042f);

    // ---------------------------------------------------------------- mouth ---------
    // Upper lip with a Cupid's bow, the stomion line, and a fuller lower lip.
    const float upperLip = bump(theta, phi, 0.0f, 2.005f, 0.115f, 0.030f);
    const float lowerLip = bump(theta, phi, 0.0f, 2.085f, 0.100f, 0.034f);
    p.z += fw * 0.0075f * upperLip;
    p.z += fw * 0.0090f * lowerLip;
    p.z += fw * 0.0022f * bump(theta, phi, 0.055f, 1.985f, 0.030f, 0.021f);   // bow peaks
    p.z += fw * 0.0022f * bump(theta, phi, -0.055f, 1.985f, 0.030f, 0.021f);
    p.z -= fw * 0.0038f * bump(theta, phi, 0.0f, 2.045f, 0.130f, 0.010f);     // stomion
    p.z -= fw * 0.0028f * bump(theta, phi, 0.0f, 2.155f, 0.100f, 0.016f);     // mentolabial
    // The corners of the mouth tuck back.
    p.z -= fw * 0.0026f * bump(std::fabs(theta), phi, 0.125f, 2.050f, 0.030f, 0.028f);

    // ---------------------------------------------------------------- neck / nape --
    // The nape tapers into the neck instead of ending in a sphere.
    const float nape = saturate(-std::cos(theta)) * smoothstepf(1.95f, 2.45f, phi);
    p.x *= 1.0f - 0.10f * nape;
    p.z *= 1.0f - 0.06f * nape;

    return p + center_;
}

Vec3 HeadModel::surfaceNormal(float theta, float phi) const {
    const float e = 1.5e-3f;
    const Vec3 p = surfacePoint(theta, phi);
    const Vec3 dt = surfacePoint(theta + e, phi) - surfacePoint(theta - e, phi);
    const Vec3 dp = surfacePoint(theta, phi + e) - surfacePoint(theta, phi - e);
    Vec3 n = cross(dt, dp);
    if (length2(n) < 1e-16f) n = Vec3(0, 1, 0);
    n = normalize(n);
    if (dot(n, p - center_) < 0.0f) n = -n;
    return n;
}

float HeadModel::hairlinePhi(float theta) const {
    // Theta = 0 is the face: the forehead hairline sits at ~63 deg from the crown, and
    // the nape hairline reaches ~2.0 rad. Interpolate smoothly around the head.
    const float front = clampf(std::cos(theta), -1.0f, 1.0f);
    return 1.10f + 0.92f * (0.5f - 0.5f * front);
}

HeadModel::ScalpPatch HeadModel::defaultPatch() const {
    ScalpPatch p;
    p.theta0 = -kPi;
    p.theta1 = kPi;
    p.phiMin = 0.06f;
    p.thetaCenter = 0.0f;
    return p;
}

namespace {

// Non-uniform tessellation: rows and columns are packed where the face needs them.
// The scalp (phi < 0.9) stays coarse - it is covered by hair - while the eyes, nose and
// mouth get roughly three times the sample density of a uniform grid at the same budget.
float phiFromV(float v) {
    static const float vk[6] = {0.00f, 0.16f, 0.30f, 0.46f, 0.62f, 1.00f};
    static const float pk[6] = {0.03f, 0.55f, 0.95f, 1.42f, 1.95f, 2.75f};
    for (int i = 0; i + 1 < 6; ++i) {
        if (v <= vk[i + 1]) {
            const float t = (v - vk[i]) / (vk[i + 1] - vk[i]);
            return pk[i] + (pk[i + 1] - pk[i]) * t;
        }
    }
    return pk[5];
}

float thetaFromU(float u) {
    static const float uk[6] = {0.00f, 0.16f, 0.28f, 0.72f, 0.84f, 1.00f};
    static const float tk[6] = {-kPi, -1.32f, -0.66f, 0.66f, 1.32f, kPi};
    for (int i = 0; i + 1 < 6; ++i) {
        if (u <= uk[i + 1]) {
            const float t = (u - uk[i]) / (uk[i + 1] - uk[i]);
            return tk[i] + (tk[i + 1] - tk[i]) * t;
        }
    }
    return tk[5];
}

// One ear: a small bowl (concha) inside an elliptical rim (helix) with a lobe, in the
// sagittal plane, appended to the head mesh with analytic normals.
void appendEar(Mesh* mesh, float side, Vec3 center) {
    const int na = 40, nr = 8;
    const uint32_t base = static_cast<uint32_t>(mesh->positions.size());
    const float ry = 0.0215f, rz = 0.0125f;   // vertical / horizontal semi-axes
    for (int j = 0; j <= nr; ++j) {
        const float r = static_cast<float>(j) / nr;
        for (int i = 0; i <= na; ++i) {
            const float a = 2.0f * kPi * static_cast<float>(i) / na;
            // Outline: narrower at the top, fuller at the lobe.
            const float lobe = 1.0f + 0.10f * saturate(-std::sin(a));
            const float y = center.y + ry * r * std::sin(a) * lobe;
            const float z = center.z + rz * r * std::cos(a);
            // The rim stands proud of the skull; the concha sinks into it.
            float x = center.x + side * (0.0020f + 0.0060f * r * r - 0.0025f * (1.0f - r));
            // Helix ridge.
            x += side * 0.0018f * std::exp(-0.5f * sq((r - 0.86f) / 0.10f));
            // The lobe is a soft blob of its own.
            x += side * 0.0012f * saturate(-std::sin(a)) * sq(r);
            mesh->positions.push_back(Vec3(x, y, z));
            mesh->uvs.push_back(Vec2(static_cast<float>(i) / na, r));
            Vec3 n(side * 0.55f, 0.0f, 0.0f);
            if (r > 0.02f) n = normalize(Vec3(side * 0.55f, (y - center.y) / ry,
                                              (z - center.z) / rz));
            mesh->normals.push_back(n);
        }
    }
    for (int j = 0; j < nr; ++j) {
        for (int i = 0; i < na; ++i) {
            const uint32_t a0 = base + static_cast<uint32_t>(j * (na + 1) + i);
            const uint32_t a1 = a0 + 1;
            const uint32_t b0 = a0 + static_cast<uint32_t>(na + 1);
            const uint32_t b1 = b0 + 1;
            if (side > 0.0f) {
                mesh->indices.insert(mesh->indices.end(), {a0, b0, a1, a1, b0, b1});
            } else {
                mesh->indices.insert(mesh->indices.end(), {a0, a1, b0, a1, b1, b0});
            }
        }
    }
}

// Shared builder for the detailed head and the cheap crowd head.
void buildHeadSurface(Mesh* mesh, int nu, int nv, const HeadModel& head, Vec3 center,
                      bool withEars) {
    mesh->positions.clear();
    mesh->normals.clear();
    mesh->uvs.clear();
    mesh->indices.clear();
    for (int j = 0; j <= nv; ++j) {
        const float phi = phiFromV(static_cast<float>(j) / nv);
        for (int i = 0; i <= nu; ++i) {
            const float theta = thetaFromU(static_cast<float>(i) / nu);
            mesh->positions.push_back(head.surfacePoint(theta, phi));
            mesh->normals.push_back(head.surfaceNormal(theta, phi));
            mesh->uvs.push_back(Vec2(static_cast<float>(i) / nu, static_cast<float>(j) / nv));
        }
    }
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const uint32_t a = static_cast<uint32_t>(j * (nu + 1) + i);
            const uint32_t b = a + 1;
            const uint32_t c = a + static_cast<uint32_t>(nu + 1);
            const uint32_t d = c + 1;
            mesh->indices.insert(mesh->indices.end(), {a, c, b, b, c, d});
        }
    }
    if (withEars) {
        const Vec3 earCenter = head.surfacePoint(1.57f, 1.56f);
        appendEar(mesh, +1.0f, Vec3(earCenter.x - 0.002f, earCenter.y + 0.002f, earCenter.z - 0.004f));
        const Vec3 earCenterL = head.surfacePoint(-1.57f, 1.56f);
        appendEar(mesh, -1.0f,
                  Vec3(earCenterL.x + 0.002f, earCenterL.y + 0.004f, earCenterL.z - 0.004f));
    }
    (void)center;
    mesh->name = "female_head";
    mesh->materialIds.assign(mesh->triangleCount(), 0);
}

}  // namespace

void HeadModel::buildHead() {
    // The hero head: dense enough for the nose wings, the lids and the lips to read.
    buildHeadSurface(&head_, 224, 132, *this, center_, true);
}

void HeadModel::buildHeadLow() {
    // The crowd head: one is drawn per character (Figure 1), where the face is a few
    // pixels tall, so the same surface is sampled coarsely and the ears are dropped.
    buildHeadSurface(&headLow_, 80, 56, *this, center_, false);
}

void HeadModel::buildBody() {
    // Neck + shoulders: a lathe from the jaw down to the shoulder line.
    const int nu = 64;
    struct Profile { float y, r; };
    const Profile profile[] = {{0.04f, 0.052f}, {-0.03f, 0.045f}, {-0.09f, 0.048f},
                               {-0.14f, 0.058f}, {-0.19f, 0.085f}, {-0.23f, 0.135f},
                               {-0.31f, 0.175f}, {-0.31f, 0.19f}};
    const int pn = static_cast<int>(sizeof(profile) / sizeof(profile[0]));
    body_.positions.clear();
    body_.normals.clear();
    body_.uvs.clear();
    body_.indices.clear();
    for (int j = 0; j < pn; ++j) {
        for (int i = 0; i <= nu; ++i) {
            const float u = static_cast<float>(i) / nu;
            const float theta = 2.0f * kPi * u;
            const float r = profile[j].r;
            Vec3 p(std::cos(theta) * r, profile[j].y, std::sin(theta) * r * 0.75f);
            p += Vec3(0.0f, center_.y, 0.0f);
            body_.positions.push_back(p);
            body_.uvs.push_back(Vec2(u, static_cast<float>(j) / (pn - 1)));
        }
    }
    for (int j = 0; j + 1 < pn; ++j) {
        for (int i = 0; i < nu; ++i) {
            const uint32_t a = static_cast<uint32_t>(j * (nu + 1) + i);
            const uint32_t b = a + 1;
            const uint32_t c = a + (nu + 1);
            const uint32_t d = c + 1;
            body_.indices.insert(body_.indices.end(), {a, c, b, b, c, d});
        }
    }
    body_.name = "body";
    body_.computeNormals();
    body_.materialIds.assign(body_.triangleCount(), 0);
}


// =============================================================================
//  Face masks
// =============================================================================
namespace {

// Inverse of the head parameterization: a point on (or just inside) the surface maps back
// to the azimuth and polar angle the anatomy is written in.
inline void toAngles(const Vec3& local, float* theta, float* phi) {
    *theta = std::atan2(local.x, local.z);
    const float c = clampf(local.y / 0.1040f, -1.0f, 1.0f);
    *phi = std::acos(c);
}

// A smooth band around a value with soft shoulders.
inline float band(float x, float centre, float half, float feather) {
    const float d = std::fabs(x - centre);
    return 1.0f - smoothstepf(half, half + feather, d);
}

}  // namespace

FaceMasks headFaceMasks(const Vec3& localPos) {
    FaceMasks m;
    float theta = 0.0f, phi = 0.0f;
    toAngles(localPos, &theta, &phi);
    const float at = std::fabs(theta);
    const float front = saturate(std::cos(theta));
    const float fw = smoothstepf(0.05f, 0.62f, front);

    // Eyebrow: a feathered band above the eye, thicker at the inner end, arched outward.
    {
        const float arch = 1.468f - 0.026f * smoothstepf(0.18f, 0.62f, at);
        const float thin = 1.0f - 0.45f * smoothstepf(0.42f, 0.62f, at);
        const float bandMask = band(phi, arch, 0.019f * thin, 0.024f * thin) *
                               band(at, 0.40f, 0.165f, 0.10f);
        // Fade the inner and outer tips so the brow does not end abruptly.
        m.brow = bandMask * fw * (0.55f + 0.45f * smoothstepf(0.12f, 0.30f, at));
    }
    // Lips: upper and lower vermilion either side of the stomion line.
    {
        const float halfWidth = 0.165f - 0.035f * smoothstepf(1.99f, 2.12f, phi);
        const float w = band(at, 0.0f, halfWidth, 0.045f);
        m.lipUpper = w * band(phi, 1.998f, 0.036f, 0.016f);
        m.lipLower = w * band(phi, 2.092f, 0.044f, 0.018f);
        m.mouthLine = band(at, 0.0f, 0.115f, 0.060f) * band(phi, 2.045f, 0.008f, 0.012f);
    }
    // Lash lines: the upper lid edge is the darkest part of the eye region.
    m.lashUpper = 0.0f;   // the lash line belongs to the eye shader, not the skin
    m.lashLower = 0.0f;
    // Nostrils.
    m.nostril = band(at, 0.058f, 0.030f, 0.020f) * band(phi, 1.822f, 0.016f, 0.012f) * fw;
    // The nose: a narrow ridge down the dorsum, and the side walls beside it that fall
    // into shadow. Without these the nose reads as a flat triangle on the face.
    m.noseBridge = fw * band(at, 0.0f, 0.032f, 0.026f) * band(phi, 1.705f, 0.105f, 0.050f);
    m.noseSide = fw * (bump(at, phi, 0.050f, 1.735f, 0.026f, 0.080f) +
                       0.6f * bump(at, phi, 0.062f, 1.812f, 0.024f, 0.030f));
    m.noseSide = saturate(m.noseSide);
    // Blush: cheeks, the tip of the nose and the ears (where the skin is thin).
    m.blush = fw * (0.9f * bump(at, phi, 0.62f, 1.90f, 0.26f, 0.16f) +
                    0.5f * bump(theta, phi, 0.0f, 1.775f, 0.12f, 0.05f));
    m.blush += 0.7f * bump(at, phi, 1.53f, 1.56f, 0.20f, 0.22f);   // ears
    m.blush = saturate(m.blush);
    // Ambient occlusion in the creases.
    m.ao = fw * (0.85f * bump(at, phi, 0.40f, 1.55f, 0.22f, 0.10f) +     // socket
                 0.45f * bump(at, phi, 0.30f, 1.93f, 0.10f, 0.13f) +     // nasolabial fold
                 0.55f * bump(at, phi, 0.0f, 2.045f, 0.14f, 0.012f) +    // lip seam
                 0.40f * bump(at, phi, 0.0f, 2.27f, 0.22f, 0.09f) +    // under the lower lip
                 0.30f * bump(at, phi, 0.56f, 1.74f, 0.15f, 0.07f) +    // hollow under the cheekbone
                 0.25f * bump(at, phi, 0.10f, 1.30f, 0.10f, 0.05f));    // temple
    m.ao += 0.5f * smoothstepf(2.30f, 2.70f, phi);                        // under the jaw
    m.ao = saturate(m.ao);
    return m;
}

// =============================================================================
//  Hair mesh construction
// =============================================================================
HairMesh buildHairMesh(const HeadModel& head, const HairStyleSpec& style) {
    const int nu = std::max(2, style.gridU);   // vertices around
    const int nv = std::max(2, style.gridV);   // vertices crown -> hairline
    const int L = std::max(2, style.numLayers);
    const HeadModel::ScalpPatch patch = head.defaultPatch();

    HairMesh mesh;
    mesh.name = style.name;
    const uint32_t numVerts = static_cast<uint32_t>(nu * nv);
    std::vector<HairMeshFace> faces;
    faces.reserve(static_cast<size_t>((nu - 1) * (nv - 1)));
    for (int j = 0; j + 1 < nv; ++j) {
        for (int i = 0; i + 1 < nu; ++i) {
            HairMeshFace f;
            f.count = 4;
            f.v[0] = static_cast<uint32_t>(j * nu + i);
            f.v[1] = static_cast<uint32_t>(j * nu + i + 1);
            f.v[2] = static_cast<uint32_t>((j + 1) * nu + i + 1);
            f.v[3] = static_cast<uint32_t>((j + 1) * nu + i);
            faces.push_back(f);
        }
    }

    // UV layout: a plain grid over the scalp patch (the "manually prepared" mapping of
    // Section 3.1). Kept inside [0.02, 0.98] to avoid sampling exactly on the border.
    std::vector<Vec2> uv(numVerts);
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const float u = 0.02f + 0.96f * (static_cast<float>(i) / (nu - 1));
            const float v = 0.02f + 0.96f * (static_cast<float>(j) / (nv - 1));
            uv[static_cast<size_t>(j * nu + i)] = Vec2(u, v);
        }
    }
    mesh.setTopology(numVerts, static_cast<uint32_t>(L), faces, uv);
    mesh.setLayerW(defaultLayerW(static_cast<uint32_t>(L)));

    // Root layer on the scalp.
    std::vector<float> theta(numVerts), phi(numVerts);
    for (int j = 0; j < nv; ++j) {
        const float v = static_cast<float>(j) / (nv - 1);
        for (int i = 0; i < nu; ++i) {
            const float u = static_cast<float>(i) / (nu - 1);
            const float th = patch.theta0 + (patch.theta1 - patch.theta0) * u;
            const float hairline = head.hairlinePhi(th) + style.hairlineDrop;
            const float ph = patch.phiMin + (hairline - patch.phiMin) * v;
            const size_t idx = static_cast<size_t>(j * nu + i);
            theta[idx] = th;
            phi[idx] = ph;
            mesh.layerVertexMutable(0, static_cast<uint32_t>(idx)) = head.surfacePoint(th, ph);
        }
    }
    mesh.computeScalpNormals();

    // Extrusion layers. A hair mesh's layers describe the hair *volume*: the first
    // layer leaves the scalp along the normal, the following layers turn towards the
    // direction in which hair falls, so that the tips hang around the head instead of
    // forming a cone. Called "the outer layers define the hairdo" in the paper.
    for (int l = 1; l < L; ++l) {
        const float s = static_cast<float>(l) / static_cast<float>(L - 1);
        for (int j = 0; j < nv; ++j) {
            for (int i = 0; i < nu; ++i) {
                const size_t idx = static_cast<size_t>(j * nu + i);
                const uint32_t vid = static_cast<uint32_t>(idx);
                const Vec3 root = mesh.layerVertex(0, vid);
                const Vec3 n = normalize(root - head.center());
                const float v = static_cast<float>(j) / (nv - 1);
                const float front = clampf(std::cos(theta[idx]), 0.0f, 1.0f);  // 1 = face
                const float back = 1.0f - front;
                const float at = std::fabs(theta[idx]);

                // Direction in which hair flows over the scalp: the surface tangent
                // pointing away from the crown, which becomes straight down off the head.
                const float e = 1e-3f;
                const float phiHere = phi[idx];
                Vec3 down = head.surfacePoint(theta[idx], phiHere + e) -
                            head.surfacePoint(theta[idx], phiHere - e);
                if (length2(down) < 1e-12f) down = Vec3(0, -1, 0);
                down = normalize(down);
                down = normalize(Vec3(down.x, std::min(down.y, -0.25f), down.z));
                Vec3 dir = normalize(lerp(n, down, smoothstepf(0.0f, 0.45f, s)));

                // A side part is what turns a wig into a hairstyle: the direction field
                // turns sideways away from the parting line, so the hair falls as two
                // curtains framing the face instead of as one cap. The rows on the part
                // itself are shortened, which opens a visible line of scalp.
                Vec3 across = head.surfacePoint(theta[idx] + e, phiHere) -
                              head.surfacePoint(theta[idx] - e, phiHere);
                if (length2(across) < 1e-12f) across = Vec3(1, 0, 0);
                across = normalize(across);
                const float partAt = 0.58f;
                const float partGap = smoothstepf(0.11f, 0.0f, std::fabs(at - partAt));
                const float lateral = clampf((at - partAt) * 1.45f, -1.0f, 1.0f);
                dir = normalize(dir + across * (lateral * (0.30f + 0.55f * s)));

                // At the hairline the hair must not simply fall forwards: that is what
                // makes a procedural hairdo look like a mop with a fringe over the brows.
                // It is combed sideways off the part and lifted slightly, so the fringe
                // frames the temples and the forehead stays readable.
                const float faceFlow = saturate((front - 0.22f) * 2.1f);
                if (faceFlow > 0.001f) {
                    const float sgn = (at >= partAt) ? 1.0f : -1.0f;
                    Vec3 sweepDir = normalize(across * sgn + Vec3(0.0f, 0.42f, -0.12f));
                    dir = normalize(lerp(dir, sweepDir, faceFlow * (0.45f + 0.45f * s)));
                }

                // Hair length: long at the back, short at the face so the features stay
                // visible (a hair mesh's front rows are always short or styled away).
                const float lengthScale = (1.0f + style.backBias * back) * (0.62f + 0.58f * v) *
                                          (1.0f - 0.74f * std::pow(front, 1.15f)) *
                                          (1.0f - 0.62f * partGap);
                const float grow = std::max(0.003f, style.length * lengthScale * s * 0.85f);
                // Volume: the hair volume is widest around mid-length.
                const float puff = style.volume * std::sin(kPi * s) * 1.2f;
                // Hair that leaves near the part lies flat against the scalp first, then
                // peels away from it, which is how the layers of a real hairstyle stack.
                const float rise = smoothstepf(0.0f, 0.14f, s) * (1.0f - 0.50f * partGap);
                Vec3 p = root + dir * (grow * rise) + n * (puff * rise);
                mesh.layerVertexMutable(static_cast<uint32_t>(l), vid) = p;
            }
        }
    }

    // Strand counts per bundle so that the model totals the requested strand count.
    const uint32_t bundles = static_cast<uint32_t>(faces.size());
    const uint32_t perBundle = std::max(1u, style.totalStrands / std::max(1u, bundles));
    mesh.setUniformStrandCount(perBundle);
    return mesh;
}

HairMesh buildTeapotHairMesh(int gridU, int gridV, int numLayers, float length, uint32_t seed) {
    HairMesh mesh;
    mesh.name = "teapot_hair";
    const int nu = gridU, nv = gridV;
    const uint32_t numVerts = static_cast<uint32_t>(nu * nv);
    std::vector<HairMeshFace> faces;
    for (int j = 0; j + 1 < nv; ++j)
        for (int i = 0; i + 1 < nu; ++i) {
            HairMeshFace f;
            f.v[0] = static_cast<uint32_t>(j * nu + i);
            f.v[1] = static_cast<uint32_t>(j * nu + i + 1);
            f.v[2] = static_cast<uint32_t>((j + 1) * nu + i + 1);
            f.v[3] = static_cast<uint32_t>((j + 1) * nu + i);
            faces.push_back(f);
        }
    std::vector<Vec2> uv(numVerts);
    for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i)
            uv[static_cast<size_t>(j * nu + i)] =
                Vec2(0.02f + 0.96f * i / (nu - 1), 0.02f + 0.96f * j / (nv - 1));
    mesh.setTopology(numVerts, static_cast<uint32_t>(numLayers), faces, uv);
    mesh.setLayerW(defaultLayerW(static_cast<uint32_t>(numLayers)));

    // A cap on the teapot's lid: the hair mesh hugs the lid (radius 0.058 on a 0.062 lid),
    // so the pot's body, spout and handle stay visible, which is what Figure 9 shows.
    const Vec3 center(0.0f, 0.112f, 0.0f);
    for (int j = 0; j < nv; ++j) {
        const float v = static_cast<float>(j) / (nv - 1);
        for (int i = 0; i < nu; ++i) {
            const float u = static_cast<float>(i) / (nu - 1);
            const float th = 2.0f * kPi * u;
            const float ph = 0.25f + (1.45f - 0.25f) * v;
            const Vec3 n(std::sin(ph) * std::cos(th), std::cos(ph), std::sin(ph) * std::sin(th));
            const size_t idx = static_cast<size_t>(j * nu + i);
            mesh.layerVertexMutable(0, static_cast<uint32_t>(idx)) = center + n * 0.058f;
            for (int l = 1; l < numLayers; ++l) {
                const float s = static_cast<float>(l) / (numLayers - 1);
                const Vec3 dir = normalize(bendDown(n, 0.75f * std::pow(s, 0.8f)));
                mesh.layerVertexMutable(static_cast<uint32_t>(l), static_cast<uint32_t>(idx)) =
                    center + n * 0.058f + dir * (length * std::pow(s, 0.95f)) +
                    n * (0.006f * s);   // modest outward puff, so the pot stays visible
            }
        }
    }
    mesh.computeScalpNormals();
    mesh.setUniformStrandCount(std::max(1u, 50000u / static_cast<uint32_t>(faces.size())));
    (void)seed;
    return mesh;
}

Mesh buildTeapot() {
    Mesh m;
    m.name = "teapot";
    auto addRing = [&](float y, float r, int nu) {
        const uint32_t base = static_cast<uint32_t>(m.positions.size());
        for (int i = 0; i <= nu; ++i) {
            const float th = 2.0f * kPi * i / nu;
            m.positions.push_back(Vec3(std::cos(th) * r, y, std::sin(th) * r));
            m.uvs.push_back(Vec2(static_cast<float>(i) / nu, y));
        }
        return base;
    };
    // Body profile of revolution.
    struct P { float y, r; };
    const P body[] = {{-0.075f, 0.045f}, {-0.07f, 0.075f}, {-0.045f, 0.105f}, {-0.01f, 0.118f},
                      {0.03f, 0.112f}, {0.06f, 0.092f}, {0.08f, 0.07f}, {0.09f, 0.062f}};
    const int nb = static_cast<int>(sizeof(body) / sizeof(body[0]));
    const int nu = 48;
    for (int j = 0; j < nb; ++j) addRing(body[j].y, body[j].r, nu);
    for (int j = 0; j + 1 < nb; ++j) {
        for (int i = 0; i < nu; ++i) {
            const uint32_t a = static_cast<uint32_t>(j * (nu + 1) + i);
            const uint32_t b = a + 1, c = a + nu + 1, d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    }
    // Lid.
    const uint32_t lidBase = static_cast<uint32_t>(m.positions.size());
    const int nl = 8;
    for (int j = 0; j <= nl; ++j) {
        const float t = static_cast<float>(j) / nl;
        const float r = 0.062f * (1.0f - t) + 0.012f * t;
        const float y = 0.09f + 0.022f * std::sin(t * kPi * 0.5f);
        addRing(y, std::max(r, 0.001f), 24);
    }
    for (int j = 0; j < nl; ++j) {
        for (int i = 0; i < 24; ++i) {
            const uint32_t a = lidBase + static_cast<uint32_t>(j * 25 + i);
            const uint32_t b = a + 1, c = a + 25, d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    }
    // Spout: swept tube.
    auto addTube = [&](Vec3 from, Vec3 to, float r0, float r1, int steps, int nring) {
        const Vec3 dir = normalize(to - from);
        const Vec3 up = std::fabs(dir.y) > 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        const Vec3 s1 = normalize(cross(dir, up));
        const Vec3 s2 = cross(dir, s1);
        const uint32_t base = static_cast<uint32_t>(m.positions.size());
        for (int k = 0; k <= steps; ++k) {
            const float t = static_cast<float>(k) / steps;
            const Vec3 c = from + (to - from) * t + Vec3(0, 0.05f * std::sin(t * kPi), 0);
            const float r = r0 + (r1 - r0) * t;
            for (int i = 0; i <= nring; ++i) {
                const float th = 2.0f * kPi * i / nring;
                m.positions.push_back(c + s1 * (std::cos(th) * r) + s2 * (std::sin(th) * r));
                m.uvs.push_back(Vec2(static_cast<float>(i) / nring, t));
            }
        }
        for (int k = 0; k < steps; ++k)
            for (int i = 0; i < nring; ++i) {
                const uint32_t a = base + static_cast<uint32_t>(k * (nring + 1) + i);
                const uint32_t b = a + 1, c = a + nring + 1, d = c + 1;
                m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
            }
    };
    addTube(Vec3(0.09f, 0.02f, 0.0f), Vec3(0.20f, 0.075f, 0.0f), 0.022f, 0.012f, 10, 14);
    addTube(Vec3(-0.09f, 0.02f, 0.0f), Vec3(-0.17f, 0.005f, 0.0f), 0.014f, 0.016f, 10, 12);
    addTube(Vec3(-0.17f, 0.005f, 0.0f), Vec3(-0.14f, -0.05f, 0.0f), 0.016f, 0.014f, 8, 12);
    addTube(Vec3(-0.14f, -0.05f, 0.0f), Vec3(-0.075f, -0.06f, 0.0f), 0.014f, 0.012f, 6, 12);
    m.computeNormals();
    m.materialIds.assign(m.triangleCount(), 2);
    return m;
}

void bundleScalpMetrics(const HairMesh& mesh, std::vector<float>* bundleSize,
                        std::vector<Vec3>* bundleCenter) {
    bundleSize->assign(mesh.numFaces(), 0.0f);
    bundleCenter->assign(mesh.numFaces(), Vec3(0, 0, 0));
    for (uint32_t f = 0; f < mesh.numFaces(); ++f) {
        const HairMeshFace& face = mesh.face(f);
        Vec3 c(0, 0, 0);
        for (uint32_t i = 0; i < face.count; ++i) c += mesh.rootVertex(face.v[i]);
        c = c * (1.0f / static_cast<float>(face.count));
        float maxEdge = 0.0f;
        for (uint32_t i = 0; i < face.count; ++i) {
            const Vec3 a = mesh.rootVertex(face.v[i]);
            const Vec3 b = mesh.rootVertex(face.v[(i + 1) % face.count]);
            maxEdge = std::max(maxEdge, length(a - b));
        }
        (*bundleCenter)[f] = c;
        // Eq. 7 uses the bundle's scalp face size: the radius of its bounding sphere or
        // the length of its longest edge. We use the longest edge.
        (*bundleSize)[f] = maxEdge;
    }
}

// =============================================================================
//  Figure 10 presets
// =============================================================================
HairStyleSpec HairStyleSpec::fromFigure10(int index, uint32_t seed) {
    HairStyleSpec s;
    s.seed = seed;
    // The five vertex counts reported in Figure 10 (185, 477, 1316, 3236, 7892) are
    // approximated by matching root-layer resolutions.
    switch (index % 10) {
        case 0: s.name = "long_straight";  s.gridU = 16; s.gridV = 12; s.numLayers = 4;
                s.length = 0.15f; s.fall = 0.92f; s.volume = 0.005f; s.backBias = 0.25f; break;
        case 1: s.name = "wavy_medium";    s.gridU = 22; s.gridV = 22; s.numLayers = 4;
                s.length = 0.20f; s.fall = 0.75f; s.volume = 0.014f; s.backBias = 0.3f; break;
        case 2: s.name = "curly_short";    s.gridU = 28; s.gridV = 24; s.numLayers = 4;
                s.length = 0.13f; s.fall = 0.55f; s.volume = 0.022f; s.backBias = 0.2f; break;
        case 3: s.name = "afro";           s.gridU = 30; s.gridV = 26; s.numLayers = 4;
                s.length = 0.16f; s.fall = 0.25f; s.volume = 0.055f; s.backBias = 0.1f; break;
        case 4: s.name = "bob";            s.gridU = 26; s.gridV = 20; s.numLayers = 4;
                s.length = 0.15f; s.fall = 0.85f; s.volume = 0.012f; s.backBias = 0.15f; break;
        case 5: s.name = "ponytail";       s.gridU = 24; s.gridV = 18; s.numLayers = 5;
                s.length = 0.30f; s.fall = 1.0f; s.volume = 0.006f; s.backBias = 0.55f; break;
        case 6: s.name = "beach_waves";    s.gridU = 34; s.gridV = 26; s.numLayers = 4;
                s.length = 0.28f; s.fall = 0.8f; s.volume = 0.016f; s.backBias = 0.35f; break;
        case 7: s.name = "curly_long";     s.gridU = 40; s.gridV = 32; s.numLayers = 4;
                s.length = 0.24f; s.fall = 0.7f; s.volume = 0.026f; s.backBias = 0.3f; break;
        case 8: s.name = "pixie";          s.gridU = 20; s.gridV = 16; s.numLayers = 3;
                s.length = 0.075f; s.fall = 0.5f; s.volume = 0.010f; s.backBias = 0.0f; break;
        default: s.name = "platinum_bob";  s.gridU = 30; s.gridV = 24; s.numLayers = 4;
                s.length = 0.17f; s.fall = 0.88f; s.volume = 0.014f; s.backBias = 0.2f; break;
    }
    s.styling = StylingParams::fromPreset(index, seed);
    // Styling amplitudes live in object space: scale them with the hair length so that
    // the same parameters read correctly on short and long styles.
    s.styling.amplitudeScale = 0.055f * s.length;
    return s;
}

}  // namespace silk
