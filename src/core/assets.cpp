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
        const float theta = 0.38f * static_cast<float>(side);
        const float phi = 1.14f;
        // Seated in the eye socket: the socket is 7.5 mm deep, so an eyeball of radius
        // 11.5 mm placed 6 mm behind the (deformed) surface leaves the cornea proud of the
        // socket without looking like a bug eye.
        const Vec3 c = surfacePoint(theta, phi) - surfaceNormal(theta, phi) * 0.0042f;
        const float r = 0.0122f;
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

Vec3 HeadModel::surfacePoint(float theta, float phi) const {
    // theta = 0 is the face (+Z), theta = +pi/2 is the right ear, theta = pi the nape.
    const float sph = std::sin(phi), cph = std::cos(phi);
    const float sx = sph * std::sin(theta);
    const float sz = sph * std::cos(theta);
    const float front = clampf(std::cos(theta), 0.0f, 1.0f);

    float x = 0.080f * sx;
    float y = 0.106f * cph;
    float z = 0.092f * sz;

    // --- jaw and chin (below the equator) ---
    if (y < 0.0f) {
        const float t = clampf(-y / 0.106f, 0.0f, 1.0f);
        const float taper = 1.0f - 0.34f * t * t;
        x *= taper;
        z *= taper * 0.92f;
        z += 0.016f * t * t * front;   // the face comes forward to the chin
        y -= 0.010f * t * t;
    }
    // --- nose ridge and tip ---
    const float noseBand = std::exp(-std::pow(phi - 1.42f, 2) / 0.020f);
    z += 0.019f * noseBand * std::exp(-std::pow(theta, 2) / 0.020f);
    z += 0.008f * std::exp(-std::pow(phi - 1.55f, 2) / 0.0016f) * std::exp(-std::pow(theta, 2) / 0.004f);
    // --- brow ridge ---
    z += 0.005f * std::exp(-std::pow(phi - 1.06f, 2) / 0.004f) * std::exp(-std::pow(theta, 2) / 0.55f);
    // --- eye sockets ---
    const float socket = std::exp(-std::pow(std::fabs(theta) - 0.38f, 2) / 0.012f) *
                         std::exp(-std::pow(phi - 1.14f, 2) / 0.0030f);
    z -= 0.0075f * socket;
    // --- cheek bones ---
    z += 0.004f * std::exp(-std::pow(std::fabs(theta) - 0.55f, 2) / 0.05f) *
         std::exp(-std::pow(phi - 1.40f, 2) / 0.02f);
    // --- lips ---
    z += 0.0055f * std::exp(-std::pow(phi - 1.72f, 2) / 0.0012f) * std::exp(-std::pow(theta, 2) / 0.020f);
    // --- ears ---
    const float ear = std::exp(-std::pow(phi - 1.25f, 2) / 0.035f) *
                      std::exp(-std::pow(std::fabs(theta) - 1.60f, 2) / 0.012f);
    x += (theta > 0.0f ? 1.0f : -1.0f) * 0.016f * ear;
    // --- back of the skull slightly flattened, crown a little taller ---
    if (z < 0.0f) z *= 0.97f;
    if (y > 0.05f) y *= 1.03f;

    return Vec3(x, y, z) + center_;
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
    return 0.98f + 1.00f * (0.5f - 0.5f * front);
}

HeadModel::ScalpPatch HeadModel::defaultPatch() const {
    ScalpPatch p;
    p.theta0 = -kPi;
    p.theta1 = kPi;
    p.phiMin = 0.06f;
    p.thetaCenter = 0.0f;
    return p;
}

void HeadModel::buildHead() {
    const int nu = 96, nv = 64;
    const float phi0 = 0.03f, phi1 = 2.35f;
    for (int j = 0; j <= nv; ++j) {
        const float v = static_cast<float>(j) / nv;
        const float phi = phi0 + (phi1 - phi0) * v;
        for (int i = 0; i <= nu; ++i) {
            const float u = static_cast<float>(i) / nu;
            const float theta = -kPi + 2.0f * kPi * u;
            head_.positions.push_back(surfacePoint(theta, phi));
            head_.normals.push_back(surfaceNormal(theta, phi));
            head_.uvs.push_back(Vec2(u, v));
        }
    }
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const uint32_t a = static_cast<uint32_t>(j * (nu + 1) + i);
            const uint32_t b = a + 1;
            const uint32_t c = a + (nu + 1);
            const uint32_t d = c + 1;
            head_.indices.insert(head_.indices.end(), {a, c, b, b, c, d});
        }
    }
    head_.name = "female_head";
    head_.materialIds.assign(head_.triangleCount(), 0);
}

void HeadModel::buildBody() {
    // Neck + shoulders: a lathe from the jaw down to the shoulder line.
    const int nu = 64;
    struct Profile { float y, r; };
    const Profile profile[] = {{0.06f, 0.052f}, {-0.02f, 0.045f}, {-0.09f, 0.048f},
                               {-0.14f, 0.058f}, {-0.19f, 0.085f}, {-0.23f, 0.135f},
                               {-0.27f, 0.175f}, {-0.31f, 0.19f}};
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

                // Hair length: long at the back, short and swept back at the face so the
                // features stay visible (a hair mesh's front rows are always short or
                // styled away from the face).
                const float lengthScale = (1.0f + style.backBias * back) * (0.55f + 0.65f * v) *
                                          (1.0f - 0.86f * std::pow(front, 1.2f));
                const float grow = std::max(0.003f, style.length * lengthScale * s * 0.85f);
                // Volume: the hair volume is widest around mid-length.
                const float puff = style.volume * std::sin(kPi * s) * 1.2f;
                // Front rows are swept back over the crown instead of hanging over the
                // face, which is how a modelled hair mesh keeps the face readable.
                const float sweep = saturate((front - 0.35f) * 1.6f) * s;
                dir = normalize(dir + Vec3(0.0f, -0.12f, -0.95f) * sweep);
                Vec3 p = root + dir * grow + n * puff;
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
