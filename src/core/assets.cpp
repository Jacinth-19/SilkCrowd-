#include "silkcrowd/assets.h"

#include <algorithm>
#include <cmath>

#include "silkcrowd/gltf.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <fstream>

namespace silk {
namespace {

std::string g_headScanAsset;   // "" until resolved

}  // namespace

std::string assetPath(const std::string& relative) {
    static std::string root;
    static const bool resolved = [] {
        if (const char* env = std::getenv("SILKCROWD_ASSET_DIR")) {
            root = env;
            if (!root.empty() && root.back() != '/') root.push_back('/');
            return true;
        }
        const char* candidates[] = {"assets/", "../assets/", "../../assets/",
#ifdef SILKCROWD_SOURCE_DIR
                                    SILKCROWD_SOURCE_DIR "/assets/"
#else
                                    ""
#endif
        };
        for (const char* candidate : candidates) {
            if (!candidate || !*candidate) continue;
            std::ifstream probe(std::string(candidate) + "heads/head_scan.glb", std::ios::binary);
            std::ifstream probe2(std::string(candidate) + "hairmeshes/04_bob.skhm",
                                 std::ios::binary);
            if (probe.good() || probe2.good()) {
                root = candidate;
                return true;
            }
        }
        root = "assets/";
        return true;
    }();
    (void)resolved;
    return root + relative;
}

std::string headScanAsset() {
    if (g_headScanAsset.empty()) g_headScanAsset = assetPath("heads/head_scan.glb");
    return g_headScanAsset;
}

void setHeadScanAsset(const std::string& path) { g_headScanAsset = path; }

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
    if (scan_) {
        const float r = scanRadius(theta, phi);
        if (r > 0.0f) {
            const float sp = std::sin(phi), cp = std::cos(phi);
            return center_ + Vec3(sp * std::sin(theta), cp, sp * std::cos(theta)) * r;
        }
        // Below the jaw the scan's bust has a neck: rays that miss it (in front of the
        // throat, under the chin) fall back to the procedural surface, which is only ever
        // sampled by hanging hair and by the hairline extension.
    }
    return surfacePointProcedural(theta, phi);
}

Vec3 HeadModel::surfacePointProcedural(float theta, float phi) const {
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
    // At the pole (phi = 0) the phi central difference would step outside the surface, so
    // it is clamped; the pole is a point where every meridian meets, and its normal is up.
    const float phiA = std::max(0.0f, phi - e);
    const float phiB = std::min(3.14159f, phi + e);
    const Vec3 dt = surfacePoint(theta + e, phi) - surfacePoint(theta - e, phi);
    const Vec3 dp = surfacePoint(theta, phiB) - surfacePoint(theta, phiA);
    Vec3 n = cross(dt, dp);
    if (length2(n) < 1e-16f) n = Vec3(0, 1, 0);
    n = normalize(n);
    if (dot(n, p - center_) < 0.0f) n = -n;
    return n;
}

namespace {

// The hairline as a function of azimuth, in the shared parameterization: 63 deg from the
// crown over the forehead, sweeping back to 116 deg at the nape, low over the temples. This is
// the *anatomical* hairline; a fringe is a styling choice, so styles that have one extend it
// with `HairStyleSpec::hairlineDrop` instead of the head moving its own hairline (which is what
// a global change did -- it gave the male scan a fringe over the eyes).
inline float hairlineAngle(float theta) {
    const float front = clampf(std::cos(theta), -1.0f, 1.0f);
    return 1.10f + 0.92f * (0.5f - 0.5f * front);
}

}  // namespace

//==============================================================================
//  Scanned head
//==============================================================================
bool HeadModel::loadScan(const std::string& glbPath, std::string* error) {
    GltfAsset asset;
    if (!loadGlb(glbPath, &asset, error)) return false;
    if (asset.mesh.uvs.size() != asset.mesh.positions.size()) {
        if (error) *error = "scan has no TEXCOORD_0";
        return false;
    }

    // ---- the head's frame ----------------------------------------------------
    // tools/prep_head.py has already put the bust in SilkCrowd's head space: midline at
    // x = 0, crown at y = 0.128, skull centroid at z = 0. center_ is the point the scalp
    // parameterization radiates from: one third of the way down the skull from the crown,
    // on the ear axis, which is where a real cranium is star-shaped in every direction the
    // hair mesh samples.
    Mesh mesh = asset.mesh;
    float crown = -1e30f;
    for (const Vec3& p : mesh.positions) crown = std::max(crown, p.y);
    // Half way down the head: the ear canal, which is the point a real cranium is
    // star-shaped around. tools/prep_head.py places the crown at 0.128 m and the chin at
    // about -0.079 m, so half the head height below the crown is where the ears are.
    center_ = Vec3(0.0f, crown - 0.100f, 0.0f);
    radius_ = 0.11f;
    float maxR = 0.0f;
    for (const Vec3& p : mesh.positions) maxR = std::max(maxR, length(p - center_));
    if (maxR > 0.0f) radius_ = maxR;

    head_ = std::move(mesh);
    head_.name = "head_scan";
    head_.materialIds.assign(head_.triangleCount(), 0);

    // ---- close the bust ---------------------------------------------------------------
    // The scan is cut through the chest and the cut is not a clean loop: the boundary is a
    // ragged fan, so from the front the silhouette has notches straight through to the
    // background and the bust looks torn in every render. Hanging a short skirt off each
    // boundary edge near the bottom closes those notches and reads as the mount the bust
    // stands on. It is a presentation fix, not a hair one, but it is in every image.
    {
        std::map<std::pair<uint32_t, uint32_t>, int> edgeUse;
        auto key = [](uint32_t a, uint32_t b) {
            return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        };
        for (size_t t = 0; t + 2 < head_.indices.size(); t += 3)
            for (int k = 0; k < 3; ++k) {
                const uint32_t a = head_.indices[t + static_cast<size_t>(k)];
                const uint32_t b = head_.indices[t + static_cast<size_t>((k + 1) % 3)];
                ++edgeUse[key(a, b)];
            }
        float yLo = 1e30f, yHi = -1e30f;
        for (const Vec3& p : head_.positions) {
            yLo = std::min(yLo, p.y);
            yHi = std::max(yHi, p.y);
        }
        const float cutBand = yLo + 0.14f * (yHi - yLo);
        // Long enough to leave the frame at the framing the images use: the point of the
        // skirt is that the viewer never sees where the scan was cut, not that the bust
        // stands on something.
        // Far below the frame: a shallow cone that is cut by the bottom of the image reads
        // as a stand, whereas an apex inside the frame reads as a party hat.
        const float skirtDrop = 0.55f;
        const size_t firstNewVertex = head_.positions.size();
        std::vector<std::pair<uint32_t, uint32_t>> openEdges;
        for (const auto& e : edgeUse) {
            if (e.second != 1) continue;
            const Vec3& pa = head_.positions[e.first.first];
            const Vec3& pb = head_.positions[e.first.second];
            if (pa.y > cutBand || pb.y > cutBand) continue;
            openEdges.push_back(e.first);
        }
        // One skirt, not one per edge: a fan from the whole boundary down to a single
        // apex. Per-edge strips each inherit a slightly different UV, and forty of them
        // shade as forty vertical bands (which is exactly what the first attempt looked
        // like). A fan has one UV and one normal for every triangle, so the mount is flat.
        if (!openEdges.empty()) {
            Vec3 c(0, 0, 0);
            for (const auto& e : openEdges) c = c + head_.positions[e.first] + head_.positions[e.second];
            c = c * (1.0f / static_cast<float>(2 * openEdges.size()));
            const size_t apex = head_.positions.size();
            head_.positions.push_back(Vec3(c.x, c.y - skirtDrop, c.z));
            head_.uvs.push_back(Vec2(0.50f, 0.34f));
            head_.normals.push_back(Vec3(0.0f, 0.15f, 1.0f));
            // With a constant normal and UV, the winding no longer matters for shading, but
            // it does for back-face rejection, so emit both orientations.
            for (const auto& e : openEdges) {
                const uint32_t a = e.first, b = e.second;
                head_.indices.insert(head_.indices.end(),
                                     {a, static_cast<uint32_t>(apex), b,
                                      b, static_cast<uint32_t>(apex), a});
                head_.materialIds.push_back(0);
                head_.materialIds.push_back(0);
            }
        }
        (void)firstNewVertex;
    }

    headLow_ = head_;      // 17.7k triangles is already crowd-cheap: the hair costs 40x more
    body_.positions.clear();
    body_.normals.clear();
    body_.uvs.clear();
    body_.indices.clear();
    body_.materialIds.clear();
    body_.name = "body_scan";
    eyes_.positions.clear();   // the scan's lids are closed
    eyes_.normals.clear();
    eyes_.uvs.clear();
    eyes_.indices.clear();
    eyes_.materialIds.clear();
    eyes_.name = "eyes_scan";

    // ---- smooth the crown ---------------------------------------------------------------
    // The bust's scalp has a pinched, slightly peaked crown (a remnant of the cap it was
    // generated with). Hair does not lie on a peak: the ribbons cross it end-on, every one
    // of them catches the key light, and the result is a bright bristly lump on the top of
    // the head in every profile render. A few Laplacian iterations over the top 22 mm --
    // fully inside the hair, and the scalp there is smooth on a real head -- removes the
    // peak without touching the face.
    {
        float crownY = -1e30f;
        for (const Vec3& p : head_.positions) crownY = std::max(crownY, p.y);
        const float band = crownY - 0.022f;
        std::vector<std::vector<uint32_t>> adj(head_.positions.size());
        for (size_t t = 0; t + 2 < head_.indices.size(); t += 3)
            for (int k = 0; k < 3; ++k) {
                const uint32_t u = head_.indices[t + static_cast<size_t>(k)];
                const uint32_t v = head_.indices[t + static_cast<size_t>((k + 1) % 3)];
                adj[u].push_back(v);
                adj[v].push_back(u);
            }
        std::vector<Vec3> next = head_.positions;
        for (int iter = 0; iter < 12; ++iter) {
            for (size_t i = 0; i < head_.positions.size(); ++i) {
                if (head_.positions[i].y < band || adj[i].empty()) continue;
                Vec3 sum(0, 0, 0);
                for (uint32_t j : adj[i]) sum = sum + head_.positions[j];
                const Vec3 avg = sum * (1.0f / static_cast<float>(adj[i].size()));
                // Weight fades in towards the crown, so the band has no visible seam.
                const float wgt = smoothstepf(band, band + 0.010f, head_.positions[i].y);
                next[i] = lerp(head_.positions[i], avg, 0.55f * wgt);
            }
            head_.positions.swap(next);
        }
        head_.computeNormals();
    }

    skinAlbedo_ = std::move(asset.albedo);

    // ---- depth map -----------------------------------------------------------
    // Rasterize the bust into (theta, phi) around center_ storing the radial distance.
    // This is the exact surface: hair roots are placed with it, so they cannot float off
    // the skull or sink into it the way a fitted ellipsoid would at the temples.
    scan_ = true;
    buildScanDepthMap();

    // Per-vertex scalp factor for the shaded skin (see Mesh::scalpMask).
    head_.scalpMask.resize(head_.positions.size());
    for (size_t i = 0; i < head_.positions.size(); ++i)
        head_.scalpMask[i] = scalpMask(head_.positions[i]);
    return true;
}

void HeadModel::buildScanDepthMap() {
    const int U = 256, V = 192;          // 1.4 deg x 0.94 deg per cell
    const float kMaxPhi = 3.05f;
    depthU_ = U;
    depthV_ = V;
    depth_.assign(static_cast<size_t>(U) * V, 0.0f);

    auto put = [&](int x, int y, float r) {
        if (y < 0 || y >= V) return;
        x = ((x % U) + U) % U;
        float& slot = depth_[static_cast<size_t>(y) * U + x];
        slot = std::max(slot, r);      // outermost surface wins (an ear over the skull)
    };

    const std::vector<Vec3>& P = head_.positions;
    for (size_t t = 0; t + 2 < head_.indices.size(); t += 3) {
        float th[3], ph[3], rad[3];
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            const Vec3 d = P[head_.indices[t + k]] - center_;
            const float r = length(d);
            if (r < 1e-5f) {
                ok = false;
                break;
            }
            rad[k] = r;
            ph[k] = std::acos(clampf(d.y / r, -1.0f, 1.0f));
            th[k] = std::atan2(d.x, d.z);
        }
        if (!ok) continue;
        // Unwrap the azimuth so that a triangle crossing the +-pi seam is rasterized as
        // one contiguous piece instead of two slivers on opposite sides of the map.
        for (int k = 1; k < 3; ++k) {
            while (th[k] - th[0] > kPi) th[k] -= 2.0f * kPi;
            while (th[k] - th[0] < -kPi) th[k] += 2.0f * kPi;
        }
        if (std::max({ph[0], ph[1], ph[2]}) > kMaxPhi) continue;

        const float u[3] = {(th[0] + kPi) / (2.0f * kPi) * U,
                            (th[1] + kPi) / (2.0f * kPi) * U,
                            (th[2] + kPi) / (2.0f * kPi) * U};
        const float v[3] = {ph[0] / kMaxPhi * V, ph[1] / kMaxPhi * V, ph[2] / kMaxPhi * V};
        const float area = (u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0]);
        if (std::fabs(area) < 1e-6f) continue;
        // A triangle that still spans most of the azimuth after unwrapping crossed the
        // seam in a way that cannot be rasterised as one piece; skipping it leaves a hole
        // the dilation pass fills, whereas rasterising it would smear a distant radius
        // (the back of the head) across the crown of the map.
        if (std::max({u[0], u[1], u[2]}) - std::min({u[0], u[1], u[2]}) > 0.75f * U) continue;
        const int x0 = static_cast<int>(std::floor(std::min({u[0], u[1], u[2]})));
        const int x1 = static_cast<int>(std::ceil(std::max({u[0], u[1], u[2]})));
        const int y0 = static_cast<int>(std::floor(std::min({v[0], v[1], v[2]})));
        const int y1 = static_cast<int>(std::ceil(std::max({v[0], v[1], v[2]})));
        const float inv = 1.0f / area;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float px = static_cast<float>(x) + 0.5f;
                const float py = static_cast<float>(y) + 0.5f;
                const float b0 = ((u[1] - px) * (v[2] - py) - (u[2] - px) * (v[1] - py)) * inv;
                const float b1 = ((u[2] - px) * (v[0] - py) - (u[0] - px) * (v[2] - py)) * inv;
                const float b2 = 1.0f - b0 - b1;
                if (b0 < -0.02f || b1 < -0.02f || b2 < -0.02f) continue;
                put(x, y, b0 * rad[0] + b1 * rad[1] + b2 * rad[2]);
            }
        }
    }

    if (std::getenv("SILKCROWD_DEBUG_DEPTH")) {
        for (int y = 0; y < 8; ++y) {
            float lo = 1e9f, hi = 0.0f;
            int n = 0;
            for (int x = 0; x < U; ++x) {
                const float d = depth_[static_cast<size_t>(y) * U + x];
                if (d <= 0.0f) continue;
                lo = std::min(lo, d);
                hi = std::max(hi, d);
                ++n;
            }
            std::fprintf(stderr, "depth row %d: n=%d lo=%.4f hi=%.4f\n", y, n, n ? lo : 0.0f, hi);
        }
    }

    // Cells a triangle does not cover (the raster is coarser than the scan's triangles are
    // small) are filled from their neighbours, so the bilinear lookup never crosses a gap.
    for (int pass = 0; pass < 12; ++pass) {
        std::vector<float> next = depth_;
        for (int y = 0; y < V; ++y) {
            for (int x = 0; x < U; ++x) {
                const size_t i = static_cast<size_t>(y) * U + x;
                if (depth_[i] > 0.0f) continue;
                float sum = 0.0f;
                int n = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int yy = y + dy;
                        if (yy < 0 || yy >= V) continue;
                        const int xx = ((x + dx) % U + U) % U;
                        const float value = depth_[static_cast<size_t>(yy) * U + xx];
                        if (value > 0.0f) {
                            sum += value;
                            ++n;
                        }
                    }
                }
                if (n > 0) next[i] = sum / static_cast<float>(n);
            }
        }
        depth_.swap(next);
    }
}

float HeadModel::scanRadius(float theta, float phi) const {
    if (depthU_ == 0 || depth_.empty() || phi < 0.0f || phi > 3.0f) return 0.0f;
    float t = theta + kPi;
    t -= 2.0f * kPi * std::floor(t / (2.0f * kPi));
    const float fx = t / (2.0f * kPi) * static_cast<float>(depthU_) - 0.5f;
    const float fy = phi / 3.05f * static_cast<float>(depthV_) - 0.5f;
    const float fxi = std::floor(fx), fyi = std::floor(fy);
    const int x0raw = static_cast<int>(fxi);
    const int y0 = static_cast<int>(clampf(fyi, 0.0f, static_cast<float>(depthV_ - 1)));
    const float tx = fx - fxi;
    const float ty = saturate(fy - fyi);
    const int y1 = std::min(depthV_ - 1, y0 + 1);
    const int x0 = ((x0raw % depthU_) + depthU_) % depthU_;
    const int x1 = (x0 + 1) % depthU_;
    const auto get = [&](int x, int y) { return depth_[static_cast<size_t>(y) * depthU_ + x]; };
    const float a = get(x0, y0) * (1.0f - tx) + get(x1, y0) * tx;
    const float b = get(x0, y1) * (1.0f - tx) + get(x1, y1) * tx;
    return a * (1.0f - ty) + b * ty;
}

float HeadModel::scalpMask(const Vec3& localPos) const {
    if (!scan_) return 0.0f;
    const Vec3 d = localPos - center_;
    const float r = length(d);
    if (r < 1e-6f) return 1.0f;
    const float phi = std::acos(clampf(d.y / r, -1.0f, 1.0f));
    const float theta = std::atan2(d.x, d.z);
    // 1 above the hairline (phi smaller = closer to the crown), feathered over the degrees
    // just below it: a hairline is a soft boundary, and the scalp has to be dark a few
    // millimetres past the last rooted hair.
    return 1.0f - smoothstepf(hairlineAngle(theta) - 0.07f, hairlineAngle(theta) + 0.035f, phi);
}

float HeadModel::hairlinePhi(float theta) const {
    // Theta = 0 is the face: the forehead hairline sits at ~63 deg from the crown, and
    // the nape hairline reaches ~2.0 rad. Interpolate smoothly around the head.
    return hairlineAngle(theta);
}

HeadModel::ScalpPatch HeadModel::defaultPatch() const {
    ScalpPatch p;
    p.theta0 = -kPi;
    p.theta1 = kPi;
    // Not exactly the pole. A spherical patch whose top row collapses to a point has
    // degenerate quads there, and the quad's *layer* vertices are a long way apart (one at
    // the apex, the next at the fall line), so every strand rooted in that ring leaves the
    // scalp as a spike standing out of the silhouette. Starting 8 degrees off the crown
    // leaves a hole the width of a fingertip, which the neighbouring rows cover with their
    // own volume, and removes the ring entirely.
    p.phiMin = 0.055f;   // 6 mm: below the mesh's own strand spacing, so no bald patch shows
    p.thetaCenter = 0.0f;
    return p;
}

namespace {

// Non-uniform tessellation: rows and columns are packed where the face needs them.
// The scalp (phi < 0.9) stays coarse - it is covered by hair - while the eyes, nose and
// mouth get roughly three times the sample density of a uniform grid at the same budget.
float phiFromV(float v) {
    static const float vk[6] = {0.00f, 0.16f, 0.30f, 0.46f, 0.62f, 1.00f};
    // The first row is the pole itself (phi = 0): starting at 0.03 rad left a ~4 mm
    // hole in the middle of the crown, which showed as a bald see-through patch.
    static const float pk[6] = {0.00f, 0.55f, 0.95f, 1.42f, 1.95f, 2.85f};
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
    // Above the hairline the surface is scalp, and scalp under dark hair is dark: if a
    // hairstyle opens a seam, what shows through must not be a bald patch of skin.
    // One hairline for the whole head: this used to duplicate the curve in `hairlineAngle`,
    // and the two drifted apart the moment either was tuned.
    const float hairline = hairlineAngle(theta);
    // 1 above the hairline. Above it, what shows through a parted lock is scalp, not skin.
    m.scalp = 1.0f - smoothstepf(hairline - 0.06f, hairline + 0.04f, phi);
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

    // The root layer is evaluated *before* the face list, because the face list has to drop
    // the degenerate quads at the pole. Every column of the patch converges on the crown, so
    // the first row's quads collapse to a point: their frame is undefined (a zero-area
    // polygon has no edge direction), which produced a handful of strands with nonsense
    // tangents standing straight out of the silhouette, and their zero area also divided
    // badly in the strand-count allocation. Dropping them costs nothing -- the row below
    // covers the same square centimetre of scalp.
    std::vector<float> theta(numVerts), phi(numVerts);
    std::vector<Vec3> rootPos(numVerts);
    // Off the pole by a few millimetres, so the patch's own quads are not degenerate, and
    // the crown itself is covered by a triangle fan from a single apex vertex. A patch that
    // starts exactly on the pole collapses its top row into zero-area quads; a patch that
    // starts off the pole leaves a bald spot; the fan is the only way to have both an even
    // grid below and no hole above.
    const float kApexPhi = patch.phiMin;   // 0: the patch starts on the pole
    // `surfacePoint(theta, 0)` has to fall back to the procedural ellipsoid for a scanned
    // head (there is no radial sample exactly at the pole), and that fallback is about a
    // centimetre inside the scan's crown -- so the top row of roots sank *below* the scalp
    // and left a notch in the silhouette. Raising that one row onto the scan's own crown
    // height fixes it without touching the topology.

    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const float u = static_cast<float>(i) / (nu - 1);
            const float vj = static_cast<float>(j) / (nv - 1);
            const float th = patch.theta0 + (patch.theta1 - patch.theta0) * u;
            const float hairline = head.hairlinePhi(th) + style.hairlineDrop;
            // The first row sits a couple of millimetres off the pole rather than exactly on
            // it: `surfacePoint(theta, 0)` has to fall back to the procedural ellipsoid for a
            // scanned head (there is no radial sample exactly at the pole), and that fallback
            // is ~1 cm inside the scan's crown -- which puts the top row of roots *below* the
            // scalp and opens a notch in the silhouette. Two millimetres is inside the
            // strands' own spacing, so no hole shows.
            const float ph = patch.phiMin + (hairline - patch.phiMin) * vj;
            const size_t idx = static_cast<size_t>(j * nu + i);
            theta[idx] = th;
            phi[idx] = ph;
            rootPos[idx] = head.surfacePoint(th, ph);
        }
    }

    std::vector<HairMeshFace> faces;
    faces.reserve(static_cast<size_t>(nu) + static_cast<size_t>((nu - 1) * (nv - 1)));
    for (int j = 0; j + 1 < nv; ++j) {
        for (int i = 0; i + 1 < nu; ++i) {
            HairMeshFace f;
            f.count = 4;
            f.v[0] = static_cast<uint32_t>(j * nu + i);
            f.v[1] = static_cast<uint32_t>(j * nu + i + 1);
            f.v[2] = static_cast<uint32_t>((j + 1) * nu + i + 1);
            f.v[3] = static_cast<uint32_t>((j + 1) * nu + i);
            Vec3 c(0, 0, 0);
            for (int k = 0; k < 4; ++k) c = c + rootPos[f.v[k]];
            c = c * 0.25f;
            float area = 0.0f;
            for (int k = 0; k < 4; ++k)
                area += 0.5f * length(cross(rootPos[f.v[k]] - c, rootPos[f.v[(k + 1) % 4]] - c));
            if (area < 2e-7f) continue;   // ~0.45 mm square: a collapsed pole quad
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

    // Root layer on the scalp (positions were evaluated above, before the face list).
    for (uint32_t v = 0; v < numVerts; ++v) mesh.layerVertexMutable(0, v) = rootPos[v];
    mesh.computeScalpNormals();

    // Extrusion layers. A hair mesh's layers describe the hair *volume*: the root layer is
    // the scalp, the outer layers are the surface the hair actually has (Section 2.2).
    //
    // The path of every strand is authored in the two phases hair really goes through:
    //
    //   1. it lies on the scalp from its root until the fall line (just above the ear),
    //      keeping a few millimetres of stand-off so the straight 4-gons between layer
    //      slices do not cut inside the skull;
    //   2. past the fall line it leaves the surface and hangs, holding the silhouette it
    //      had at the fall line. That is what makes a bob a bob: the head is 16 cm wide at
    //      its widest point, so hair hanging from there is a cylinder of that width with a
    //      flat, blunt hem, and it does not wrap around the jaw and neck the way an
    //      extrusion along the scalp does.
    //
    // Both phases are parameterised by *arc length in metres along the strand*, measured
    // from the crown, so a hem depth is a real distance ("the hair is 21 cm long") rather
    // than a polar angle that means something different on every part of the head. A
    // strand rooted at the hairline therefore ends at the same depth as one rooted at the
    // crown, and the fringe needs no special case beyond being short.
    const float kArcPerPhi = 0.105f;     // metres of scalp arc per radian (r ~ 10.5 cm)
    const float kFallPhi = 1.30f;        // just above the top of the ear
    const float fallArcGlobal = kFallPhi * kArcPerPhi;
    for (int l = 1; l < L; ++l) {
        const float s = static_cast<float>(l) / static_cast<float>(L - 1);
        for (int j = 0; j < nv; ++j) {
            for (int i = 0; i < nu; ++i) {
                const size_t idx = static_cast<size_t>(j * nu + i);
                const uint32_t vid = static_cast<uint32_t>(idx);
                const float th0 = theta[idx];
                const float phi0 = phi[idx];
                const float hairline = head.hairlinePhi(th0) + style.hairlineDrop;
                const float front = clampf(std::cos(th0), 0.0f, 1.0f);   // 1 at the face
                const float back = 1.0f - front;
                const float at = std::fabs(th0);
                const float faceW = smoothstepf(0.25f, 0.85f, front);
                // Rows rooted at the front hairline become the fringe: they leave the scalp
                // at their root and fall over the forehead instead of continuing down the
                // side of the head.
                // Only the last rows before the hairline are the fringe. Widening this to
                // the last third of the forehead (what it did at first) turns the whole
                // front of the hair into a vertical sheet, i.e. a hat brim.
                const float fringeRow =
                    faceW * smoothstepf(0.885f * hairline, 0.995f * hairline, phi0) *
                    smoothstepf(1.02f, 0.80f, at);

                // Side part: the rows on the parting are shortened so a fine line of scalp
                // shows through, and the rest is combed away from it.
                const float partAt = 0.52f;
                const float partGap = smoothstepf(0.13f, 0.0f, std::fabs(at - partAt));

                // ---- 1. hem depth, in metres of strand -------------------------------
                // Three zones, one continuous field: the front is the fringe, the sides and
                // the back are the curtain, and they blend over 0.5 rad so no boundary
                // shows. (An earlier version used smoothstep(0.62, 1.02, at) for the side
                // length, which is zero at the face centre and therefore overrode the fringe
                // with the short side length -- and left the forehead bare.)
                float hemLen = 0.155f + 0.60f * style.length;
                hemLen *= 1.0f + 0.10f * back * style.backBias;
                // Side-swept fringe. `hemLen` is an arc measured from the *crown*, so a fringe
                // that hangs L past the hairline is (hairline arc + L): getting that wrong is
                // what made the fringe vanish, because 9 cm of "hem" is still above the
                // hairline and the strand never reached its hanging phase.
                // The fringe has to *reach the brow*: the hairline sits 5.5 cm above it, so
                // 1-2 cm of fringe leaves a bare forehead with a hard edge across it (which
                // is what the last render showed). 4-5 cm lands the hem on the brow.
                // There is no fringe. A fringe is a sheet of hair held over the brow, and a
                // sheet that hangs from the hairline is 2 cm off the skull (the forehead
                // recedes) and then has to be pushed back onto it -- which is precisely the
                // read of a fur hat, and no shading fixes it. A centre-parted bob instead
                // takes the whole front section and combs it *off* the face, around the
                // temple, where it hangs beside the eye and over the cheek. That is what the
                // reference wears.

                // A hem is never a machined edge. Two scales: a slow ripple for the outline,
                // and a lock-scale term, because the reference bob's hem is a row of
                // *points* -- the hair ends in locks and each lock was cut at its own length.
                // Without the lock term the best a bob can be is a bowl.
                hemLen += (0.003f + 0.003f * (1.0f - style.fall)) * std::sin(th0 * 7.0f) +
                          0.0018f * std::sin(th0 * 13.0f);
                // Side-swept fringe. The reference bob carries its fringe diagonally across
                // the brow -- short above the parting eye, long over the far temple -- so the
                // hem of the front section is a ramp in th, not a level. A level fringe cut
                // straight across the brow is the single most "helmet" thing the front of a
                // bob can do.
                const float browSweep = smoothstepf(-1.15f, 1.15f, th0);
                // Lock-scale raggedness, at the same frequencies as the lock grooves
                // (23/39/61 across the head): a fringe is cut lock by lock, and the *edge*
                // is what the eye judges. A fringe edge that is one smooth line is a cap
                // brim no matter how the strands above it are shaded.
                const float fringeRag = 0.020f * std::sin(th0 * 23.0f + phi0 * 7.0f) +
                                        0.012f * std::sin(th0 * 39.0f - phi0 * 13.0f + 1.9f) +
                                        0.007f * std::sin(th0 * 61.0f + phi0 * 21.0f + 4.1f);
                const float fringeLen = (hairline + 0.02f) * kArcPerPhi +
                                        0.016f + 0.030f * browSweep + 0.014f * style.length +
                                        fringeRag;
                const float frontBlend = 1.0f - smoothstepf(0.45f, 1.05f, at);
                hemLen = mixf(hemLen, fringeLen, frontBlend * faceW);
                // Fringe hem specifically: short, high-frequency ragged tips, so the edge over
                // the brow is a broken line of points and not the brim of a cap.
                hemLen += 0.011f * frontBlend * faceW * (1.15f - style.fall) *
                          std::sin(th0 * 167.0f + 0.7f) * std::sin(th0 * 61.0f - 2.1f);
                // The parting opens the hem too: hair either side of it is combed away...
                // (see partGap above).
                // A blunt cut is cut: the lock-scale raggedness that a curly style needs
                // (so its hem is a row of ringlet ends) is exactly wrong for a bob, whose
                // hem has to be a level line with only a faint ripple. `style.fall` is a
                // good proxy for "how cut is this silhouette" -- a bob is 0.93, an afro
                // 0.25 -- so the raggedness scales with how *unkempt* the style is.
                hemLen += (0.004f + 0.030f * style.length * (1.15f - style.fall)) *
                          (0.55f * std::sin(th0 * 23.0f + phi0 * 17.0f) +
                           0.30f * std::sin(th0 * 39.0f - phi0 * 11.0f + 1.7f) +
                           0.15f * std::sin(th0 * 61.0f + phi0 * 29.0f + 4.1f));
                // (The hem is *not* shortened at the parting. Cutting it there leaves a line
                // of strand ends standing up along the part -- stubble, which reads as a
                // bristly ridge and is the brightest thing on the crown. The parting shows
                // because the volume collapses and the hair is combed away from it, which is
                // handled below.)

                // ---- 2. where this strand leaves the scalp ---------------------------
                const float arc0 = phi0 * kArcPerPhi;
                // The fall line can never be below the hairline: hair leaves the scalp at
                // the hairline, it does not slide down the forehead first.
                const float fallArcLimit = (hairline + 0.02f) * kArcPerPhi;
                float fallArc = std::min(std::max(arc0, fallArcGlobal), fallArcLimit);
                // Layered cuts (a long style with shorter layers on top) are what stop a
                // hairstyle from reading as one solid block: the rows rooted higher up the
                // scalp fall a little sooner, which builds real depth into the volume.
                fallArc = mixf(fallArc, arc0, 0.35f * smoothstepf(0.15f, 0.75f, phi0) *
                                                       (1.0f - fringeRow) * style.fall * 0.4f);
                const float hangLen = std::max(0.012f, hemLen - fallArc);
                const float totalArc = fallArc + hangLen;

                // ---- 3. the point on the path at layer s -----------------------------
                // Strands rooted on the crown have no scalp arc to travel along, so the
                // linear layer->arc mapping drops their first layer a full layer-spacing
                // away from the apex. Between two adjacent columns that leaves a wedge-shaped
                // gap right at the top of the head -- the notch you can see straight through
                // in profile. Easing the first layers of the crown rows outwards makes the
                // surface fan smoothly from the apex instead, which is what a cowlick does.
                const float crownEase = 1.0f - smoothstepf(0.0f, 0.55f, phi0);
                const float sArc = mixf(s, s * s, crownEase);
                const float arc = arc0 + (totalArc - arc0) * sArc;
                const float phiHere = std::min(arc / kArcPerPhi, hairline + 0.55f);
                // Lateral combing accumulates towards the tips: hair leaves the part and
                // sweeps outwards, and at the nape it is combed out over the collar.
                const float driftSign = (at >= partAt ? 1.0f : -1.0f);
                const float drift = (driftSign * 0.055f * (1.0f - faceW) *
                                         smoothstepf(0.02f, 0.16f, std::fabs(at - partAt)) +
                                     (th0 >= 0.0f ? 1.0f : -1.0f) * 0.10f *
                                         smoothstepf(1.30f, 2.10f, phiHere)) *
                                    s;
                // A bob is worn like a curtain: the hair between the temple and the jaw
                // falls in front of the cheek, and only what is behind the ear goes behind
                // it. `drift` sweeps hair away from the parting; this keeps the front locks
                // on the face side of the ear.
                // Wider and stronger than a hint: in the reference the front lock hangs in
                // *front of the ear* and over the jaw, and the ear is not visible at all. The
                // band used to stop at the temple, so the side of the head was combed behind
                // the ear and the cut read as a cap.
                const float curtain = smoothstepf(0.22f, 0.62f, at) *
                                      (1.0f - smoothstepf(1.45f, 1.85f, at)) *
                                      smoothstepf(0.15f, 0.70f, s);
                // The front section is swept *around* the head, not straight down over the
                // nose: the lateral travel is proportional to how far the strand has dropped
                // past the hairline, so the roots stay on the parting and only the hanging
                // part turns the corner at the temple. Without it the centre-front strands
                // hang in front of the face (there is no fringe to hold them up any more).
                const float sideSign = (th0 >= 0.0f) ? 1.0f : -1.0f;
                // ---- front-section tuck -------------------------------------------------
                // Hair rooted anywhere in the face section cannot hang straight down: the
                // gravity drop from the front hairline passes over the nose. Real hair combed
                // from a parting travels *around* the head -- along the hairline, across the
                // temple -- and only then falls. That is one extra phase in the path, and
                // without it every style whose front section is long ends up with a sheet
                // over the eyes (a bob survives it because its fringe is short; the curly and
                // wavy presets did not).
                // Stronger where the style does not fall straight down: a curly front section has to
                // be combed right around the temple or it hangs into the eyes (which is what
                // the curly render did before this weight).
                const float tuck = faceW * (1.0f - fringeRow) * (0.45f + 0.55f * style.fall);
                const float latArc = std::max(0.0f, arc - arc0);
                const float latBudget = std::max(0.0f, 1.30f - at) * kArcPerPhi;
                const float latSpan = std::min(latArc, latBudget);
                const float latDrop = std::max(0.0f, latArc - latBudget);
                const float thTuck = th0 + sideSign * latSpan / kArcPerPhi;
                const float sweep = 0.0f;
                const float th = th0 + drift - sideSign * 0.20f * curtain * style.fall +
                                 sideSign * sweep;
                const Vec3 scalpPt = head.surfacePoint(th, std::min(phiHere, fallArc / kArcPerPhi + 0.9f));
                const Vec3 nHere = normalize(scalpPt - head.center());
                // Volume: a few millimetres of stand-off everywhere (an extrusion cuts
                // inside a convex scalp), plus the style's body, which is thickest in the
                // middle of the volume and gone at the root and at the hem.
                // The volume belongs to the *body* of the hair, not to a hat: it is thin at
                // the crown, thickest through the middle of the mass and gone at the hem,
                // and it is thicker on the sides and the back than over the forehead.
                // Volume belongs to the *lower* half of the hair. Hair at the crown lies on
                // the skull -- that is why a bob's top is a head, not a dome -- and the mass
                // only swells below the ear, where the locks leave the surface.
                const float lowerMass = smoothstepf(0.30f, 0.85f, s) *
                                        smoothstepf(0.55f, 1.35f, phi0);
                const float body = style.volume * 0.55f * lowerMass * (1.0f - 0.35f * partGap);
                // The parting is combed away to both sides, so the volume collapses towards
                // the scalp there and a fine line of it shows through; the crown lifts a
                // little, or the top is a dome and the style reads as a cap.
                const float crownLift = 0.003f * (1.0f - smoothstepf(0.0f, 0.55f, phi0)) *
                                        (1.0f - 0.6f * partGap);
                // The crown rows need the most stand-off of anywhere on the head: the front
                // sheet hangs forward and the back sheet hangs back, and where they meet at
                // the top a millimetre of clearance between them and the skull opens a notch
                // straight through the silhouette. Two extra millimetres there closes it.
                const float apexLift = 0.0045f * (1.0f - smoothstepf(0.0f, 0.30f, phi0));
                const float standoff =
                    (0.0040f + body + apexLift) * (1.0f - 0.45f * partGap) + crownLift;
                Vec3 p = scalpPt;
                if (arc > fallArc) {
                    // Hanging phase: leave the scalp at the fall line and drop. `style.fall`
                    // bends the drop from "straight out" (0, an afro) to "straight down"
                    // (1, a bob); the small outward component keeps the hair clear of the
                    // jaw and the shoulders instead of intersecting them.
                    const Vec3 anchor = head.surfacePoint(th, fallArc / kArcPerPhi);
                    const Vec3 nAnchor = normalize(anchor - head.center());
                    const float drop = arc - fallArc;
                    const float outward = drop * (0.045f + 0.32f * (1.0f - style.fall));
                    p = anchor + nAnchor * (standoff + 0.002f) + Vec3(0.0f, -drop, 0.0f) +
                        nAnchor * outward;
                    // A blunt bob turns *under* at the hem: the last few centimetres curl
                    // back towards the neck, which is what gives the silhouette its inward
                    // curve at the jaw. Kicking the tips outwards (what this did first)
                    // flares the hem and reads as a fur hat.
                    const float tip = smoothstepf(0.62f, 1.0f, s);
                    p -= nAnchor * (0.004f * tip * style.fall);
                    // The body swells below the ear and is widest a third of the way down,
                    // not at the hem.
                    // A blunt bob's wall is *straight*: it does not belly out and it does
                    // not taper. The reference cut is a curtain with a level hem, and every
                    // term that rounds it off (belly, taper, turn-under) reads as a wig.
                    const float belly = std::sin(kPi * clampf(s * 1.35f, 0.0f, 1.0f));
                    p += nAnchor * (0.18f * style.volume * belly);
                    // Taper: a bob's wall narrows as it descends, because the locks are cut
                    // to follow the shoulder line rather than to hang off the widest point of
                    // the skull. Without it the silhouette is a box.
                    p -= nAnchor * (0.006f * style.fall * smoothstepf(0.45f, 1.0f, s));
                    // Lock grooves. A hairstyle is not a smooth solid: it is a bundle of
                    // locks, and the grooves between them are what catch the light and the
                    // shadow. A radial modulation at lock scale (about 24 locks around the
                    // head, incommensurate frequencies so no two rows agree) changes the
                    // silhouette, the specular and the shadow map together, which is what
                    // makes the difference between hair and a helmet.
                    const float groove =
                        0.0042f * std::sin(th0 * 24.0f + phi0 * 9.0f) +
                        0.0022f * std::sin(th0 * 41.0f - phi0 * 15.0f + 2.3f);
                    p += nAnchor * (groove * smoothstepf(0.10f, 0.42f, s));
                    // Hair over the *front* of the head does not hang off it. The skull
                    // recedes about 2 cm under the hairline, so a sheet that leaves the
                    // hairline and drops straight down opens a 2 cm gap above the brow -- that
                    // gap is what made this style read as a winter hat. The front third is
                    // pulled back onto the head at a constant 9 mm, down to the brow; beyond
                    // that the fringe hangs clear of the face, as a fringe does.
                    // The sheet is held on the forehead only for the first couple of
                    // centimetres and then it comes *off* the skin. Held all the way to the
                    // brow it is a straight line pressed onto the skull -- a visor -- and no
                    // amount of shading rescues that read.
                    // Held on the brow for the first couple of centimetres, then released:
                    // pressed on the skull all the way down it is a visor.
                    // The cling has to fade *in* across the same rows that become the
                    // fringe, and it has to hold the same stand-off the crown does. Ramping
                    // it in over a couple of rows, or pulling the sheet to a different
                    // height than its neighbours, leaves a step in the mesh -- and a step in
                    // a hair mesh is a raised line of strand ends, i.e. a bristly ridge
                    // across the crown that no amount of shading hides.
                    const float clingW = (1.0f - smoothstepf(0.55f, 1.30f, at)) * faceW *
                                         smoothstepf(0.60f, 0.90f, phi0 / std::max(hairline, 1e-3f)) *
                                         (1.0f - smoothstepf(0.038f, 0.068f, drop));
                    if (clingW > 0.0f) {
                        const float clingPhi = std::min(hairline + 0.30f,
                                                        phi0 + drop / kArcPerPhi * 1.35f);
                        const Vec3 cling = head.surfacePoint(th, clingPhi) + nHere * standoff;
                        p = lerp(p, cling, clingW * smoothstepf(0.0f, 0.10f, drop));
                    }
                } else {
                    p = scalpPt + nHere * standoff;
                }
                // ---- the tucked path ---------------------------------------------------
                // Comb around the temple, then fall. `latDrop` is the part of the strand
                // that is left over once it has reached the temple, and that is what hangs.
                // Only the *hanging* part may be tucked. Applied with full weight from the
                // innermost layer the tuck teleports the layer beside the root up to a
                // radian sideways, which folds the mesh over itself and shows as a bright
                // crimped ridge along the crest -- and no shading hides a fold in geometry.
                const float tuckW = tuck * smoothstepf(0.10f, 0.55f, s);
                if (tuckW > 0.0f) {
                    const Vec3 base = head.surfacePoint(thTuck, phi0);
                    const Vec3 nT = normalize(base - head.center());
                    Vec3 pt = base + nT * (standoff + 0.002f) +
                              Vec3(0.0f, -latDrop, 0.0f) +
                              nT * (latDrop * (0.045f + 0.32f * (1.0f - style.fall)));
                    pt -= nT * (0.011f * style.fall * smoothstepf(0.62f, 1.0f, s));
                    p = lerp(p, pt, tuckW);
                }
                // ---- the pole row ------------------------------------------------------
                // Every column of the patch starts on the crown, so the strands rooted there
                // have no arc to travel along the scalp and their first layer sits a whole
                // layer-spacing away from the pole. Left alone they stand up as a tuft of
                // spikes -- the one artefact that no amount of shading or density hides.
                // A whorl is what really covers the crown: the strands sweep *across* the
                // top. Giving the pole row an explicit short sweep in its own meridional
                // direction produces that, and it also fills the few millimetres of scalp the
                // patch does not reach.
                if (phi0 < 1e-4f) {
                    const Vec3 top = rootPos[static_cast<size_t>(0 * nu + i)];
                    // The sweep starts a touch *beyond* the pole so the neighbouring columns'
                    // overlap the apex instead of meeting it: the crown then reads as one
                    // surface, where meeting exactly leaves a slot at the top.
                    const float spread = 0.012f + 0.030f * style.length;
                    const float rad = spread * smoothstepf(0.0f, 1.0f, s);
                    const Vec3 out(std::sin(th0), 0.0f, std::cos(th0));
                    const float phiOver = -0.09f + rad / kArcPerPhi;   // start past the apex
                    const float phiClamped = std::max(0.0f, phiOver);
                    const Vec3 onScalp = head.surfacePoint(th0, phiClamped);
                    const Vec3 over = onScalp + normalize(onScalp - head.center()) *
                                                   0.005f;
                    p = lerp(p, over, 1.0f);
                    (void)top;
                }
                mesh.layerVertexMutable(static_cast<uint32_t>(l), vid) = p;
            }
        }
    }

    // ---- the apex tuft ----------------------------------------------------------------
    // The one vertex at the pole has no column of its own, so it gets a hand-written path: a
    // short sweep that leaves the crown toward the front and lies down along the skull. That
    // is what a cowlick looks like, and it is what covers the few millimetres the fan spans.

    // Strand counts per bundle so that the model totals the requested strand count.
    const uint32_t bundles = static_cast<uint32_t>(faces.size());
    const uint32_t perBundle = std::max(1u, style.totalStrands / std::max(1u, bundles));
    // Area-weighted rather than uniform per face: see setAreaWeightedStrandCount. With a
    // uniform count the crown -- where every column of the root patch converges -- carried
    // 26 bundles of strands in a square centimetre and grew fur, while the widest faces on
    // the sides ran sparse.
    (void)perBundle;
    mesh.setAreaWeightedStrandCount(style.totalStrands);
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

    // ---- the pot's silhouette ------------------------------------------------------------
    // Figure 9 (200 Utah teapots with unique hair) shows the hair as a *mop*: it grows out
    // of the shoulder/lid of each pot, drapes over the body, and ends in a ragged flare
    // around the foot, with the lid knob, the spout and the handle still showing. So the
    // hair has to follow the pot's profile of revolution all the way down. Sampling the
    // radius at every height (body below y=0.09, lid dome above it) is what lets the mesh
    // hug the pot instead of standing off it as a sphere or a mushroom.
    struct Prof { float y, r; };
    const Prof profile[] = {
        {-0.095f, 0.030f}, {-0.075f, 0.045f}, {-0.072f, 0.070f}, {-0.055f, 0.098f},
        {-0.045f, 0.105f}, {-0.030f, 0.115f}, {-0.010f, 0.118f}, {0.010f, 0.116f},
        {0.030f, 0.112f}, {0.050f, 0.100f}, {0.060f, 0.092f}, {0.075f, 0.077f},
        {0.090f, 0.062f}, {0.101f, 0.048f}, {0.107f, 0.034f}, {0.112f, 0.012f},
    };
    const int np = static_cast<int>(sizeof(profile) / sizeof(profile[0]));
    auto potRadius = [&](float y) {
        if (y <= profile[0].y) return profile[0].r;
        if (y >= profile[np - 1].y) return profile[np - 1].r;
        for (int k = 0; k + 1 < np; ++k) {
            if (y <= profile[k + 1].y) {
                const float t = (y - profile[k].y) / (profile[k + 1].y - profile[k].y);
                return profile[k].r + (profile[k + 1].r - profile[k].r) * t;
            }
        }
        return profile[np - 1].r;
    };

    // Roots. They sit on the lid and the shoulder (v runs from just under the knob down to
    // the widest part of the body), because a mop of hair that starts on the *top* of the
    // lid has to cross the lid's edge to get down the pot, and the crossing is what folds
    // into the mushroom silhouette.
    const float yTop = 0.1115f, yRootLow = 0.045f;
    for (int j = 0; j < nv; ++j) {
        const float v = static_cast<float>(j) / (nv - 1);
        for (int i = 0; i < nu; ++i) {
            const float u = static_cast<float>(i) / (nu - 1);
            const float th0 = 2.0f * kPi * u;
            const float yRoot = yTop + (yRootLow - yTop) * v;
            const float rRoot = potRadius(yRoot);
            const size_t idx = static_cast<size_t>(j * nu + i);
            mesh.layerVertexMutable(0, static_cast<uint32_t>(idx)) =
                Vec3(rRoot * std::cos(th0), yRoot, rRoot * std::sin(th0));

            // Per-lock length: each lock of the mop was cut to its own depth, which is what
            // gives Figure 9 its ragged hem instead of a bowl.
            const float lockJit = 0.78f + 0.44f * hashToFloat(pcgHash(seed + static_cast<uint32_t>(i) * 977u));
            const float dropTotal = length * lockJit;
            for (int l = 1; l < numLayers; ++l) {
                const float s = static_cast<float>(l) / static_cast<float>(numLayers - 1);
                const float drop = dropTotal * s;
                float yHere = yRoot - drop;
                // Below the foot there is no pot left to follow: the mop flares out.
                float rSurf = potRadius(yHere);
                if (yHere < -0.075f) rSurf = potRadius(-0.075f) + (-0.075f - yHere) * 0.55f;
                // The hair stands off the porcelain further as it falls, so the mop is a
                // shell with real thickness and the pot reads as *under* it.
                const float standoff = 0.004f + 0.065f * smoothstepf(0.10f, 1.0f, s) +
                                       0.0025f * std::sin(th0 * 17.0f + 0.7f) *
                                           smoothstepf(0.2f, 1.0f, s);
                const float th = th0 + 0.22f * s;   // a slight swirl, as a mop has
                mesh.layerVertexMutable(static_cast<uint32_t>(l), static_cast<uint32_t>(idx)) =
                    Vec3((rSurf + standoff) * std::cos(th), yHere,
                         (rSurf + standoff) * std::sin(th));
            }
        }
    }
    mesh.computeScalpNormals();
    mesh.setUniformStrandCount(std::max(1u, 50000u / static_cast<uint32_t>(faces.size())));
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
        case 0: s.name = "long_straight";  s.gridU = 18; s.gridV = 14; s.numLayers = 6;
                s.length = 0.34f; s.fall = 0.92f; s.volume = 0.005f; s.backBias = 0.25f; break;
        case 1: s.name = "wavy_medium";    s.gridU = 22; s.gridV = 22; s.numLayers = 6;
                s.length = 0.26f; s.fall = 0.75f; s.volume = 0.014f; s.backBias = 0.3f; break;
        case 2: s.name = "curly_short";    s.gridU = 28; s.gridV = 24; s.numLayers = 4;
                s.length = 0.18f; s.fall = 0.55f; s.volume = 0.022f; s.backBias = 0.2f; break;
        case 3: s.name = "afro";           s.gridU = 30; s.gridV = 26; s.numLayers = 4;
                s.length = 0.16f; s.fall = 0.25f; s.volume = 0.055f; s.backBias = 0.1f; break;
        case 4: s.name = "bob";            s.gridU = 26; s.gridV = 22; s.numLayers = 6;
                // The reference bob is a collarbone-length curtain cut blunt: long enough to
                // cover the jaw and the near shoulder, nearly vertical, and with almost no
                // volume at the crown (it swells at the jaw instead).
                // (Chin length: hemLen is 0.155 + 0.60*length in metres of strand, so 0.18
                // puts the hem at the jaw. 0.30 put it on the collarbone, which is not the
                // bob in the reference.)
                s.length = 0.215f; s.fall = 0.93f; s.volume = 0.008f; s.backBias = 0.10f; break;
        case 5: s.name = "ponytail";       s.gridU = 24; s.gridV = 18; s.numLayers = 5;
                s.length = 0.36f; s.fall = 1.0f; s.volume = 0.006f; s.backBias = 0.55f; break;
        case 6: s.name = "beach_waves";    s.gridU = 34; s.gridV = 26; s.numLayers = 6;
                s.length = 0.30f; s.fall = 0.8f; s.volume = 0.016f; s.backBias = 0.35f; break;
        case 7: s.name = "curly_long";     s.gridU = 40; s.gridV = 32; s.numLayers = 6;
                s.length = 0.30f; s.fall = 0.7f; s.volume = 0.026f; s.backBias = 0.3f; break;
        case 8: s.name = "pixie";          s.gridU = 20; s.gridV = 16; s.numLayers = 3;
                s.length = 0.10f; s.fall = 0.5f; s.volume = 0.010f; s.backBias = 0.0f; break;
        default: s.name = "blonde_layered"; s.gridU = 30; s.gridV = 24; s.numLayers = 6;
                // Longer than the bob and layered: it has to read as a *different cut*, not
                // the same helmet in another colour.
                s.length = 0.24f; s.fall = 0.90f; s.volume = 0.016f; s.backBias = 0.18f; break;
    }
    // A fringe is a styling decision, and every preset here except the slicked-back ones has
    // one: `hairlineDrop` extends the hairline's *front* only, so the fringe covers the
    // forehead without the head's own hairline moving (which is what put a fringe over the
    // male scan's eyes when this was done globally).
    switch (index % 10) {
        case 0: s.hairlineDrop = 0.06f; break;   // long_straight: parted, some sweep
        case 1: s.hairlineDrop = 0.07f; break;   // wavy_medium
        case 2: s.hairlineDrop = 0.10f; break;   // curly_short
        case 3: s.hairlineDrop = 0.05f; break;   // afro: hairline stays high
        case 4: s.hairlineDrop = 0.02f; break;   // bob: fringe starts at the natural hairline
        case 5: s.hairlineDrop = 0.04f; break;   // ponytail: hair is pulled back
        case 6: s.hairlineDrop = 0.08f; break;   // beach_waves
        case 7: s.hairlineDrop = 0.08f; break;   // curly_long
        case 8: s.hairlineDrop = 0.06f; break;   // pixie
        default: s.hairlineDrop = 0.12f; break;  // platinum_bob
    }
    s.styling = StylingParams::fromPreset(index, seed);
    // Styling amplitudes live in object space: scale them with the hair length so that
    // the same parameters read correctly on short and long styles.
    s.styling.amplitudeScale = 0.055f * s.length;
    return s;
}

}  // namespace silk
