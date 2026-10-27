#include "silkcrowd/styling.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>

namespace silk {
namespace {

// Splits a hair mesh face into triangles, oriented so that `v` is the first corner.
// Returns the (a, b) partner indices for every triangle containing `v`.
bool faceTrianglesAround(const HairMeshFace& f, uint32_t v, uint32_t aOut[2], uint32_t bOut[2],
                         int* countOut) {
    uint32_t tri[6];
    int n = 0;
    if (f.count == 4) {
        // (v0,v1,v2) and (v0,v2,v3)
        const uint32_t t0[3] = {f.v[0], f.v[1], f.v[2]};
        const uint32_t t1[3] = {f.v[0], f.v[2], f.v[3]};
        for (int i = 0; i < 3; ++i) tri[n++] = t0[i];
        for (int i = 0; i < 3; ++i) tri[n++] = t1[i];
    } else {
        for (int i = 0; i < 3; ++i) tri[n++] = f.v[i];
    }
    const int triCount = n / 3;
    int out = 0;
    for (int t = 0; t < triCount; ++t) {
        const uint32_t* tv = &tri[t * 3];
        int local = -1;
        for (int i = 0; i < 3; ++i)
            if (tv[i] == v) local = i;
        if (local < 0) continue;
        aOut[out] = tv[(local + 1) % 3];
        bOut[out] = tv[(local + 2) % 3];
        ++out;
    }
    *countOut = out;
    return out > 0;
}

}  // namespace

// =============================================================================
//  Eq. 1-4
// =============================================================================
StylingFrames computeStylingFrames(const HairMesh& mesh) {
    StylingFrames frames;
    frames.resize(mesh.numVertices(), mesh.numLayers());

    const uint32_t nv = mesh.numVertices();
    const uint32_t nl = mesh.numLayers();

    // Accumulate per (layer, vertex).
    std::vector<Vec3> accU(static_cast<size_t>(nv) * nl, Vec3(0, 0, 0));
    std::vector<Vec3> accV(static_cast<size_t>(nv) * nl, Vec3(0, 0, 0));
    std::vector<float> accArea(static_cast<size_t>(nv) * nl, 0.0f);

    for (uint32_t fi = 0; fi < mesh.numFaces(); ++fi) {
        const HairMeshFace& f = mesh.face(fi);
        for (uint32_t c = 0; c < f.count; ++c) {
            const uint32_t v0 = f.v[c];
            for (uint32_t layer = 0; layer < nl; ++layer) {
                uint32_t aIdx[2], bIdx[2];
                int count = 0;
                faceTrianglesAround(f, v0, aIdx, bIdx, &count);
                for (int t = 0; t < count; ++t) {
                    const uint32_t v1 = aIdx[t], v2 = bIdx[t];
                    // Eq. 2: T_i = [[u1-u0, u2-u0], [v1-v0, v2-v0]] in texture space.
                    const Vec2 uv0 = mesh.rootUV()[v0];
                    const Vec2 uv1 = mesh.rootUV()[v1];
                    const Vec2 uv2 = mesh.rootUV()[v2];
                    const float a = uv1.x - uv0.x;
                    const float b = uv2.x - uv0.x;
                    const float cc = uv1.y - uv0.y;
                    const float d = uv2.y - uv0.y;
                    const float det = a * d - b * cc;
                    const float area = 0.5f * std::fabs(det);
                    if (area < 1e-12f) continue;  // degenerate in texture space
                    const float invDet = 1.0f / det;

                    // Eq. 3: [u_i v_i] = [p1-p0 p2-p0] * T_i^-1
                    const Vec3 p0 = mesh.layerVertex(layer, v0);
                    const Vec3 e1 = mesh.layerVertex(layer, v1) - p0;
                    const Vec3 e2 = mesh.layerVertex(layer, v2) - p0;
                    // T^-1 = 1/det * [[d, -b], [-c, a]]
                    const Vec3 ui = e1 * (d * invDet) + e2 * (-cc * invDet);
                    const Vec3 vi = e1 * (-b * invDet) + e2 * (a * invDet);

                    const size_t key = static_cast<size_t>(layer) * nv + v0;
                    accU[key] += ui * area;
                    accV[key] += vi * area;
                    accArea[key] += area;
                }
            }
        }
    }

    for (uint32_t layer = 0; layer < nl; ++layer) {
        for (uint32_t v = 0; v < nv; ++v) {
            const size_t key = static_cast<size_t>(layer) * nv + v;
            Vec3 u, vv;
            if (accArea[key] > 1e-12f) {
                // Eq. 1: u = sum(A_i u_i) / sum(A_i)
                u = accU[key] * (1.0f / accArea[key]);
                vv = accV[key] * (1.0f / accArea[key]);
            } else {
                u = Vec3(1, 0, 0);
                vv = Vec3(0, 1, 0);
            }
            u = normalize(u);
            vv = normalize(vv);
            // Eq. 4: w_hat = (u_hat x v_hat) / ||u_hat x v_hat||
            Vec3 w = cross(u, vv);
            if (length2(w) < 1e-12f) {
                // Degenerate (parallel) frame: pick any orthogonal direction.
                const Vec3 axis = std::fabs(u.z) < 0.9f ? Vec3(0, 0, 1) : Vec3(0, 1, 0);
                vv = normalize(cross(cross(u, axis), u));
                w = normalize(cross(u, vv));
            } else {
                w = normalize(w);
            }
            frames.u[key] = u;
            frames.v[key] = vv;
            frames.w[key] = w;
        }
    }
    return frames;
}

StylingFrames computeFramesFiniteDifference(const HairMesh& mesh) {
    // Per bundle (i.e. per root face) constant frame from the bilinear bundle itself:
    // u = dp/ds*, v = dp/dt*. Two neighbouring bundles therefore disagree along their
    // shared edge, which is the seam of Figure 4a. Corner values are written back
    // per-face so the discontinuity survives the texture round trip.
    StylingFrames frames;
    frames.resize(mesh.numVertices(), mesh.numLayers());
    // We cannot store per-corner data in a per-vertex array; emulate it by giving each
    // face its own frame at *all four* of its corners and averaging by face order. The
    // renderer only uses this to produce the comparison image, so we store the face
    // frame indexed by the face's first corner vertex.
    std::vector<Vec3> accU(static_cast<size_t>(mesh.numVertices()) * mesh.numLayers(), Vec3(0));
    std::vector<float> accN(static_cast<size_t>(mesh.numVertices()) * mesh.numLayers(), 0.0f);
    for (uint32_t fi = 0; fi < mesh.numFaces(); ++fi) {
        const HairMeshFace& f = mesh.face(fi);
        if (f.count < 3) continue;
        for (uint32_t layer = 0; layer < mesh.numLayers(); ++layer) {
            const Vec3 p0 = mesh.layerVertex(layer, f.v[0]);
            const Vec3 p1 = mesh.layerVertex(layer, f.v[1]);
            const Vec3 p2 = mesh.layerVertex(layer, f.v[2]);
            const Vec3 p3 = f.count == 4 ? mesh.layerVertex(layer, f.v[3]) : p2;
            const Vec3 du = ((p1 - p0) + (p2 - p3)) * 0.5f;
            const Vec3 dv = ((p3 - p0) + (p2 - p1)) * 0.5f;
            const Vec3 u = normalize(du);
            (void)dv;
            for (uint32_t c = 0; c < f.count; ++c) {
                const size_t key = static_cast<size_t>(layer) * mesh.numVertices() + f.v[c];
                accU[key] += u;
                accN[key] += 1.0f;
            }
            (void)p0;
        }
    }
    for (uint32_t layer = 0; layer < mesh.numLayers(); ++layer) {
        for (uint32_t v = 0; v < mesh.numVertices(); ++v) {
            const size_t key = static_cast<size_t>(layer) * mesh.numVertices() + v;
            Vec3 u = accN[key] > 0 ? normalize(accU[key]) : Vec3(1, 0, 0);
            Vec3 tmp = std::fabs(u.z) < 0.9f ? Vec3(0, 0, 1) : Vec3(0, 1, 0);
            Vec3 vv = normalize(cross(cross(u, tmp), u));
            frames.u[key] = u;
            frames.v[key] = vv;
            frames.w[key] = normalize(cross(u, vv));
        }
    }
    return frames;
}

std::string FrameContinuityReport::toString() const {
    std::ostringstream os;
    os << "shared bundle edges tested: " << sharedVerticesTested
       << ", max frame disagreement = " << maxUVariation << " rad"
       << ", mean disagreement = " << meanUVariation << " rad";
    return os.str();
}

namespace {

// Frame at a point on the root slice of one bundle's 2x2 block.
Mat3 frameAt(const HairMeshTextureSet& tex, uint32_t face, Vec2 ab, FrameMode mode) {
    const Vec2 st = tex.layout().blockToSlice(face, ab);
    if (mode == FrameMode::PerVertexTextures) {
        Vec3 u, v;
        tex.sampleFrame(st, 0.0f, &u, &v);
        u = normalize(u);
        const Vec3 vOrtho = v - u * dot(v, u);
        const Vec3 vHat = length2(vOrtho) > 1e-12f ? normalize(vOrtho) : normalize(v);
        return Mat3(u, vHat, normalize(cross(u, vHat)));
    }
    // Finite differences on the bilinear bundle (Figure 4a).
    const float e = 0.02f;
    const Vec2 stS = tex.layout().blockToSlice(face, Vec2(clampf(ab.x + e, 0.0f, 1.0f), ab.y));
    const Vec2 stSm = tex.layout().blockToSlice(face, Vec2(clampf(ab.x - e, 0.0f, 1.0f), ab.y));
    const Vec2 stT = tex.layout().blockToSlice(face, Vec2(ab.x, clampf(ab.y + e, 0.0f, 1.0f)));
    const Vec2 stTm = tex.layout().blockToSlice(face, Vec2(ab.x, clampf(ab.y - e, 0.0f, 1.0f)));
    const Vec3 ds = tex.sampleSlice(stS, 0.0f) - tex.sampleSlice(stSm, 0.0f);
    const Vec3 dt = tex.sampleSlice(stT, 0.0f) - tex.sampleSlice(stTm, 0.0f);
    const Vec3 u = normalize(ds);
    Vec3 v = dt - u * dot(dt, u);
    v = length2(v) < 1e-12f ? normalize(cross(cross(u, Vec3(0, 1, 0)), u)) : normalize(v);
    return Mat3(u, v, normalize(cross(u, v)));
}

// Block coordinates of the midpoint of the edge between two of a face's corners,
// pushed slightly towards the face interior.
Vec2 edgeMidpointInside(const HairMeshFace& f, int cornerA, int cornerB, float inset) {
    auto cornerAB = [](const HairMeshFace& ff, int c) -> Vec2 {
        // corner 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)
        static const float cx[4] = {0, 1, 1, 0};
        static const float cy[4] = {0, 0, 1, 1};
        (void)ff;
        return Vec2(cx[c & 3], cy[c & 3]);
    };
    Vec2 a = cornerAB(f, cornerA);
    Vec2 b = cornerAB(f, cornerB);
    Vec2 mid = (a + b) * 0.5f;
    // move towards the center of the block
    Vec2 toCenter = Vec2(0.5f, 0.5f) - mid;
    const float len = length(toCenter);
    if (len > 1e-6f) toCenter = toCenter * (inset / len);
    return Vec2(clampf(mid.x + toCenter.x, 0.0f, 1.0f), clampf(mid.y + toCenter.y, 0.0f, 1.0f));
}

}  // namespace

FrameContinuityReport measureFrameContinuityImpl(const HairMeshTextureSet& textures,
                                                 FrameMode mode) {
    FrameContinuityReport r;
    const std::vector<HairMeshFace>& faces = textures.meshFaces();
    double sum = 0.0;
    int count = 0;
    std::set<std::pair<uint32_t, uint32_t>> done;

    for (uint32_t fa = 0; fa < faces.size(); ++fa) {
        const HairMeshFace& a = faces[fa];
        for (int ca = 0; ca < static_cast<int>(a.count); ++ca) {
            const uint32_t va = a.v[ca];
            const uint32_t vb = a.v[(ca + 1) % a.count];
            for (uint32_t fb = fa + 1; fb < faces.size(); ++fb) {
                const HairMeshFace& b = faces[fb];
                int cb = -1;
                for (int i = 0; i < static_cast<int>(b.count); ++i) {
                    const uint32_t x = b.v[i];
                    const uint32_t y = b.v[(i + 1) % b.count];
                    if ((x == va && y == vb) || (x == vb && y == va)) cb = i;
                }
                if (cb < 0) continue;
                const auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
                if (!done.insert(key).second) continue;

                const Vec2 abA = edgeMidpointInside(a, ca, (ca + 1) % a.count, 0.0f);
                const Vec2 abB = edgeMidpointInside(b, cb, (cb + 1) % b.count, 0.0f);
                const Mat3 ma = frameAt(textures, fa, abA, mode);
                const Mat3 mb = frameAt(textures, fb, abB, mode);

                // Disagreement of the frame, measured as the angle by which the whole
                // matrix rotates one basis vector onto the other (sign-insensitive).
                const float du = std::acos(clampf(std::fabs(dot(ma.c0, mb.c0)), 0.0f, 1.0f));
                const float dv = std::acos(clampf(std::fabs(dot(ma.c1, mb.c1)), 0.0f, 1.0f));
                r.maxUVariation = std::max(r.maxUVariation, std::max(du, dv));
                sum += std::max(du, dv);
                ++count;
            }
        }
    }
    r.sharedVerticesTested = count;
    r.meanUVariation = count > 0 ? static_cast<float>(sum / count) : 0.0f;
    r.maxVVariation = r.maxUVariation;
    return r;
}

FrameContinuityReport measureFrameContinuity(const HairMeshTextureSet& textures, FrameMode mode) {
    return measureFrameContinuityImpl(textures, mode);
}

// =============================================================================
//  Procedural styling
// =============================================================================
StylingParams StylingParams::fromPreset(int preset, uint32_t seed) {
    StylingParams p;
    const float r0 = hashToFloat(pcgHash(seed * 7u + 1u));
    const float r1 = hashToFloat(pcgHash(seed * 7u + 2u));
    const float r2 = hashToFloat(pcgHash(seed * 7u + 3u));
    const float r3 = hashToFloat(pcgHash(seed * 7u + 4u));
    switch (preset % 10) {
        case 0:  // long straight
            p.clumping = 0.46f; p.curl = 0.05f; p.frizz = 0.06f; p.straightness = 0.55f;
            p.curlFrequency = 1.0f; p.color = {0.115f, 0.062f, 0.033f}; break;
        case 1:  // wavy
            p.clumping = 0.45f; p.curl = 0.30f; p.frizz = 0.08f; p.straightness = 0.10f;
            p.curlFrequency = 2.2f; p.color = {0.28f, 0.16f, 0.08f}; break;
        case 2:  // curly: honey ringlets, not fuzz
            // The second reference photograph's middle head is not a frizz: it is a set of
            // big, separate, glossy locks wound about four or five times over the length of
            // the hair. Frequency is cycles over the normalised strand length, so 9 cycles
            // (what this was) reads as wool; 4.5 reads as a curl. The lock has to be fat as
            // well, or the ringlets are thin spirals that alias into noise.
            p.clumping = 0.90f; p.clumpScale = 4.2f; p.clumpTipTightness = 0.85f;
            p.curl = 1.0f; p.curlRadiusM = 0.0120f; p.curlFrequency = 4.5f;
            p.clumpRadiusM = 0.0060f; p.frizz = 0.05f;
            p.color = {0.42f, 0.28f, 0.14f}; break;
        case 3:  // afro / coily
            p.clumping = 0.75f; p.curl = 0.85f; p.frizz = 0.35f; p.volume = 0.35f;
            p.curlFrequency = 7.0f; p.clumpScale = 10.0f; p.color = {0.10f, 0.06f, 0.04f}; break;
        case 4:  // bob: honey blonde, as in the reference photograph
            p.clumping = 0.52f; p.curl = 0.10f; p.frizz = 0.05f; p.lengthJitter = 0.10f;
            p.curlFrequency = 1.6f; p.color = {0.34f, 0.21f, 0.10f};
            p.straightness = 0.85f; break;
        case 5:  // ponytail (strong clumping)
            p.clumping = 0.88f; p.clumpScale = 3.0f; p.clumpTipTightness = 3.0f;
            p.curl = 0.12f; p.frizz = 0.07f; p.color = {0.20f, 0.11f, 0.06f}; break;
        case 6:  // blonde beach waves
            p.clumping = 0.62f; p.clumpScale = 5.0f; p.curl = 0.75f;
            p.curlRadiusM = 0.0045f; p.curlFrequency = 3.4f;
            p.clumpRadiusM = 0.0035f; p.frizz = 0.12f;
            p.color = {0.62f, 0.46f, 0.24f}; p.melanin = 0.25f; break;
        case 7:  // red curls
            p.clumping = 0.82f; p.clumpScale = 8.0f; p.clumpTipTightness = 1.2f;
            p.curl = 1.0f; p.curlRadiusM = 0.0090f; p.curlFrequency = 8.0f;
            p.clumpRadiusM = 0.0048f; p.frizz = 0.12f; p.curlFrequency = 8.0f;
            p.color = {0.45f, 0.16f, 0.06f}; p.melanin = 0.45f; p.redness = 0.85f; break;
        case 8:  // short pixie
            p.clumping = 0.30f; p.curl = 0.10f; p.frizz = 0.10f; p.lengthJitter = 0.06f;
            p.straightness = 0.25f; p.color = {0.11f, 0.07f, 0.045f}; break;
        default: // champagne blonde, longer and layered (the third head of the set)
            p.clumping = 0.40f; p.curl = 0.24f; p.frizz = 0.06f;
            p.straightness = 0.70f; p.curlRadiusM = 0.0035f; p.curlFrequency = 2.4f;
            p.color = {0.64f, 0.49f, 0.27f}; p.melanin = 0.16f; break;
    }
    // Small procedural variation so that every crowd member is unique (Figure 10).
    p.clumping = clampf(p.clumping + (r0 - 0.5f) * 0.20f, 0.0f, 1.0f);
    p.curl = clampf(p.curl * (0.7f + 0.6f * r1), 0.0f, 1.2f);
    p.frizz = clampf(p.frizz * (0.6f + 0.8f * r2), 0.0f, 0.6f);
    p.curlFrequency = clampf(p.curlFrequency * (0.8f + 0.5f * r3), 0.4f, 16.0f);
    p.lengthJitter = clampf(p.lengthJitter, 0.0f, 0.5f);
    return p;
}

StrandStylingState makeStrandStylingState(const StylingParams& p, Vec2 uv, uint32_t strandIndex,
                                          uint32_t bundleIndex) {
    StrandStylingState s;
    s.uv = uv;
    s.seed = pcgHash(strandIndex * 1973u + bundleIndex * 9277u + 7u);

    // Clump centers: shared by all strands of a bundle neighbourhood so that nearby
    // strands converge (clumping). Cell-based so that no per-strand state is needed.
    const float scale = std::max(p.clumpScale, 1.0f);
    const float cx = std::floor(uv.x * scale);
    const float cy = std::floor(uv.y * scale);
    const float jx = hashToFloat(pcgHash(static_cast<uint32_t>(cx) * 7349u +
                                         static_cast<uint32_t>(cy) * 1931u + 11u));
    const float jy = hashToFloat(pcgHash(static_cast<uint32_t>(cx) * 3739u +
                                         static_cast<uint32_t>(cy) * 6151u + 23u));
    s.clumpUV = Vec2((cx + 0.25f + 0.5f * jx) / scale, (cy + 0.25f + 0.5f * jy) / scale);

    const float a = hashToFloat(pcgHash(s.seed + 1u)) * 2.0f * kPi;
    s.randomDir = Vec2(std::cos(a), std::sin(a));
    // The curl phase is *shared by the clump* (with a small per-strand jitter), so the
    // strands of one clump wind together into a ringlet instead of each frizzing on its
    // own. This one change is the difference between curls and fuzz.
    s.phase = hashToFloat(pcgHash(static_cast<uint32_t>(cx) * 4099u +
                                 static_cast<uint32_t>(cy) * 3571u + 2u)) *
                  2.0f * kPi +
              (hashToFloat(pcgHash(s.seed + 2u)) - 0.5f) * 0.55f;
    // A fixed point inside the clump's tube, so the strands pack rather than coincide.
    const float r = std::sqrt(hashToFloat(pcgHash(s.seed + 5u)));
    const float ta = hashToFloat(pcgHash(s.seed + 6u)) * 2.0f * kPi;
    s.clumpDisk = Vec2(std::cos(ta) * r, std::sin(ta) * r);
    // Flyaways: 6 % of strands get a tip excursion large enough to leave the hair mesh's
    // own volume, which is what breaks a groomed silhouette. Everything about them comes from
    // the strand's seed, so nothing is stored and the GPU path derives the same values.
    const float fly = hashToFloat(pcgHash(s.seed + 9u));
    // Flyaways live at the hem and the nape, never at the site of the root: a fibre that
    // leaves the scalp at an angle near the crown stands straight off the silhouette, and one
    // of those on top of the head reads as a sprout. `bodyMask` in styleStrandVertex tapers
    // them near the root; this keeps them off the crown altogether.
    if (fly > 0.92f) s.flyaway = (fly - 0.92f) / 0.08f;
    s.flyawayAngle = hashToFloat(pcgHash(s.seed + 10u)) * 2.0f * kPi;
    s.lengthScale = 1.0f - p.lengthJitter * hashToFloat(pcgHash(s.seed + 3u));
    s.thicknessScale = 0.75f + 0.5f * hashToFloat(pcgHash(s.seed + 4u));
    return s;
}

Vec3 styleStrandVertex(const StylingParams& p, const StrandStylingState& s, Vec3 baseP, float w,
                       int k, int ns, const Mat3& Mp) {
    if (k == 0) return baseP;  // styling is never applied to root vertices

    // Styling ramps in over the first `rootFade` of the strand so the root stays locked
    // to the scalp and the mesh never looks like a jiggling rug.
    const float ramp = smoothstepf(0.0f, std::max(p.rootFade, 1e-3f), w);

    // Curl: a helix around the base curve, expressed in the local styling plane.
    // The ringlet helix below shares this phase, so cos/sin are evaluated once per vertex
    // instead of twice: the two trig pairs are 40-50 cycles each and this function runs
    // once per strand vertex (tens of millions of times in a crowd frame).
    Vec2 d = Vec2(0, 0);
    float cosPhase = 1.0f, sinPhase = 0.0f;
    const bool curls = p.curl != 0.0f;
    if (curls) {
        const float phase = 2.0f * kPi * p.curlFrequency * w + s.phase;
        cosPhase = std::cos(phase);
        sinPhase = std::sin(phase);
        const float amp = p.curl * ramp;
        d += Vec2(cosPhase, sinPhase) * amp;
        // A little bit of lengthwise motion keeps the curl from looking like a ribbon.
        d += s.randomDir * (amp * p.curlRadius * 0.25f);
    }

    // Clumping: converge toward the shared clump center, strongest at the tip.
    if (p.clumping > 0.0f) {
        const Vec2 toCenter = s.clumpUV - s.uv;
        // pow() with an integral small exponent is the common case (1.0, 2.0) and is exactly
        // what pow() returns for it, without the ~50-cycle library call.
        const float tip = p.clumpTipTightness == 2.0f ? w * w
                          : p.clumpTipTightness == 1.0f ? w
                                                        : std::pow(w, p.clumpTipTightness);
        const float weight = p.clumping * ramp * tip * 0.55f;
        d += toCenter * weight * 2.0f;
    }
    // The tube: every strand of a clump keeps its own place across the lock's cross-section
    // for the whole length, so the clump is a bundle of fibres and not a fan.
    Vec3 tube = Mp * Vec3(s.clumpDisk.x, s.clumpDisk.y, 0.0f) * (p.clumpRadiusM * ramp);
    // Ringlet helix, in metres, around the (already clumped) centre line.
    Vec3 helix(0, 0, 0);
    if (p.curlRadiusM > 0.0f && curls) {
        helix = Mp * Vec3(cosPhase, sinPhase, 0.0f) *
                (p.curlRadiusM * clampf(p.curl, 0.0f, 1.2f) * ramp);
    }

    // Frizz: high-frequency perturbation, per vertex.
    if (p.frizz > 0.0f) {
        const float fk = static_cast<float>(k);
        const float f1 = hash11(fk * 0.731f + static_cast<float>(s.seed & 0xFFFFu) * 0.000137f);
        const float f2 = hash11(fk * 1.117f + static_cast<float>(s.seed >> 8) * 0.000211f);
        d += Vec2(f1 - 0.5f, f2 - 0.5f) * (p.frizz * 2.0f * ramp *
                                          (0.6f + 0.4f * std::sin(fk * p.frizzFrequency)));
    }

    // Per-strand lateral offset, zero at the root.
    if (p.randomOffset != 0.0f) d += s.randomDir * (p.randomOffset * ramp);

    // Volume: puff outward (away from the bundle center) as the strand grows.
    // Styling amplitudes are converted to object space here (Section 3.4).
    Vec3 flyaway(0, 0, 0);
    if (s.flyaway > 0.0f) {
        // In metres, so a flyaway is the same real length on a pixie and on long curls.
        // Only the body and the hem get them: at the crown a flyaway sticks straight up out
        // of the parting and reads as a mohawk, which is not what a flyaway is.
        const float bodyMask = smoothstepf(0.10f, 0.45f, w);
        flyaway = Mp * Vec3(std::cos(s.flyawayAngle), std::sin(s.flyawayAngle), 0.0f) *
                  (0.020f * s.flyaway * ramp * w * bodyMask);
    }
    Vec3 offset = Mp * Vec3(d.x, d.y, 0.0f) * p.amplitudeScale + tube + helix + flyaway;
    if (p.volume != 0.0f) {
        offset = offset * (1.0f + p.volume * ramp * 0.5f);
    }
    return baseP + offset;
}

Vec3 styleStrandVertexWithoutFrame(const StylingParams& p, const StrandStylingState& s, Vec3 baseP,
                                   float w, int k, int ns) {
    // Same displacements but applied directly in object space: neighbouring bundles
    // disagree, producing the seam shown in Figure 4a.
    if (k == 0) return baseP;
    const float ramp = smoothstepf(0.0f, std::max(p.rootFade, 1e-3f), w);
    Vec3 d(0, 0, 0);
    if (p.curl != 0.0f) {
        const float phase = 2.0f * kPi * p.curlFrequency * w + s.phase;
        d += Vec3(std::cos(phase), 0.0f, std::sin(phase)) * (p.curl * ramp);
    }
    if (p.clumping > 0.0f) {
        const Vec2 toCenter = s.clumpUV - s.uv;
        const float weight = p.clumping * std::pow(w, p.clumpTipTightness) * 0.55f;
        d += Vec3(toCenter.x, 0.0f, toCenter.y) * (weight * 2.0f * ramp);
    }
    if (p.frizz != 0.0f) {
        const float fk = static_cast<float>(k);
        d += Vec3(hash11(fk * 0.731f + static_cast<float>(s.seed & 0xFFFFu) * 0.000137f) - 0.5f,
                  0.0f,
                  hash11(fk * 1.117f + static_cast<float>(s.seed >> 8) * 0.000211f) - 0.5f) *
             (p.frizz * 2.0f * ramp);
    }
    return baseP + d;
}

}  // namespace silk
