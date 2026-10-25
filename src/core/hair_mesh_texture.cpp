#include "silkcrowd/hair_mesh_texture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "silkcrowd/catmull_rom.h"

namespace silk {
namespace {

// --- float16 round trip (used to model the half-float texel formats) ---------
inline float halfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal
            uint32_t e = 127 - 15 + 1;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t floatToHalf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int32_t shift = 14 - exp;
        uint32_t m = mant >> shift;
        // round to nearest even
        if ((mant >> (shift - 1)) & 1u) m += 1u;
        return static_cast<uint16_t>(sign | m);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m += 1u;  // round to nearest
    if (m & 0x0400u) { m = 0; ++exp; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (m & 0x03FFu));
}

inline float quantize(float v, TexelFormat f) {
    if (f == TexelFormat::Float16) return halfToFloat(floatToHalf(v));
    return v;
}

inline void bilinearWeights(float coord, uint32_t dim, uint32_t* i0, uint32_t* i1, float* a) {
    float c = coord * static_cast<float>(dim) - 0.5f;
    c = clampf(c, 0.0f, static_cast<float>(dim - 1));
    const uint32_t i = static_cast<uint32_t>(std::floor(c));
    *i0 = i;
    *i1 = std::min(i + 1, dim - 1);
    *a = c - static_cast<float>(i);
}

}  // namespace

// =============================================================================
//  Texture sampling - mirrors VK_FILTER_LINEAR + VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
// =============================================================================
void Texture2D::sample(float s, float t, float* out) const {
    uint32_t x0, x1, y0, y1;
    float ax, ay;
    bilinearWeights(s, width, &x0, &x1, &ax);
    bilinearWeights(t, height, &y0, &y1, &ay);
    const float* t00 = texel(x0, y0);
    const float* t10 = texel(x1, y0);
    const float* t01 = texel(x0, y1);
    const float* t11 = texel(x1, y1);
    for (uint32_t c = 0; c < components; ++c) {
        const float bottom = t00[c] + (t10[c] - t00[c]) * ax;
        const float top = t01[c] + (t11[c] - t01[c]) * ax;
        out[c] = bottom + (top - bottom) * ay;
    }
}

void Texture3D::sample(float s, float t, float r, float* out) const {
    uint32_t x0, x1, y0, y1, z0, z1;
    float ax, ay, az;
    bilinearWeights(s, width, &x0, &x1, &ax);
    bilinearWeights(t, height, &y0, &y1, &ay);
    bilinearWeights(r, depth, &z0, &z1, &az);
    for (uint32_t c = 0; c < components; ++c) {
        const float* b00 = texel(x0, y0, z0);
        const float* b10 = texel(x1, y0, z0);
        const float* b01 = texel(x0, y1, z0);
        const float* b11 = texel(x1, y1, z0);
        const float* c00 = texel(x0, y0, z1);
        const float* c10 = texel(x1, y0, z1);
        const float* c01 = texel(x0, y1, z1);
        const float* c11 = texel(x1, y1, z1);
        const float lo0 = b00[c] + (b10[c] - b00[c]) * ax;
        const float lo1 = b01[c] + (b11[c] - b01[c]) * ax;
        const float lo = lo0 + (lo1 - lo0) * ay;
        const float hi0 = c00[c] + (c10[c] - c00[c]) * ax;
        const float hi1 = c01[c] + (c11[c] - c01[c]) * ax;
        const float hi = hi0 + (hi1 - hi0) * ay;
        out[c] = lo + (hi - lo) * az;
    }
}

// =============================================================================
//  Build
// =============================================================================
void HairMeshTextureSet::clear() {
    hairMesh = Texture3D{};
    uv = Texture2D{};
    w = Texture3D{};
    u = Texture3D{};
    v = Texture3D{};
    layout_ = RootSliceLayout{};
    numLayers_ = 0;
    numVertices_ = 0;
    faces_.clear();
}

void HairMeshTextureSet::build(const HairMesh& mesh, const StylingFrames& frames,
                               TexelFormat format, uint32_t gutter) {
    clear();
    format_ = format;
    numLayers_ = mesh.numLayers();
    numVertices_ = mesh.numVertices();
    faces_ = mesh.faces();

    const uint32_t faceCount = mesh.numFaces();
    if (faceCount == 0 || numLayers_ == 0) return;

    // Lattice for the root slice: roughly square, one 2x2 block per hair mesh face.
    layout_.blocksX = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(faceCount))));
    layout_.blocksY = (faceCount + layout_.blocksX - 1) / layout_.blocksX;
    layout_.gutter = gutter;

    const uint32_t W = layout_.width();
    const uint32_t H = layout_.height();
    const uint32_t D = hairMeshTextureDepth(numLayers_);  // Figure 3e
    const uint32_t AD = attributeTextureDepth(numLayers_);

    hairMesh.resize(W, H, D, 3);
    uv.resize(W, H, 2);
    w.resize(W, H, AD, 1);
    u.resize(W, H, AD, 3);
    v.resize(W, H, AD, 3);

    writeRootAndAttributes(mesh, frames);
    writeIntermediates(mesh);
}

void HairMeshTextureSet::writeRootAndAttributes(const HairMesh& mesh, const StylingFrames& frames) {
    // Corner -> hair mesh vertex index. Quads store v0..v3; triangles duplicate v2
    // onto the second 2x2 corner so that the block is still a full bilinear square
    // (Section 3.1: "Triangle faces are handled by duplicating one of their vertices").
    auto cornerVertex = [](const HairMeshFace& f, int corner) -> uint32_t {
        if (f.count == 4) {
            static const int map[4] = {0, 1, 2, 3};
            return f.v[map[corner & 3]];
        }
        static const int map[4] = {0, 1, 2, 2};
        return f.v[map[corner & 3]];
    };

    for (uint32_t fi = 0; fi < mesh.numFaces(); ++fi) {
        const HairMeshFace& f = mesh.face(fi);
        for (int corner = 0; corner < 4; ++corner) {
            uint32_t x, y;
            layout_.cornerTexel(fi, corner, &x, &y);
            const uint32_t vi = cornerVertex(f, corner);

            // --- uv-texture (static, 2D) ---
            float* uvp = uv.texel(x, y);
            const Vec2 tc = mesh.rootUV()[vi];
            uvp[0] = quantize(tc.x, format_);
            uvp[1] = quantize(tc.y, format_);

            for (uint32_t layer = 0; layer < numLayers_; ++layer) {
                // --- hair mesh texture: layer slice at 4*layer (Figure 3e) ---
                float* hp = hairMesh.texel(x, y, layer * kSlicesPerInterval);
                const Vec3 p = mesh.layerVertex(layer, vi);
                hp[0] = quantize(p.x, format_);
                hp[1] = quantize(p.y, format_);
                hp[2] = quantize(p.z, format_);

                // --- w-texture (static, no intermediate slices) ---
                w.texel(x, y, layer)[0] = quantize(mesh.layerW()[layer], format_);

                // --- u-/v-texture (dynamic, no intermediate slices) ---
                const size_t idx = static_cast<size_t>(layer) * frames.numVertices + vi;
                const Vec3 fu = frames.u[idx];
                const Vec3 fv = frames.v[idx];
                float* up = u.texel(x, y, layer);
                float* vp = v.texel(x, y, layer);
                up[0] = quantize(fu.x, format_);
                up[1] = quantize(fu.y, format_);
                up[2] = quantize(fu.z, format_);
                vp[0] = quantize(fv.x, format_);
                vp[1] = quantize(fv.y, format_);
                vp[2] = quantize(fv.z, format_);
            }
        }
    }
}

void HairMeshTextureSet::writeIntermediates(const HairMesh& mesh) {
    // Figure 3e: between layer slice 4L and 4(L+1) we store the control points of the
    // two quadratic Beziers that replace the cubic Bezier of that Catmull-Rom segment.
    auto cornerVertex = [](const HairMeshFace& f, int corner) -> uint32_t {
        if (f.count == 4) {
            static const int map[4] = {0, 1, 2, 3};
            return f.v[map[corner & 3]];
        }
        static const int map[4] = {0, 1, 2, 2};
        return f.v[map[corner & 3]];
    };

    std::vector<Vec3> curve;
    std::vector<float> knots;
    for (uint32_t fi = 0; fi < mesh.numFaces(); ++fi) {
        const HairMeshFace& f = mesh.face(fi);
        for (int corner = 0; corner < 4; ++corner) {
            const uint32_t vi = cornerVertex(f, corner);
            curve = mesh.vertexCurve(vi);          // root -> tip through all layers
            knots = centripetalKnots(curve);       // centripetal parameterization
            uint32_t x, y;
            layout_.cornerTexel(fi, corner, &x, &y);
            for (uint32_t seg = 0; seg + 1 < curve.size(); ++seg) {
                const CubicBezier cubic = catmullRomToCubicBezier(knots, curve, static_cast<int>(seg));
                const QuadraticPair quad = cubicToTwoQuadratics(cubic);
                const uint32_t base = seg * kSlicesPerInterval;
                float* s1 = hairMesh.texel(x, y, base + 1);
                float* s2 = hairMesh.texel(x, y, base + 2);
                float* s3 = hairMesh.texel(x, y, base + 3);
                s1[0] = quantize(quad.q0plus.x, format_);
                s1[1] = quantize(quad.q0plus.y, format_);
                s1[2] = quantize(quad.q0plus.z, format_);
                s2[0] = quantize(quad.q05.x, format_);
                s2[1] = quantize(quad.q05.y, format_);
                s2[2] = quantize(quad.q05.z, format_);
                s3[0] = quantize(quad.q5minus.x, format_);
                s3[1] = quantize(quad.q5minus.y, format_);
                s3[2] = quantize(quad.q5minus.z, format_);
            }
        }
    }
}

void HairMeshTextureSet::updateDynamic(const HairMesh& mesh, const StylingFrames& frames) {
    // Table 1: only the hair mesh texture, the u-texture and the v-texture change when
    // the hair mesh deforms. uv-texture and w-texture are static.
    writeRootAndAttributes(mesh, frames);
    writeIntermediates(mesh);
}

// =============================================================================
//  Sampling / curve evaluation
// =============================================================================
Vec3 HairMeshTextureSet::sampleSlice(Vec2 st, float sliceCoord) const {
    float out[3] = {0, 0, 0};
    const float r = (sliceCoord + 0.5f) / static_cast<float>(hairMesh.depth);
    hairMesh.sample(st.x, st.y, r, out);
    return Vec3(out[0], out[1], out[2]);
}

Vec3 HairMeshTextureSet::evalCurveWithIntermediate(Vec2 st, float t, Vec3* ab, Vec3* bc,
                                                   float* uOut) const {
    const float layers = static_cast<float>(numLayers_);
    t = clampf(t, 0.0f, layers - 1.0f);
    const int seg = std::min(static_cast<int>(std::floor(t)), static_cast<int>(numLayers_) - 2);
    const float u = t - static_cast<float>(seg);
    const float base = static_cast<float>(seg * kSlicesPerInterval);

    // Two quadratic Beziers per layer interval (Figure 3e). u < 0.5 -> first quadratic,
    // u >= 0.5 -> second quadratic. Each costs 2 trilinear fetches + 1 lerp.
    float s;
    float sliceA, sliceB;
    if (u < 0.5f) {
        s = u * 2.0f;
        sliceA = base + s;
        sliceB = base + 1.0f + s;
    } else {
        s = (u - 0.5f) * 2.0f;
        sliceA = base + 2.0f + s;
        sliceB = base + 3.0f + s;
    }
    const Vec3 a = sampleSlice(st, sliceA);
    const Vec3 b = sampleSlice(st, sliceB);
    if (ab) *ab = a;
    if (bc) *bc = b;
    if (uOut) *uOut = s;
    return a + (b - a) * s;  // the single linear interpolation done in software
}

Vec3 HairMeshTextureSet::evalCurve(Vec2 st, float t) const {
    return evalCurveWithIntermediate(st, t, nullptr, nullptr, nullptr);
}

Vec2 HairMeshTextureSet::sampleUV(Vec2 st) const {
    float out[2] = {0, 0};
    uv.sample(st.x, st.y, out);
    return {out[0], out[1]};
}

float HairMeshTextureSet::sampleW(Vec2 st, float layerCoord) const {
    const float r = (layerCoord + 0.5f) / static_cast<float>(w.depth);
    float out[1] = {0};
    w.sample(st.x, st.y, r, out);
    return out[0];
}

void HairMeshTextureSet::sampleFrame(Vec2 st, float layerCoord, Vec3* uOut, Vec3* vOut) const {
    const float r = (layerCoord + 0.5f) / static_cast<float>(u.depth);
    float uo[3] = {0, 0, 0}, vo[3] = {0, 0, 0};
    u.sample(st.x, st.y, r, uo);
    v.sample(st.x, st.y, r, vo);
    if (uOut) *uOut = Vec3(uo[0], uo[1], uo[2]);
    if (vOut) *vOut = Vec3(vo[0], vo[1], vo[2]);
}

Mat3 HairMeshTextureSet::stylingMatrix(Vec2 st, float layerCoord) const {
    Vec3 uu, vv;
    sampleFrame(st, layerCoord, &uu, &vv);
    // The u-/v-textures are interpolated bilinearly, which shortens the vectors between
    // two vertices whose frames differ; the styling matrix is built from the
    // *normalised* directions (u_hat, v_hat) as in Eq. 1 and Eq. 4.
    uu = normalize(uu);
    if (length2(vv) < 1e-12f) vv = Vec3(0, 1, 0);
    // Re-orthogonalise v against u so that Mp is a proper frame even though Eq. 1 does
    // not guarantee u_hat . v_hat == 0 (the skew is preserved for the styling basis).
    const Vec3 vOrtho = vv - uu * dot(vv, uu);
    const Vec3 vHat = length2(vOrtho) > 1e-12f ? normalize(vOrtho) : normalize(vv);
    // Eq. 4: w_hat = (u_hat x v_hat) / ||u_hat x v_hat||
    const Vec3 ww = normalize(cross(uu, vHat));
    return Mat3(uu, vHat, ww);
}

// =============================================================================
//  Accounting
// =============================================================================
HairMeshTextureSet::SizeReport HairMeshTextureSet::sizeReport(TexelFormat format) const {
    const uint64_t bpc = bytesPerComponent(format);
    SizeReport r;
    r.hairMeshBytes = static_cast<uint64_t>(hairMesh.width) * hairMesh.height * hairMesh.depth * 3 * bpc;
    r.uvBytes = static_cast<uint64_t>(uv.width) * uv.height * 2 * bpc;
    r.wBytes = static_cast<uint64_t>(w.width) * w.height * w.depth * 1 * bpc;
    r.uBytes = static_cast<uint64_t>(u.width) * u.height * u.depth * 3 * bpc;
    r.vBytes = static_cast<uint64_t>(v.width) * v.height * v.depth * 3 * bpc;
    return r;
}

// =============================================================================
//  Validation
// =============================================================================
Vec3 HairMeshTextureSet::texelPosition(uint32_t face, int corner, uint32_t slice) const {
    uint32_t x, y;
    layout_.cornerTexel(face, corner, &x, &y);
    const float* p = hairMesh.texel(x, y, std::min(slice, hairMesh.depth - 1));
    return {p[0], p[1], p[2]};
}

std::string HairMeshTextureSet::LayoutCheck::toString() const {
    std::ostringstream os;
    os << "Figure 3e depth (1 + (L-1)*4): " << (depthMatchesFigure3e ? "OK" : "FAIL") << "\n"
       << "attribute texture depths (no intermediates): " << (attributeDepthsCorrect ? "OK" : "FAIL") << "\n"
       << "root slice == scalp vertices: " << (rootSliceMatchesMesh ? "OK" : "FAIL") << "\n"
       << "uv-texture == root uv: " << (uvTextureMatchesMesh ? "OK" : "FAIL") << "\n"
       << "w-texture monotonic: " << (wTextureMonotonic ? "OK" : "FAIL") << "\n"
       << "intermediate slices == Bezier control points: "
       << (intermediateSlicesConsistent ? "OK" : "FAIL")
       << "  (max error " << maxMidpointError << ")\n"
       << "texture eval reproduces quadratic Beziers: max error "
       << maxCurveReprojectionError << "\n"
       << "texture eval vs exact Catmull-Rom: max error " << maxCatmullRomError;
    return os.str();
}

HairMeshTextureSet::LayoutCheck HairMeshTextureSet::validate(const HairMesh& mesh,
                                                             TexelFormat format) const {
    LayoutCheck c;
    const uint32_t L = numLayers_;
    c.depthMatchesFigure3e = (hairMesh.depth == hairMeshTextureDepth(L));

    const bool tex2dOk = (uv.width > 0 && uv.height > 0 && uv.components == 2);
    c.attributeDepthsCorrect = tex2dOk && w.depth == attributeTextureDepth(L) &&
                               u.depth == attributeTextureDepth(L) &&
                               v.depth == attributeTextureDepth(L);

    // Root slice + uv texture vs. the mesh.
    bool rootOk = true, uvOk = true;
    auto cornerVertex = [](const HairMeshFace& f, int corner) -> uint32_t {
        if (f.count == 4) {
            static const int map[4] = {0, 1, 2, 3};
            return f.v[map[corner & 3]];
        }
        static const int map[4] = {0, 1, 2, 2};
        return f.v[map[corner & 3]];
    };
    const float tol = (format == TexelFormat::Float16) ? 2e-3f : 1e-6f;
    for (uint32_t fi = 0; fi < mesh.numFaces(); ++fi) {
        const HairMeshFace& f = mesh.face(fi);
        for (int corner = 0; corner < 4; ++corner) {
            const uint32_t vi = cornerVertex(f, corner);
            uint32_t x, y;
            layout_.cornerTexel(fi, corner, &x, &y);
            const Vec3 p = texelPosition(fi, corner, 0);
            const Vec3 e = mesh.rootVertex(vi);
            if (length(p - e) > tol * (1.0f + length(e))) rootOk = false;
            const float* uvp = uv.texel(x, y);
            const Vec2 tc = mesh.rootUV()[vi];
            if (std::fabs(uvp[0] - tc.x) > tol || std::fabs(uvp[1] - tc.y) > tol) uvOk = false;
        }
    }
    c.rootSliceMatchesMesh = rootOk;
    c.uvTextureMatchesMesh = uvOk;

    // w monotonic from root to tip (Section 3.4).
    bool wOk = true;
    for (uint32_t l = 1; l < L; ++l)
        if (!(mesh.layerW()[l] > mesh.layerW()[l - 1])) wOk = false;
    wOk = wOk && std::fabs(mesh.layerW()[0]) < 1e-6f;
    c.wTextureMonotonic = wOk;

    // Intermediate slices must hold q0+ = (p0+b0+)/2, q05 = C(0.5), q5- = (b5-+p5)/2.
    float maxErr = 0.0f;
    std::vector<Vec3> curve;
    std::vector<float> knots;
    for (uint32_t fi = 0; fi < mesh.numFaces(); ++fi) {
        const HairMeshFace& f = mesh.face(fi);
        for (int corner = 0; corner < 4; ++corner) {
            const uint32_t vi = cornerVertex(f, corner);
            curve = mesh.vertexCurve(vi);
            knots = centripetalKnots(curve);
            for (uint32_t seg = 0; seg + 1 < curve.size(); ++seg) {
                const CubicBezier cb = catmullRomToCubicBezier(knots, curve, static_cast<int>(seg));
                const QuadraticPair q = cubicToTwoQuadratics(cb);
                const uint32_t base = seg * kSlicesPerInterval;
                const float e1 = length(texelPosition(fi, corner, base + 1) - q.q0plus);
                const float e2 = length(texelPosition(fi, corner, base + 2) - q.q05);
                const float e3 = length(texelPosition(fi, corner, base + 3) - q.q5minus);
                maxErr = std::max(maxErr, std::max(e1, std::max(e2, e3)));
            }
        }
    }
    c.intermediateSlicesConsistent = maxErr <= tol;
    c.maxMidpointError = maxErr;

    // Reproducibility: the 2 trilinear + 1 linear path must reproduce the quadratic
    // Beziers, and must stay close to the exact Catmull-Rom curve it approximates.
    float maxReproj = 0.0f, maxCr = 0.0f;
    const int steps = 16;
    for (uint32_t fi = 0; fi < mesh.numFaces(); fi += std::max(1u, mesh.numFaces() / 8)) {
        const HairMeshFace& f = mesh.face(fi);
        for (int corner = 0; corner < 4; ++corner) {
            const uint32_t vi = cornerVertex(f, corner);
            curve = mesh.vertexCurve(vi);
            knots = centripetalKnots(curve);
            // Sample the block exactly at this corner's texel: a bilinear fetch there
            // returns that vertex's control points with no blending.
            static const float cornerX[4] = {0, 1, 1, 0};
            static const float cornerY[4] = {0, 0, 1, 1};
            const Vec2 st = layout_.blockToSlice(fi, Vec2(cornerX[corner & 3], cornerY[corner & 3]));
            for (uint32_t seg = 0; seg + 1 < curve.size(); ++seg) {
                const CubicBezier cb = catmullRomToCubicBezier(knots, curve, static_cast<int>(seg));
                const QuadraticPair q = cubicToTwoQuadratics(cb);
                for (int k = 0; k <= steps; ++k) {
                    const float u = static_cast<float>(k) / steps;
                    const float t = static_cast<float>(seg) + u;
                    const Vec3 got = evalCurve(st, t);
                    maxReproj = std::max(maxReproj, length(got - q.eval(u)));
                    const Vec3 exact = catmullRomEval(knots, curve, static_cast<int>(seg), u);
                    maxCr = std::max(maxCr, length(got - exact));
                }
            }
        }
    }
    c.maxCurveReprojectionError = maxReproj;
    c.maxCatmullRomError = maxCr;
    return c;
}

}  // namespace silk
