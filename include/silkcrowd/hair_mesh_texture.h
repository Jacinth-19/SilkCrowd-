// =============================================================================
//  SilkCrowd - hair_mesh_texture.h
//  The 5-texture hair mesh representation of Bhokare et al. 2024,
//  "Real-Time Hair Rendering with Hair Meshes" (SIGGRAPH 2024).
//
//  THIS IS THE TEXTURE SYSTEM OF THE PROJECT. Hair strands are never stored in a
//  vertex/strand buffer anywhere in SilkCrowd: all strand geometry is regenerated
//  every pass from these five textures, exactly as described in Section 3.2.
//
//  Table 1 (Section 3.1) - the five textures, verbatim:
//    hair mesh texture | 3D | Object-space vertex position      (dynamic)
//    uv-texture        | 2D | Root vertex uv styling coordinate  (static)
//    w-texture         | 3D | Vertex w styling coordinate        (static)
//    u-texture         | 3D | Object-space styling direction u   (dynamic)
//    v-texture         | 3D | Object-space styling direction v   (dynamic)
//  Only hair mesh texture, u-texture and v-texture are re-uploaded when the hair
//  mesh animates; uv- and w-texture are uploaded once.
//
//  Figure 3e layout (the central optimisation):
//    Each 2x2 block of texels on a slice holds the four vertices of one scalp face,
//    so a bilinear texture fetch interpolates within a bundle.
//    Each Catmull-Rom segment is converted to a cubic Bezier (p0, b0+, b5-, p5) and
//    then approximated by TWO quadratic Beziers (Truong et al. 2020):
//         (p0 , q0+, q0,5)   and   (q0,5 , q5-, p5)
//    Three new control points per segment therefore need to live on three
//    *intermediate slices* placed between two layer slices, hence
//
//        depth = 1 + (numLayers - 1) * 4
//
//    with slice index of layer L equal to 4L and intermediates at 4L+1, 4L+2, 4L+3
//      slice 4L+0 : p_L            (layer slice)
//      slice 4L+1 : q_L+           (intermediate)
//      slice 4L+2 : q_L,L+1        (intermediate)  = curve point at the cubic midpoint
//      slice 4L+3 : q_(L+1)-       (intermediate)
//      slice 4L+4 : p_(L+1)        (layer slice)
//
//    Evaluating a strand point then costs exactly 2 trilinear fetches + 1 linear
//    interpolation (de Casteljau of a quadratic), which the texture units perform
//    for the bilinear/trilinear part in hardware.
//
//  Conventions matching hardware filtering:
//    * texel centers are at (i + 0.5) / dimension in normalized coordinates,
//    * sampling between texel i and i+1 with fraction f uses (i + f + 0.5) / dim,
//    * clamp-to-edge addressing, no mipmaps (filtering is the algorithm, not an
//      anti-aliasing convenience).
// =============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "silkcrowd/hair_mesh.h"
#include "silkcrowd/math.h"

namespace silk {

// --- layout constants (Figure 3e) -------------------------------------------
inline constexpr uint32_t kIntermediateSlicesPerInterval = 3;  // q+, q_mid, q-
inline constexpr uint32_t kSlicesPerInterval = 4;              // + the next layer slice
inline constexpr uint32_t kQuadraticSlicesPerSegment = 2;      // 2 trilinear fetches / quad

// depth = 1 + (numLayers - 1) * 4   (Figure 3e)
inline constexpr uint32_t hairMeshTextureDepth(uint32_t numLayers) {
    return numLayers <= 1u ? 1u : 1u + (numLayers - 1u) * kSlicesPerInterval;
}
// uv-/w-/u-/v-textures need no intermediate slices (Section 3.1).
inline constexpr uint32_t attributeTextureDepth(uint32_t numLayers) {
    return numLayers < 1u ? 1u : numLayers;
}

enum class TexelFormat : uint8_t {
    Float32 = 0,  // VK_FORMAT_R32G32B32_SFLOAT / R32_SFLOAT / R32G32_SFLOAT
    Float16 = 1,  // VK_FORMAT_R16G16B16_SFLOAT / R16_SFLOAT / R16G16_SFLOAT
};

inline uint32_t bytesPerComponent(TexelFormat f) { return f == TexelFormat::Float16 ? 2u : 4u; }

// --- CPU mirror of a GPU texture --------------------------------------------
// The GPU path uses VkImage + VkImageView with the same dimensions and formats;
// this struct is the exact same texel data so that the CPU reference renderer and
// the validation suite exercise the *same* sampling code the shaders mirror.
struct Texture2D {
    uint32_t width = 0, height = 0;
    uint32_t components = 0;
    std::vector<float> data;

    void resize(uint32_t w, uint32_t h, uint32_t c) {
        width = w; height = h; components = c;
        data.assign(static_cast<size_t>(w) * h * c, 0.0f);
    }
    float* texel(uint32_t x, uint32_t y) {
        return data.data() + (static_cast<size_t>(y) * width + x) * components;
    }
    const float* texel(uint32_t x, uint32_t y) const {
        return data.data() + (static_cast<size_t>(y) * width + x) * components;
    }
    // Hardware-style bilinear fetch with clamp-to-edge.
    void sample(float s, float t, float* out) const;
};

struct Texture3D {
    uint32_t width = 0, height = 0, depth = 0;
    uint32_t components = 0;
    std::vector<float> data;

    void resize(uint32_t w, uint32_t h, uint32_t d, uint32_t c) {
        width = w; height = h; depth = d; components = c;
        data.assign(static_cast<size_t>(w) * h * d * c, 0.0f);
    }
    size_t texelIndex(uint32_t x, uint32_t y, uint32_t z) const {
        return ((static_cast<size_t>(z) * height + y) * width + x);
    }
    float* texel(uint32_t x, uint32_t y, uint32_t z) { return data.data() + texelIndex(x, y, z) * components; }
    const float* texel(uint32_t x, uint32_t y, uint32_t z) const {
        return data.data() + texelIndex(x, y, z) * components;
    }
    // Hardware-style trilinear fetch with clamp-to-edge.
    void sample(float s, float t, float r, float* out) const;
};

// --- root slice layout -------------------------------------------------------
// The root layer is mapped onto a 2D lattice. For each quad face of the root layer
// the vertex positions are copied onto the texels of a 2x2 block; triangle faces
// duplicate one of their vertices and are likewise copied onto a 2x2 block
// (Section 3.1). Scalp meshes whose topology does not allow a single lattice are
// split into multiple pieces -- see `RootSliceLayout::piece` below.
struct RootSliceLayout {
    uint32_t blocksX = 0, blocksY = 0;  // lattice of 2x2 blocks
    uint32_t gutter = 0;                // optional blank texel between blocks (0 = paper layout)
    uint32_t pieceOfFace = 0;           // reserved for multi-piece scalps

    uint32_t width() const { return blocksX * 2 + (blocksX ? (blocksX - 1) * gutter : 0); }
    uint32_t height() const { return blocksY * 2 + (blocksY ? (blocksY - 1) * gutter : 0); }

    void blockOrigin(uint32_t face, uint32_t* x, uint32_t* y) const {
        const uint32_t bx = face % blocksX;
        const uint32_t by = face / blocksX;
        *x = bx * (2 + gutter);
        *y = by * (2 + gutter);
    }

    // Maps face-local block coordinates (a,b) in [0,1]^2 onto normalized (s*,t*)
    // such that a bilinear fetch returns exactly the bilinear interpolation of the
    // four vertices of that face.
    Vec2 blockToSlice(uint32_t face, Vec2 ab) const {
        uint32_t x, y;
        blockOrigin(face, &x, &y);
        const float s = (static_cast<float>(x) + 0.5f + saturate(ab.x)) / static_cast<float>(width());
        const float t = (static_cast<float>(y) + 0.5f + saturate(ab.y)) / static_cast<float>(height());
        return {s, t};
    }
    // Texel index of a face corner: corner 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1).
    void cornerTexel(uint32_t face, int corner, uint32_t* x, uint32_t* y) const {
        uint32_t ox, oy;
        blockOrigin(face, &ox, &oy);
        static const int cx[4] = {0, 1, 1, 0};
        static const int cy[4] = {0, 0, 1, 1};
        *x = ox + cx[corner & 3];
        *y = oy + cy[corner & 3];
    }
};

// --- styling frames ----------------------------------------------------------
// u/v/w per hair mesh vertex per layer (Eq. 1-4 of Section 3.4). Stored in the
// u-texture / v-texture / w-texture.
struct StylingFrames {
    uint32_t numVertices = 0, numLayers = 0;
    std::vector<Vec3> u;  // [layer * numVertices + v]
    std::vector<Vec3> v;
    std::vector<Vec3> w;
    void resize(uint32_t nv, uint32_t nl) {
        numVertices = nv; numLayers = nl;
        u.assign(static_cast<size_t>(nv) * nl, Vec3(1, 0, 0));
        v.assign(static_cast<size_t>(nv) * nl, Vec3(0, 1, 0));
        w.assign(static_cast<size_t>(nv) * nl, Vec3(0, 0, 1));
    }
};

// --- the five textures -------------------------------------------------------
class HairMeshTextureSet {
public:
    // --- dimensions ---------------------------------------------------------
    uint32_t width() const { return hairMesh.width; }
    uint32_t height() const { return hairMesh.height; }
    // depth = 1 + (numLayers - 1) * 4
    uint32_t depth() const { return hairMesh.depth; }

    uint32_t numLayers() const { return numLayers_; }
    uint32_t numFaces() const { return layout_.blocksX * layout_.blocksY; }

    const HairMeshTextureSet& textures() const { return *this; }
    const Texture3D& hairMeshTexture() const { return hairMesh; }
    const Texture2D& uvTexture() const { return uv; }
    const Texture3D& wTexture() const { return w; }
    const Texture3D& uTexture() const { return u; }
    const Texture3D& vTexture() const { return v; }

    const RootSliceLayout& layout() const { return layout_; }
    const std::vector<HairMeshFace>& meshFaces() const { return faces_; }

    // --- construction -------------------------------------------------------
    // Builds all five textures from a hair mesh and its precomputed styling frames.
    // `frames` must be produced by `computeStylingFrames()` (Eq. 1-4) so that the
    // u/v-textures hold the seam-free, per-vertex shared frame.
    void build(const HairMesh& mesh, const StylingFrames& frames, TexelFormat format = TexelFormat::Float32,
               uint32_t gutter = 0);

    // Fast path used when the hair mesh deforms: rewrites only the dynamic textures
    // (hair mesh texture, u-texture, v-texture) - Table 1.
    void updateDynamic(const HairMesh& mesh, const StylingFrames& frames);

    // --- sampling (mirrors the GLSL in shaders/hair_common.glsl) ------------
    // Trilinear fetch of the hair mesh texture: bilinear inside the bundle, linear
    // between two successive slices. `sliceCoord` is in units of slice indices,
    // i.e. p_L lives at sliceCoord == 4*L.
    Vec3 sampleSlice(Vec2 st, float sliceCoord) const;

    // Evaluates the hair curve inside the hair mesh. `t` is the layer parameter in
    // [0, numLayers-1]. Costs exactly 2 trilinear fetches + 1 lerp (Figure 3e).
    Vec3 evalCurve(Vec2 st, float t) const;

    // Same, but also returns the two de Casteljau intermediates so that callers can
    // compute the analytic derivative for free.
    Vec3 evalCurveWithIntermediate(Vec2 st, float t, Vec3* ab, Vec3* bc, float* u) const;

    Vec2 sampleUV(Vec2 st) const;

    // Single bilinear fetch on the attribute textures (no intermediate slices).
    // `layerCoord` is in [0, numLayers-1].
    float sampleW(Vec2 st, float layerCoord) const;
    void sampleFrame(Vec2 st, float layerCoord, Vec3* uOut, Vec3* vOut) const;

    // Mp* = [u_hat v_hat w_hat] at (st, layerCoord); separates styling space from
    // object space (Section 3.4).
    Mat3 stylingMatrix(Vec2 st, float layerCoord) const;

    // --- accounting ---------------------------------------------------------
    struct SizeReport {
        uint64_t hairMeshBytes = 0;
        uint64_t uvBytes = 0;
        uint64_t wBytes = 0;
        uint64_t uBytes = 0;
        uint64_t vBytes = 0;
        uint64_t total() const { return hairMeshBytes + uvBytes + wBytes + uBytes + vBytes; }
        double totalKB() const { return static_cast<double>(total()) / 1024.0; }
    };
    SizeReport sizeReport(TexelFormat format) const;

    // Structural self-check: verifies the Figure 3e invariants.
    struct LayoutCheck {
        bool depthMatchesFigure3e = false;
        bool attributeDepthsCorrect = false;
        bool rootSliceMatchesMesh = false;   // all block corners hold the right root vertices
        bool uvTextureMatchesMesh = false;
        bool wTextureMonotonic = false;      // w = 0 at root, increasing to the tip
        bool intermediateSlicesConsistent = false;  // slice 4L+2 == cubic midpoint, 4L+1/4L+3 == halves
        float maxMidpointError = 0.0f;
        float maxCurveReprojectionError = 0.0f;  // texture eval vs the quadratic Beziers
        float maxCatmullRomError = 0.0f;         // quadratic pair vs the exact Catmull-Rom
        std::string toString() const;
    };
    LayoutCheck validate(const HairMesh& mesh, TexelFormat format) const;

    void clear();

    // Internal: exposes the texel written for (face, corner, slice) for tests.
    Vec3 texelPosition(uint32_t face, int corner, uint32_t slice) const;

private:
    void writeRootAndAttributes(const HairMesh& mesh, const StylingFrames& frames);
    void writeIntermediates(const HairMesh& mesh);

    Texture3D hairMesh;  // 3D, RGB, depth = 1 + (L-1)*4, dynamic
    Texture2D uv;        // 2D, RG, static
    Texture3D w;         // 3D, R,  depth = L, static
    Texture3D u;         // 3D, RGB, depth = L, dynamic
    Texture3D v;         // 3D, RGB, depth = L, dynamic
    RootSliceLayout layout_;
    uint32_t numLayers_ = 0;
    uint32_t numVertices_ = 0;
    TexelFormat format_ = TexelFormat::Float32;
    std::vector<HairMeshFace> faces_;  // copy, needed by validate()
};

}  // namespace silk
