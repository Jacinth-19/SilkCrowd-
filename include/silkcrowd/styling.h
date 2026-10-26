// Styling coordinates (paper Section 3.4, Eq. 1-4, Figures 4-5) and the procedural
// styling operations applied on top of the hair-mesh base curves.
#pragma once

#include <cstdint>
#include <vector>

#include "silkcrowd/hair_mesh.h"
#include "silkcrowd/hair_mesh_texture.h"
#include "silkcrowd/math.h"

namespace silk {

// ---------------------------------------------------------------------------
//  Eq. 1-4: consistent object-space u/v/w coordinate frame per hair mesh vertex.
//
//  For each hair mesh vertex, u_hat is the texture-space-area weighted average of the
//  linearly-transformed directions u_i of every triangle i that surrounds the vertex
//  at the same layer (Figure 5). Because the frame is defined per *vertex* and shared
//  by all incident bundles, neighbouring bundles agree along their shared edge and the
//  styling perturbations stay continuous -- this is what removes the seam that naive
//  finite differences produce (Figure 4a).
// ---------------------------------------------------------------------------
StylingFrames computeStylingFrames(const HairMesh& mesh);

// Per-face-corner frames computed by finite differences on the bilinear bundle
// (the *wrong* method of Figure 4a). Provided only so the seam can be reproduced and
// measured; never used for rendering.
StylingFrames computeFramesFiniteDifference(const HairMesh& mesh);

// Measures the discontinuity of a frame field along shared bundle edges. For the Eq. 1-4
// frames this is ~0; for finite differences it is large (Figure 4a's vertical seam).
struct FrameContinuityReport {
    float maxUVariation = 0.0f;   // max |u_a - u_b| over shared vertices
    float maxVVariation = 0.0f;
    float meanUVariation = 0.0f;
    int sharedVerticesTested = 0;
    std::string toString() const;
};
// Which frame field to measure.
enum class FrameMode {
    PerVertexTextures,   // Eq. 1-4 stored in the u-/v-textures (Figure 4b)
    FiniteDifferences,   // derivatives of the bilinear bundle (Figure 4a)
};
FrameContinuityReport measureFrameContinuity(const HairMeshTextureSet& textures, FrameMode mode);

// ---------------------------------------------------------------------------
//  Procedural styling (paper Section 3.4 / supplemental, Figure 10).
//  All operations are applied *after* the base curve generation, in the local styling
//  space (u,v,w), and with object-space amplitudes so that bundle expansion/contraction
//  does not scale them (Section 3.4).
// ---------------------------------------------------------------------------
struct StylingParams {
    // clumping: strands converge towards shared clump centers
    float clumping = 0.55f;      // 0..1
    float clumpScale = 6.0f;     // number of clump cells across the uv atlas
    float clumpTipTightness = 2.0f;
    // curl: helical displacement around the base curve
    float curl = 0.35f;          // amplitude in object-space units
    float curlFrequency = 2.5f;  // turns along the strand
    float curlRadius = 0.6f;     // amount of radial (v) offset relative to tangent offset
    // Helix radius in *metres*. This is what turns a wave into a ringlet: a curl whose
    // radius is 1-2 mm is frizz, 6-10 mm is a ringlet. It is authored in object space
    // (Section 3.4) so a short pixie and a long curly style curl at the same real size.
    float curlRadiusM = 0.0f;
    // Clump tube radius in metres: strands of one clump are packed into a tube of this
    // radius instead of collapsing onto a line, which is what a lock of hair actually is.
    float clumpRadiusM = 0.0015f;
    // frizz: high-frequency random offsets
    float frizz = 0.12f;
    float frizzFrequency = 6.0f;
    // straightness: 0 = follow the hair mesh, 1 = pulled onto the bundle center line
    float straightness = 0.0f;
    // global
    float rootFade = 0.12f;      // fraction of the strand over which styling ramps in
    float lengthJitter = 0.15f;  // per-strand length variation
    float volume = 0.0f;         // outward (radial) puffing
    float randomOffset = 0.08f;  // per-strand lateral offset at the root
    float thickness = 1.0f;      // strand thickness multiplier (also driven by strand LOD)
    // Object-space amplitude of the styling displacements. The perturbations are
    // authored in styling units and multiplied by this value, so that the amplitudes
    // stay in object space and are not scaled by bundle expansion (Section 3.4).
    float amplitudeScale = 0.010f;
    Vec3 color{0.22f, 0.13f, 0.07f};
    float melanin = 0.82f;
    float redness = 0.5f;

    static StylingParams fromPreset(int preset, uint32_t seed);
};

// Per-strand state, computed once for the whole strand (cheap, a few hashes).
struct StrandStylingState {
    Vec2 uv{0.5f, 0.5f};       // root uv styling coordinate
    Vec2 clumpUV{0.5f, 0.5f};  // clump center in uv space
    Vec2 randomDir{0.0f, 0.0f};
    // Where this strand sits inside its clump's cross-section (a point in the unit disk,
    // fixed for the strand's whole length: a lock of hair is a tube of fibres).
    Vec2 clumpDisk{0.0f, 0.0f};
    float phase = 0.0f;
    float lengthScale = 1.0f;
    float thicknessScale = 1.0f;
    uint32_t seed = 0;
};

StrandStylingState makeStrandStylingState(const StylingParams& p, Vec2 uv, uint32_t strandIndex,
                                          uint32_t bundleIndex);

// Applies all enabled operations to one strand vertex.
//  baseP  : position on the hair mesh curve (already interpolated by the textures)
//  w      : w styling coordinate of this vertex (0 at root, 1 at tip)
//  Mp     : object-space styling frame [u_hat v_hat w_hat]
//  k, ns  : vertex index and number of strand segments (for vertex-level LOD morphing)
// Returns the final object-space position. The root vertex (k == 0) is returned
// unchanged: styling is never applied to root vertices (Section 3.2).
Vec3 styleStrandVertex(const StylingParams& params, const StrandStylingState& state,
                       Vec3 baseP, float w, int k, int ns, const Mat3& Mp);

// Styling applied without any coordinate frame (used to demonstrate Figure 4a).
Vec3 styleStrandVertexWithoutFrame(const StylingParams& params,
                                   const StrandStylingState& state, Vec3 baseP, float w,
                                   int k, int ns);

}  // namespace silk
