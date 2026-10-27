// =============================================================================
//  SilkCrowd - hair_common.glsl
//  GPU mirror of the paper's hair generation (Bhokare et al. 2024).
//  Every function here corresponds to a function in include/silkcrowd/*.h that is
//  covered by the CPU validation suite, so the two implementations can be compared
//  numerically (see --validate in the CLI and docs/VALIDATION.md).
//
//  Table 1 textures:
//    hairMeshTex : 3D  object-space position, depth = 1 + (numLayers-1)*4
//    uvTex       : 2D  root uv styling coordinate
//    wTex        : 3D  w styling coordinate, depth = numLayers
//    uTex        : 3D  styling direction u,    depth = numLayers
//    vTex        : 3D  styling direction v,    depth = numLayers
// =============================================================================
#ifndef SILKCROWD_HAIR_COMMON_GLSL
#define SILKCROWD_HAIR_COMMON_GLSL

#extension GL_GOOGLE_include_directive : require

// The task payload shared by hair.task and hair.mesh (GL_EXT_mesh_shader). It is declared
// once here so the two shaders cannot drift: the task shader fills it, the mesh shader reads
// it, and any mismatch in the field list is a silent, very hard to find rendering bug.
//
// Declared as a struct + `taskPayloadSharedEXT TaskPayload payload;` rather than as a
// `taskPayloadSharedEXT TaskPayload { ... } payload;` block: glslang and glslc (16.x and the
// SDK builds, which is what CI has) only accept blocks for uniform/buffer/in/out and reject
// the block form with "only uniform, buffer, in, or out blocks are supported".
struct TaskPayload {
    uint bundle;
    uint strandBase;
    uint numStrands;
    uint vertexExponent;
    float gamma;            // Eq. 12, shared by all strands of the bundle
    float thicknessScale;   // strand-LOD thickness compensation, min(1/L, N)
};

// Table 1: the five hair mesh textures. Samplers cannot be block members in Vulkan GLSL,
// so each texture gets its own binding (all in set 4, as laid out by
// createDescriptorLayouts()). The sampler objects travel with the descriptors.
layout(set = 4, binding = 0) uniform sampler3D hairMeshTex;  // 3D, static
layout(set = 4, binding = 1) uniform sampler3D wTex;         // 3D, static
layout(set = 4, binding = 2) uniform sampler3D uTex;         // 3D, dynamic
layout(set = 4, binding = 3) uniform sampler3D vTex;         // 3D, dynamic
layout(set = 4, binding = 4) uniform sampler2D uvTex;        // 2D, dynamic

// Per-hair-model constants (one descriptor per model; no instancing of hair data).
layout(std140, set = 1, binding = 0) uniform HairModel {
    mat4 model;             // object -> world
    mat4 modelViewProj;     // object -> clip
    mat4 modelInv;          // world -> object (for object-space ribbon widening)
    mat4 lightModelViewProj; // object -> light clip (shadow pass)
    vec4 textureSize;       // hair mesh texture width, height, depth, numLayers
    vec4 blocks;            // blocksX, blocksY, gutter, numFaces
    vec4 cameraPos;         // world-space camera position
    vec4 viewportSize;      // width, height, 1/width, 1/height
    vec4 lodParams;         // alphaStrand, alphaVertex, lambda, lodBias
    vec4 lodScreen;         // vertical resolution, tan(fov), 0, 0
    vec4 styling0;          // clumping, clumpScale, clumpTipTightness, curl
    vec4 styling1;          // curlFrequency, curlRadius, frizz, frizzFrequency
    vec4 styling2;          // straightness, rootFade, lengthJitter, volume
    vec4 styling3;          // randomOffset, thickness, amplitudeScale, seed
    vec4 styling4;          // clumpRadiusM, curlRadiusM (object-space, metres), 0, 0
    vec4 hairColor;         // rgb + melanin
    vec4 hairParams;        // strandSeed, strandsPerBundle, maxStrandsPerBundle, flags
} model;

#define SC_SLICES_PER_INTERVAL 4u   // Figure 3e: layer slice + 3 intermediates

// ---------------------------------------------------------------------------
//  Texture-space helpers
// ---------------------------------------------------------------------------
// Maps face-local block coordinates (a,b) in [0,1]^2 to the 2D position on the root
// slice that a bilinear fetch returns the bilinear combination of the face corners for.
vec2 blockToSlice(uint face, vec2 ab) {
    uint bx = face % uint(model.blocks.x);
    uint by = face / uint(model.blocks.x);
    float gutter = model.blocks.z;
    float x = float(bx) * (2.0 + gutter);
    float y = float(by) * (2.0 + gutter);
    return vec2((x + 0.5 + clamp(ab.x, 0.0, 1.0)) / model.textureSize.x,
                (y + 0.5 + clamp(ab.y, 0.0, 1.0)) / model.textureSize.y);
}

// Trilinear fetch of the hair mesh texture at a fractional slice coordinate.
vec3 sampleSlice(vec2 st, float sliceCoord) {
    float r = (sliceCoord + 0.5) / model.textureSize.z;
    return texture(hairMeshTex, vec3(st, r)).xyz;
}

// Figure 3e: two trilinear fetches + one linear interpolation (de Casteljau).
// t is the layer parameter in [0, numLayers-1]; the three intermediate slices between
// two layer slices hold the control points of two quadratic Beziers.
vec3 evalCurve(vec2 st, float t) {
    float layers = model.textureSize.w;
    t = clamp(t, 0.0, layers - 1.0);
    uint seg = min(uint(floor(t)), uint(layers) - 2u);
    float u = t - float(seg);
    float base = float(seg * SC_SLICES_PER_INTERVAL);
    float s, sliceA, sliceB;
    if (u < 0.5) {
        s = u * 2.0;
        sliceA = base + s;
        sliceB = base + 1.0 + s;
    } else {
        s = (u - 0.5) * 2.0;
        sliceA = base + 2.0 + s;
        sliceB = base + 3.0 + s;
    }
    vec3 a = sampleSlice(st, sliceA);
    vec3 b = sampleSlice(st, sliceB);
    return mix(a, b, s);
}

vec2 sampleUV(vec2 st) { return texture(uvTex, st).xy; }
float sampleW(vec2 st, float layerCoord) {
    float r = (layerCoord + 0.5) / model.textureSize.w;
    return texture(wTex, vec3(st, r)).x;
}
void sampleFrame(vec2 st, float layerCoord, out vec3 uHat, out vec3 vHat) {
    float r = (layerCoord + 0.5) / model.textureSize.w;
    uHat = texture(uTex, vec3(st, r)).xyz;
    vHat = texture(vTex, vec3(st, r)).xyz;
}

// Mp = [u_hat v_hat w_hat]; Eq. 4 with a re-orthogonalised v (Eq. 1 does not require
// u_hat and v_hat to be perpendicular, the skew is a property of the styling basis).
mat3 stylingMatrix(vec2 st, float layerCoord) {
    vec3 u, v;
    sampleFrame(st, layerCoord, u, v);
    u = normalize(u + vec3(1e-9));
    vec3 vOrtho = v - u * dot(v, u);
    if (dot(vOrtho, vOrtho) < 1e-12) vOrtho = normalize(cross(u, vec3(0.0, 1.0, 0.0)) + vec3(1e-9));
    vec3 vHat = normalize(vOrtho);
    vec3 wHat = normalize(cross(u, vHat));
    return mat3(u, vHat, wHat);
}

// ---------------------------------------------------------------------------
//  Section 3.3: root placement
// ---------------------------------------------------------------------------
// Triangle samples live in the half square xi_s + xi_t < 1. The tabulated mapping
// reproduces barycentric interpolation exactly:
//     xi_Delta_s = xi_s / (1 - xi_t),  xi_Delta_t = xi_t
// (barycentrics: (1 - xi_s - xi_t, xi_s, xi_t), with xi_t the duplicated vertex).
vec2 triangleRootPosition(vec2 xi) {
    float denom = max(1.0 - xi.y, 1e-5);
    return vec2(clamp(xi.x / denom, 0.002, 0.998), clamp(xi.y, 0.002, 0.998));
}

// ---------------------------------------------------------------------------
//  Section 3.6: level of detail
// ---------------------------------------------------------------------------
// Eq. 7:  L = min(1, h R / (2 d tan(theta/2)) * alpha)
//   lodScreen.x = R (vertical resolution)
//   lodScreen.y = 2 tan(theta/2)  (the frustum height per unit distance)
float detailLevel(float h, float distance, float alpha) {
    float d = max(distance, 1e-3);
    return min(1.0, (h * model.lodScreen.x) / (d * model.lodScreen.y) * alpha);
}

// Eq. 8: N_LOD = ceil(L (N + delta)), clamped to [1, N].
uint strandCountLod(uint strandsInBundle, float delta, float L) {
    if (L <= 0.0) return 0u;
    float v = ceil(min(L, 1.0) * (float(strandsInBundle) + clamp(delta, 0.0, 0.9999)));
    return min(strandsInBundle, max(1u, uint(v)));
}

// Eq. 9: e = max(0, ceil(log2(L * ns)))
uint vertexLodExponent(float L, uint nsMaxSegments) {
    if (L <= 0.0) return 0u;
    return clamp(uint(ceil(log2(max(L * float(nsMaxSegments), 1e-6)))), 0u, uint(log2(float(nsMaxSegments))));
}
// Eq. 10: e~ = max(0, ceil(log2(L * ns) - lambda))
uint previousVertexLodExponent(float L, uint nsMaxSegments, float lambda) {
    if (L <= 0.0) return 0u;
    return clamp(uint(ceil(log2(max(L * float(nsMaxSegments), 1e-6)) - lambda)), 0u,
                 uint(log2(float(nsMaxSegments))));
}
// Eq. 12: gamma = 1 - (log2(L ns) - e~) / lambda
float vertexMorphGamma(float L, uint nsMaxSegments, float lambda) {
    float x = log2(max(L * float(nsMaxSegments), 1e-6));
    uint e = vertexLodExponent(L, nsMaxSegments);
    uint ePrev = previousVertexLodExponent(L, nsMaxSegments, lambda);
    if (e <= ePrev || lambda <= 0.0) return 0.0;
    return clamp(1.0 - (x - float(ePrev)) / lambda, 0.0, 1.0);
}

// Per-bundle random increment delta_i of Eq. 8.
float bundleDelta(uint bundle, uint seed) {
    uint h = bundle * 2654435761u + seed * 40503u + 17u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return float(h >> 8) * (1.0 / 16777216.0) * 0.9999;
}

// ---------------------------------------------------------------------------
//  Section 3.4 / supplemental: procedural styling, in the local frame
// ---------------------------------------------------------------------------
float scHash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

// Per-strand state that is computed once per strand (a handful of hashes).
struct StrandState {
    vec2 uv;          // root uv styling coordinate
    vec2 clumpUV;     // clump centre in uv space
    vec2 randomDir;
    vec2 clumpDisk;   // where the strand sits inside its clump's cross-section (unit disk)
    float phase;
    float flyaway;    // 0 = none, 1 = the furthest out of the 8 % that get one
    float flyawayAngle;
    float thicknessScale;
    uint seed;
};

StrandState makeStrandState(vec2 uv, uint strandIndex, uint bundle) {
    StrandState s;
    s.uv = uv;
    uint seed = strandIndex * 1973u + bundle * 9277u + 7u;
    seed ^= seed >> 16; seed *= 0x7feb352du; seed ^= seed >> 15;
    s.seed = seed;
    float r0 = float((seed >> 8) & 0xFFFFu) * (1.0 / 65536.0);
    float r1 = float((seed >> 16) & 0xFFFFu) * (1.0 / 65536.0);
    s.randomDir = vec2(cos(r0 * 6.28318530718), sin(r0 * 6.28318530718));
    s.thicknessScale = 0.75 + 0.5 * r1;

    // Clumping: strands converge towards a shared clump centre defined by a uv-cell grid,
    // so no per-strand or per-clump data has to be stored.
    float clumpScale = max(model.styling0.y, 1.0);
    vec2 cell = floor(uv * clumpScale);
    float jx = scHash11(cell.x * 7349.0 + cell.y * 1931.0 + 11.0);
    float jy = scHash11(cell.x * 3739.0 + cell.y * 6151.0 + 23.0);
    s.clumpUV = vec2((cell.x + 0.25 + 0.5 * jx) / clumpScale,
                     (cell.y + 0.25 + 0.5 * jy) / clumpScale);
    // The curl phase is shared by the clump (plus a small per-strand jitter), so a clump
    // winds into one ringlet instead of every strand frizzing on its own. Mirrors
    // makeStrandStylingState() in src/core/styling.cpp.
    s.phase = scHash11(cell.x * 4099.0 + cell.y * 3571.0 + 2.0) * 6.28318530718 +
              (r1 - 0.5) * 0.55;
    float diskR = sqrt(float((seed >> 4) & 0xFFFFu) * (1.0 / 65536.0));
    float diskA = float((seed >> 12) & 0xFFFFu) * (1.0 / 65536.0) * 6.28318530718;
    s.clumpDisk = vec2(cos(diskA), sin(diskA)) * diskR;
    // Flyaways: 8 % of strands get a tip excursion big enough to leave the hair mesh's own
    // volume. Derived from the seed only - nothing is stored - and mirrors
    // makeStrandStylingState() in src/core/styling.cpp.
    float fly = scHash11(float(seed & 0xFFFFu) * 0.000137 + 9.0);
    s.flyaway = fly > 0.92 ? (fly - 0.92) / 0.08 : 0.0;
    s.flyawayAngle = scHash11(float((seed >> 16) & 0xFFFFu) * 0.000211 + 10.0) * 6.28318530718;
    return s;
}

// Styling displacement in *styling space* (u,v,w) for one strand vertex. The caller
// transforms it with Mp = [u_hat v_hat w_hat] so that it follows the deforming hair mesh
// continuously (Section 3.4).
vec2 styleDisplacement(in StrandState s, float w, int k) {
    float clumping = model.styling0.x;
    float tipTightness = model.styling0.z;
    float curl = model.styling0.w;
    float curlFrequency = model.styling1.x;
    float curlRadius = model.styling1.y;
    float frizz = model.styling1.z;
    float frizzFrequency = model.styling1.w;
    float rootFade = model.styling2.y;
    float randomOffset = model.styling3.x;

    // Styling ramps in over the first `rootFade` of the strand: the root vertex stays
    // locked to the scalp so the hair never looks like a floating rug.
    float ramp = smoothstep(0.0, max(rootFade, 1e-3), w);
    vec2 d = vec2(0.0);
    if (curl != 0.0) {
        float ph = 6.28318530718 * curlFrequency * w + s.phase;
        float amp = curl * ramp;
        d += vec2(cos(ph), sin(ph)) * amp;
        d += s.randomDir * (amp * curlRadius * 0.25);
    }
    if (clumping > 0.0) {
        float weight = clumping * ramp * pow(w, tipTightness) * 0.55;
        d += (s.clumpUV - s.uv) * weight * 2.0;
    }
    if (frizz > 0.0) {
        float fk = float(k);
        float f1 = scHash11(fk * 0.731 + float(s.seed & 0xFFFFu) * 0.000137);
        float f2 = scHash11(fk * 1.117 + float((s.seed >> 8) & 0xFFFFu) * 0.000211);
        d += vec2(f1 - 0.5, f2 - 0.5) * (frizz * 2.0 * ramp * (0.6 + 0.4 * sin(fk * frizzFrequency)));
    }
    d += s.randomDir * (randomOffset * ramp);
    d *= model.styling3.z;  // object-space amplitude (not scaled by bundle expansion)
    return d;
}

// Object-space part of the styling: the clump tube (a lock of hair is a bundle of fibres,
// not a fan) and the ringlet helix, both authored in metres so that a short pixie and a
// long curly style curl at the same real size (Section 3.4).
vec3 styleDisplacementObject(in StrandState s, float w) {
    float rootFade = model.styling2.y;
    float ramp = smoothstep(0.0, max(rootFade, 1e-3), w);
    vec3 d = vec3(0.0);
    if (model.styling4.x != 0.0) {
        d += vec3(s.clumpDisk, 0.0) * (model.styling4.x * ramp);
    }
    if (s.flyaway > 0.0) {
        // In metres, like the clump tube and the ringlet helix.
        d += vec3(cos(s.flyawayAngle), sin(s.flyawayAngle), 0.0) *
             (0.034 * s.flyaway * ramp * w * w);
    }
    if (model.styling4.y != 0.0 && model.styling0.w != 0.0) {
        float ph = 6.28318530718 * model.styling1.x * w + s.phase;
        d += vec3(cos(ph), sin(ph), 0.0) *
             (model.styling4.y * clamp(model.styling0.w, 0.0, 1.2) * ramp);
    }
    return d;
}

// Section 3.5: tangents from the neighbouring vertices, with the endpoint rules
// (Eq. 5 and Eq. 6) that keep shading continuous across vertex-LOD changes.
vec3 tangentInterior(vec3 prev, vec3 next) { return next - prev; }
vec3 tangentLast(vec3 last, vec3 prev, vec3 tangentPrev) {
    vec3 q = normalize(last - prev);
    return normalize(2.0 * q * dot(normalize(tangentPrev), q) - normalize(tangentPrev));
}
vec3 tangentFirst(vec3 first, vec3 next, vec3 tangentNext) {
    vec3 q = normalize(next - first);
    return normalize(2.0 * q * dot(normalize(tangentNext), q) - normalize(tangentNext));
}

#endif  // SILKCROWD_HAIR_COMMON_GLSL
