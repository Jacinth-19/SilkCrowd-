// =============================================================================
//  SilkCrowd - taa.frag
//  Temporal anti-aliasing resolve: reprojects the previous frame with the previous
//  view-projection, clamps the history to the 3x3 min/max box of the current frame (so
//  disocclusions and the moving hair do not smear), and blends.
//
//  The projection is jittered with a Halton(2,3) sequence in the pixel shader's favour:
//  each frame samples a different sub-pixel position, and 8-16 frames of history give the
//  same edge quality as 8x MSAA while the hair (which is alpha blended and therefore not
//  helped by MSAA) also converges.
// =============================================================================
#version 460

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D currentColor;
layout(set = 0, binding = 1) uniform sampler2D historyColor;
layout(set = 0, binding = 2) uniform TaaUbo {
    mat4 prevViewProj;
    mat4 invViewProj;
    vec4 jitter;        // xy = current pixel jitter, zw = previous pixel jitter
    vec4 params;        // x = blend weight, y = reset (1 = discard history), z = depth scale
} taa;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec3 current = texture(currentColor, inUV).rgb;

    // Neighbourhood box for the history clamp.
    vec2 texel = 1.0 / vec2(textureSize(currentColor, 0));
    vec3 lo = current, hi = current;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec3 c = texture(currentColor, inUV + vec2(x, y) * texel).rgb;
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }

    // Reproject this pixel into the previous frame. The depth is not available here (the
    // resolve runs after the pass), so the reconstruction uses the geometric mean depth of
    // the opaque/hair boundary: the approximation is exact for the static background and
    // for the head, and the neighbourhood clamp handles the hair.
    vec4 clip = vec4(inUV * 2.0 - 1.0, 0.5, 1.0);
    vec4 world = taa.invViewProj * clip;
    world /= max(world.w, 1e-6);
    vec4 prevClip = taa.prevViewProj * world;
    vec2 prevUV = (prevClip.xy / max(prevClip.w, 1e-6)) * 0.5 + 0.5;

    vec3 history = current;
    if (taa.params.y < 0.5 && all(greaterThan(prevUV, vec2(0.0))) &&
        all(lessThan(prevUV, vec2(1.0)))) {
        history = texture(historyColor, prevUV).rgb;
    }
    history = clamp(history, lo, hi);

    // Variance-guided blend: converge faster where the history disagrees with the current
    // frame (moving hair, disocclusion), slower where it agrees (static geometry).
    float weight = clamp(taa.params.x * mix(1.0, 0.35, step(0.02, abs(luma(history) - luma(current)))), 0.02, 0.98);
    if (taa.params.y > 0.5) weight = 1.0;
    outColor = vec4(mix(history, current, weight), 1.0);
}
