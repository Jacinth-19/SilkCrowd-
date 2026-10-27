// =============================================================================
//  SilkCrowd - composite.frag
//  Full-screen pass: procedural sky / infinite grid floor for missed pixels, bloom,
//  tone mapping and sRGB encoding.
// =============================================================================
#version 460

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrColor;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;
layout(set = 0, binding = 2) uniform CameraUbo {
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 sunDirection;
    vec4 sunColor;
    vec4 skyZenith;
    vec4 skyHorizon;
    vec4 groundColor;
    vec4 params;        // x = exposure, y = bloom strength, z = time, w = grid enabled
    vec4 viewport;      // width, height, 1/width, 1/height
} cam;

// Analytic sky: a Hosek-Wilkie-style gradient with a sun aureole. The parameters are
// fitted so that the zenith/horizon colours match the sun elevation used by the lighting.
vec3 skyColor(vec3 dir) {
    float h = clamp(dir.y, -1.0, 1.0);
    float t = pow(1.0 - abs(h), 1.7);
    vec3 c = mix(cam.skyZenith.rgb, cam.skyHorizon.rgb, t);
    if (h < 0.0) c = mix(cam.skyHorizon.rgb * 0.62, cam.groundColor.rgb * 1.1, clamp(-h * 2.2, 0.0, 1.0));
    float mu = clamp(dot(dir, normalize(cam.sunDirection.xyz)), 0.0, 1.0);
    float glow = pow(mu, 400.0) * 12.0 + pow(mu, 24.0) * 0.55 + pow(mu, 5.0) * 0.16;
    return c + cam.sunColor.rgb * glow;
}

// Infinite grid floor with distance fade and a soft contact shadow.
vec3 ground(vec3 hit, float t) {
    float fade = clamp(1.0 - t / 16.0, 0.0, 1.0);
    vec3 g = cam.groundColor.rgb * (0.55 + 0.45 * exp(-t * 0.25));
    vec2 coord = hit.xz / 0.25;
    vec2 f = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line = 1.0 - min(min(f.x, f.y), 1.0);
    vec2 coordBig = hit.xz / 1.0;
    vec2 fb = abs(fract(coordBig - 0.5) - 0.5) / fwidth(coordBig);
    float lineBig = 1.0 - min(min(fb.x, fb.y), 1.0);
    g = mix(g, vec3(0.40, 0.44, 0.50), line * 0.28 * fade);
    g = mix(g, vec3(0.55, 0.60, 0.66), lineBig * 0.35 * fade);
    float d = length(hit.xz);
    g *= 1.0 - 0.65 * exp(-d * d / 0.035);
    return mix(g, g, fade);
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 toSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, 1e-5), vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}

void main() {
    vec3 color = texture(hdrColor, inUV).rgb;

    if (cam.params.w > 0.5) {
        // Reconstruct the world-space view ray and shade the background analytically.
        vec2 ndc = inUV * 2.0 - 1.0;
        vec4 nearH = cam.invViewProj * vec4(ndc, 0.0, 1.0);
        vec4 farH = cam.invViewProj * vec4(ndc, 1.0, 1.0);
        vec3 nearP = nearH.xyz / nearH.w;
        vec3 farP = farH.xyz / farH.w;
        vec3 dir = normalize(farP - nearP);
        // Depth of the opaque pass decides whether a pixel was covered.
        if (texture(hdrColor, inUV).a < 0.5) {
            vec3 bg = skyColor(dir);
            if (dir.y < -1e-4) {
                float t = -cam.cameraPos.y / dir.y;
                if (t > 0.0 && t < 200.0) bg = ground(cam.cameraPos.xyz + dir * t, t);
            }
            color = bg;
        }
    }

    vec3 bloom = texture(bloomTex, inUV).rgb;
    color = mix(color, bloom, cam.params.y);
    color = aces(color * cam.params.x);
    outColor = vec4(toSrgb(color), 1.0);
}
