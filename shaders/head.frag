// SilkCrowd - skin / teapot shading (opaque pass).
#version 460
layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorld;
layout(location = 2) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 sunDirection;
    vec4 sunColor;
    vec4 skyZenith;
    vec4 skyHorizon;
    vec4 lodScreen;
} cam;
layout(push_constant) uniform HeadPush { mat4 mvp; mat4 model; vec4 params; } pc;  // rgb albedo, w = specular
void main() {
    vec3 n = normalize(inNormal);
    vec3 v = normalize(cam.cameraPos.xyz - inWorld);
    vec3 l = normalize(cam.sunDirection.xyz);
    float ndl = max(dot(n, l), 0.0);
    float wrap = clamp((dot(n, l) + 0.4) / 1.4, 0.0, 1.0);   // soft skin terminator
    float skin = pc.params.w < 0.5 ? wrap : ndl;              // teapot uses lambert
    vec3 albedo = pc.params.rgb;
    vec3 color = albedo * cam.sunColor.rgb * (skin * cam.sunDirection.w * 0.55);
    vec3 ambient = mix(cam.skyHorizon.rgb * 0.38, cam.skyZenith.rgb * 0.85,
                       clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    color += albedo * ambient * 0.9;
    vec3 h = normalize(l + v);
    color += cam.sunColor.rgb * pow(max(dot(n, h), 0.0), pc.params.w < 0.5 ? 26.0 : 90.0) *
             (pc.params.w < 0.5 ? 0.08 : 0.45);
    outColor = vec4(color, 1.0);
}
