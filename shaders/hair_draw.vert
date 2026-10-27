// SilkCrowd - vertex stage for the compute-shader fallback path.
// Reads the ribbon geometry produced by hair_compute.comp and forwards the same
// attributes as the mesh shader, so hair.frag is shared between both paths.
#version 460
#extension GL_GOOGLE_include_directive : require
layout(std430, set = 3, binding = 3) readonly buffer StrandVertices {
    vec4 positions[];   // xyz = object-space position, w = w styling coordinate
} verts;
layout(std430, set = 3, binding = 4) readonly buffer StrandTangents {
    vec4 tangents[];    // xyz = object-space tangent
} tangs;
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 params;        // x = first vertex, y = vertex count
    vec4 params2;
} pc;
layout(location = 0) out vec3 outTangent;
layout(location = 1) out float outW;
layout(location = 2) flat out uint outSeed;
layout(location = 3) out vec3 outWorldPos;
void main() {
    vec4 p = verts.positions[gl_VertexIndex];
    vec4 t = tangs.tangents[gl_VertexIndex];
    outWorldPos = (pc.model * vec4(p.xyz, 1.0)).xyz;
    outTangent = normalize((pc.model * vec4(t.xyz, 0.0)).xyz + vec3(1e-9));
    outW = p.w;
    outSeed = floatBitsToUint(t.w);
    gl_Position = pc.mvp * vec4(p.xyz, 1.0);
}
