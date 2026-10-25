// SilkCrowd - head / body / teapot geometry (opaque pass and shadow pass).
#version 460
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(push_constant) uniform HeadPush { mat4 mvp; mat4 model; vec4 params; } pc;
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outWorld;
layout(location = 2) out vec2 outUV;
void main() {
    outWorld = (pc.model * vec4(inPosition, 1.0)).xyz;
    outNormal = mat3(pc.model) * inNormal;
    outUV = inUV;
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
