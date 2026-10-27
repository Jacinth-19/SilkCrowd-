// SilkCrowd - fullscreen triangle (no vertex buffer).
#version 460
layout(location = 0) out vec2 outUV;
void main() {
    // 3-vertex fullscreen triangle covering the whole viewport.
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
    outUV.y = 1.0 - outUV.y;
}
