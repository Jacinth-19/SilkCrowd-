// SilkCrowd - bright pass + separable blur for bloom (single pass over a 1/4 res target).
#version 460
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D srcTex;
layout(push_constant) uniform BloomParams { vec4 params; } p;  // x = threshold, y = texel size
void main() {
    vec2 texel = vec2(p.params.y);
    vec3 sum = vec3(0.0);
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x) {
            vec3 c = texture(srcTex, inUV + vec2(float(x), float(y)) * texel).rgb;
            float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
            sum += c * max(l - p.params.x, 0.0) / max(l, 1e-4);
        }
    outColor = vec4(sum / 25.0, 1.0);
}
