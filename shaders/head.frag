// SilkCrowd - skin / teapot shading (opaque pass).
//
// This is the GPU twin of shadeSkin()/shadeEye() in src/render/software_renderer.cpp and
// evaluates the same analytical face masks as headFaceMasks() in src/core/assets.cpp, so
// the reference rasteriser and the Vulkan path shade a face identically. The kind of draw
// is carried in the sign of pc.params.w:
//   +w   plain surface (teapot, neck, shoulders), w is the specular gain
//   -1   skin with the face masks evaluated in head-local space
//   -2   eyeball (the palpebral fissure, iris, lashes and catchlights are analytic)
#version 460
layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorld;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inLocal;
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
layout(push_constant) uniform HeadPush { mat4 mvp; mat4 model; vec4 params; } pc;  // rgb albedo, w = specular / kind

const float kHeadCenterY = 0.02;
const float kHeadRy = 0.1040;

float sat(float x) { return clamp(x, 0.0, 1.0); }
float sq(float x) { return x * x; }

float band(float x, float centre, float halfWidth, float feather) {
    return 1.0 - smoothstep(halfWidth, halfWidth + feather, abs(x - centre));
}

float bump(float theta, float phi, float th0, float ph0, float thSigma, float phSigma) {
    float dt = atan(sin(theta - th0), cos(theta - th0));
    return exp(-0.5 * (sq(dt / thSigma) + sq((phi - ph0) / phSigma)));
}

struct FaceMasks {
    float brow;
    float lipUpper;
    float lipLower;
    float mouthLine;
    float nostril;
    float noseBridge;
    float noseSide;
    float blush;
    float ao;
    float scalp;
};

// The inverse of the head's parameterisation: theta = 0 faces +Z, +pi/2 is the right ear.
FaceMasks headFaceMasks(vec3 localPos) {
    FaceMasks m;
    float theta = atan(localPos.x, localPos.z);
    float phi = acos(clamp(localPos.y / kHeadRy, -1.0, 1.0));
    float at = abs(theta);
    float front = sat(localPos.z >= 0.0 ? cos(theta) : cos(theta));
    float fw = smoothstep(0.05, 0.62, front);

    // Eyebrow: a feathered band above the eye, thicker at the inner end, arched outward.
    float arch = 1.468 - 0.026 * smoothstep(0.18, 0.62, at);
    float thin = 1.0 - 0.45 * smoothstep(0.42, 0.62, at);
    m.brow = band(phi, arch, 0.019 * thin, 0.024 * thin) *
             band(at, 0.40, 0.165, 0.10) * fw * (0.55 + 0.45 * smoothstep(0.12, 0.30, at));

    // Lips: upper and lower vermilion either side of the stomion line.
    float halfLip = 0.165 - 0.035 * smoothstep(1.99, 2.12, phi);
    float w = band(at, 0.0, halfLip, 0.045);
    m.lipUpper = w * band(phi, 1.998, 0.036, 0.016);
    m.lipLower = w * band(phi, 2.092, 0.044, 0.018);
    m.mouthLine = band(at, 0.0, 0.115, 0.060) * band(phi, 2.045, 0.008, 0.012);

    // Nostrils.
    m.nostril = band(at, 0.058, 0.030, 0.020) * band(phi, 1.822, 0.016, 0.012) * fw;
    // The nose: a narrow ridge down the dorsum, plus the side walls beside it.
    m.noseBridge = fw * band(at, 0.0, 0.032, 0.026) * band(phi, 1.705, 0.105, 0.050);
    m.noseSide = sat(fw * (bump(at, phi, 0.050, 1.735, 0.026, 0.080) +
                           0.6 * bump(at, phi, 0.062, 1.812, 0.024, 0.030)));

    // Blush: cheeks, the tip of the nose and the ears.
    m.blush = fw * (0.9 * bump(at, phi, 0.62, 1.90, 0.26, 0.16) +
                    0.5 * bump(theta, phi, 0.0, 1.775, 0.12, 0.05));
    m.blush += 0.7 * bump(at, phi, 1.53, 1.56, 0.20, 0.22);
    m.blush = sat(m.blush);

    // Ambient occlusion in the creases.
    m.ao = fw * (0.85 * bump(at, phi, 0.40, 1.55, 0.22, 0.10) +
                 0.45 * bump(at, phi, 0.30, 1.93, 0.10, 0.13) +
                 0.55 * bump(at, phi, 0.0, 2.045, 0.14, 0.012) +
                 0.40 * bump(at, phi, 0.0, 2.27, 0.22, 0.09) +
                 0.30 * bump(at, phi, 0.56, 1.74, 0.15, 0.07) +
                 0.25 * bump(at, phi, 0.10, 1.30, 0.10, 0.05));
    m.ao += 0.5 * smoothstep(2.30, 2.70, phi);
    m.ao = sat(m.ao);
    float hairline = 1.10 + 0.92 * (0.5 - 0.5 * cos(theta));
    m.scalp = 1.0 - smoothstep(hairline - 0.06, hairline + 0.04, phi);
    return m;
}

// Small value noise, used for the skin mottling and the eyebrow strands.
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n = 0.0;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                vec3 c = i + vec3(float(dx), float(dy), float(dz));
                float wgt = (dx == 1 ? f.x : 1.0 - f.x) * (dy == 1 ? f.y : 1.0 - f.y) *
                            (dz == 1 ? f.z : 1.0 - f.z);
                n += wgt * hash13(c);
            }
    return n;
}

// Skin: a wrapped diffuse term with the red scattering at the terminator, a two-lobe
// specular, a key/fill/rim studio rig, and the analytic masks.
vec3 shadeSkin(vec3 localPos, vec3 nrm, vec3 V, vec3 baseAlbedo, float specGain,
               bool useMasks) {
    FaceMasks fm;
    if (useMasks) {
        fm = headFaceMasks(localPos);
    } else {
        fm.brow = 0.0; fm.lipUpper = 0.0; fm.lipLower = 0.0; fm.mouthLine = 0.0;
        fm.nostril = 0.0; fm.noseBridge = 0.0; fm.noseSide = 0.0; fm.blush = 0.0; fm.ao = 0.0;
    }
    vec3 L = normalize(cam.sunDirection.xyz);
    float mott = valueNoise(localPos * 900.0) * 0.6 + valueNoise(localPos * 240.0) * 0.4;
    vec3 alb = baseAlbedo * (0.96 + 0.07 * mott);
    alb = mix(alb, vec3(0.76, 0.42, 0.36), fm.blush * 0.42);
    alb = mix(alb, vec3(0.64, 0.29, 0.28), sat(fm.lipUpper * 1.1));
    alb = mix(alb, vec3(0.60, 0.24, 0.25), sat(fm.lipLower * 1.1));
    float browHair = 0.45 + 0.55 * (0.5 + 0.5 * sin(localPos.x * 2400.0 + localPos.y * 950.0)) *
                                       (0.72 + 0.28 * sin(localPos.x * 520.0 - localPos.y * 310.0));
    alb = mix(alb, vec3(0.22, 0.13, 0.09), fm.brow * 0.86 * browHair);
    alb = mix(alb, alb * 1.10, fm.noseBridge * 0.55);
    alb = mix(alb, alb * vec3(0.78, 0.74, 0.74), fm.noseSide * 0.85);
    alb = mix(alb, vec3(0.07, 0.04, 0.035), fm.nostril * 0.9);
    alb = mix(alb, vec3(0.26, 0.11, 0.12), fm.mouthLine * 0.95);
    alb = mix(alb, vec3(0.028, 0.017, 0.012), fm.scalp * 0.97);

    vec3 keyDir = normalize(L + vec3(0.0, 0.15, 0.0));
    vec3 fillDir = normalize(vec3(-keyDir.x * 0.9, 0.25, -keyDir.z * 0.35 + 0.55));
    vec3 rimDir = normalize(vec3(-keyDir.x * 0.6, 0.45, -keyDir.z));

    float ndl = dot(nrm, keyDir);
    float wrapDiff = sat((ndl + 0.42) / 1.42);
    float terminator = exp(-ndl * ndl * 6.0);
    vec3 diffuse = alb * wrapDiff + vec3(0.34, 0.09, 0.06) * terminator * 0.26;
    diffuse *= cam.sunDirection.w * 0.55;
    diffuse += alb * cam.skyHorizon.rgb * (sat(dot(nrm, fillDir)) * 0.30 + 0.06);

    vec3 H = normalize(keyDir + V);
    float nh = sat(dot(nrm, H));
    float f0 = 0.028;
    float fres = f0 + (1.0 - f0) * pow(1.0 - sat(dot(nrm, V)), 5.0);
    float gloss = 1.0 + 2.5 * sat(fm.lipUpper + fm.lipLower);
    float spec = (pow(nh, 60.0) * 0.55 + pow(nh, 10.0) * 0.05) * specGain * gloss;

    float ao = 1.0 - 0.72 * fm.ao;
    vec3 color = diffuse * cam.sunColor.rgb * ao + vec3(spec * fres) * cam.sunColor.rgb * ao;
    vec3 amb = mix(cam.skyHorizon.rgb * 0.40, cam.skyZenith.rgb * 0.88, sat(nrm.y * 0.5 + 0.5));
    color += alb * amb * (0.80 * ao);
    float rim = pow(1.0 - sat(dot(nrm, V)), 3.5) * sat(dot(nrm, rimDir));
    color += cam.skyZenith.rgb * (rim * 0.65);
    return color;
}

// The eyeball: an almond palpebral fissure, the iris/limbal ring/pupil, individual lashes
// and two catchlights. Measured in the eye's own frame.
vec3 shadeEye(vec3 nrm, vec3 V, vec3 localPos) {
    float side = localPos.x >= 0.0 ? 1.0 : -1.0;
    vec3 outward = normalize(vec3(side * 0.39, 0.0, 0.92));
    vec3 gaze = normalize(outward * 0.45 + V * 0.55);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), gaze));
    vec3 up = cross(gaze, right);

    float f = dot(nrm, gaze);
    float h = dot(nrm, right);
    float v = dot(nrm, up);

    float tilt = 0.10 * h * side;
    float ax = abs(h) / 0.575;
    float ay = abs(v - tilt) / 0.255;
    float r = pow(ax, 1.7) + ay * ay;
    float outside = smoothstep(1.0, 1.22, r);

    vec3 sclera = vec3(0.85, 0.81, 0.775);
    sclera = mix(sclera, vec3(0.72, 0.66, 0.64), sat(abs(h) * 1.4 - 0.35));
    float iris = smoothstep(0.835, 0.872, f);
    vec3 irisColor = vec3(0.36, 0.22, 0.11);
    irisColor *= 0.75 + 0.55 * sat(-v * 1.5 + 0.35);
    float ang = atan(v, h);
    irisColor *= 0.85 + 0.15 * sin(ang * 26.0);
    irisColor = mix(irisColor, vec3(0.15, 0.09, 0.05), smoothstep(0.90, 0.95, f));
    vec3 c = mix(sclera, irisColor, iris);
    c = mix(c, vec3(0.02, 0.02, 0.02), smoothstep(0.968, 0.982, f));
    float limbal = exp(-sq((f - 0.872) / 0.016));
    c = mix(c, vec3(0.04, 0.03, 0.03), limbal * 0.75);

    vec3 lid = vec3(0.66, 0.45, 0.39);
    c = mix(c, lid, outside);
    float lash = exp(-sq((r - 1.03) / 0.055));
    float lashUp = smoothstep(-0.28, 0.05, v - tilt);
    c = mix(c, vec3(0.045, 0.035, 0.030), lash * (0.12 + 0.55 * lashUp));
    float crease = exp(-sq((v - tilt - 0.42) / 0.10));
    c = mix(c, lid * 0.72, sat(crease) * 0.55 * smoothstep(0.9, 1.15, r));
    c *= 1.0 - 0.34 * sat(smoothstep(-0.05, 0.55, v)) * (1.0 - outside);

    // Eyelashes: individual tapered strands off the lash line, longest over the middle.
    {
        float vTop = tilt + 0.255 * sqrt(max(0.0, 1.0 - pow(ax, 1.7)));
        float arc = sat((h + 0.60) / 1.20);
        float lenUp = (0.30 + 0.16 * sin(arc * 3.1)) *
                      (1.0 - 0.55 * pow(abs(h) / 0.62, 3.0));
        float comb = 0.5 + 0.5 * sin(arc * 118.0 + 0.8);
        float above = v - vTop;
        if (above > -0.05) {
            float t = above / max(1e-3, lenUp);
            if (t > -0.2 && t < 1.0) {
                float taper = smoothstep(1.0, 0.55, sat(t));
                c = mix(c, vec3(0.05, 0.038, 0.032), taper * smoothstep(0.42, 0.86, comb) * 0.92);
            }
        }
        float lenLo = 0.13 * (1.0 - 0.5 * pow(abs(h) / 0.62, 2.0));
        float combLo = 0.5 + 0.5 * sin(arc * 74.0 + 2.1);
        float vBot = tilt - 0.255 * sqrt(max(0.0, 1.0 - pow(ax, 1.7)));
        float below = vBot - v;
        if (below > -0.04) {
            float t = below / max(1e-3, lenLo);
            if (t > -0.2 && t < 1.0) {
                float taper = smoothstep(1.0, 0.5, sat(t));
                c = mix(c, vec3(0.09, 0.07, 0.06),
                        taper * smoothstep(0.55, 0.95, combLo) * 0.62);
            }
        }
    }

    vec3 L = normalize(cam.sunDirection.xyz);
    vec3 H = normalize(L + V);
    c += cam.sunColor.rgb * (pow(sat(dot(nrm, H)), 400.0) * 2.6);
    c += cam.skyZenith.rgb * (pow(sat(dot(nrm, normalize(V + vec3(-0.3, 0.5, 0.4)))), 260.0) * 0.8);
    return c * (0.72 + 0.28 * sat(dot(nrm, L)));
}

void main() {
    vec3 n = normalize(inNormal);
    vec3 v = normalize(cam.cameraPos.xyz - inWorld);
    vec3 local = inLocal - vec3(0.0, kHeadCenterY, 0.0);
    float kind = pc.params.w;

    if (kind < -1.5) {
        outColor = vec4(shadeEye(n, v, local), 1.0);
        return;
    }
    if (kind < -0.5) {
        outColor = vec4(shadeSkin(local, n, v, pc.params.rgb, 0.55, true), 1.0);
        return;
    }

    // Plain surfaces: teapot glaze, neck and shoulders without the face features.
    float ndl = max(dot(n, cam.sunDirection.xyz), 0.0);
    float skin = clamp((ndl + 0.4) / 1.4, 0.0, 1.0);   // soft skin terminator
    vec3 albedo = pc.params.rgb;
    vec3 color = albedo * cam.sunColor.rgb * (skin * cam.sunDirection.w * 0.55);
    vec3 ambient = mix(cam.skyHorizon.rgb * 0.38, cam.skyZenith.rgb * 0.85,
                       clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    color += albedo * ambient * 0.9;
    vec3 h = normalize(cam.sunDirection.xyz + v);
    color += cam.sunColor.rgb * pow(max(dot(n, h), 0.0), 60.0) * 0.10;
    outColor = vec4(color, 1.0);
}
