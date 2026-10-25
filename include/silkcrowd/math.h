// SilkCrowd - dependency-free math for the CPU reference implementation.
// The GPU backend uses GLM; this header exists so that the paper-critical
// math can be unit-tested without any Vulkan/Qt/GLM dependency.
#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace silk {

constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
    float x = 0.0f, y = 0.0f;
    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
    float& operator[](int i) { return i == 0 ? x : y; }
    float operator[](int i) const { return i == 0 ? x : y; }
};

inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }
inline Vec2 operator*(float s, Vec2 a) { return {a.x * s, a.y * s}; }
inline Vec2 operator/(Vec2 a, float s) { return {a.x / s, a.y / s}; }
inline Vec2& operator+=(Vec2& a, Vec2 b) { a.x += b.x; a.y += b.y; return a; }
inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline float cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
inline float length(Vec2 a) { return std::sqrt(dot(a, a)); }

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    Vec3() = default;
    Vec3(float v) : x(v), y(v), z(v) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    float& operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3 operator*(float s, Vec3 a) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline Vec3& operator*=(Vec3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline float length2(Vec3 a) { return dot(a, a); }
inline Vec3 normalize(Vec3 a) {
    const float l = length(a);
    return l > 1e-20f ? a * (1.0f / l) : Vec3(0.0f, 1.0f, 0.0f);
}
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
inline Vec3 minv(Vec3 a, Vec3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline Vec3 maxv(Vec3 a, Vec3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
inline Vec3 reflect(Vec3 d, Vec3 n) { return d - n * (2.0f * dot(d, n)); }

// 3x3 matrix, column-major (columns are basis vectors), matching GLSL layout so
// that Mp = [u v w] built from the texture lookups can be uploaded verbatim.
struct Mat3 {
    Vec3 c0{1, 0, 0}, c1{0, 1, 0}, c2{0, 0, 1};
    Mat3() = default;
    Mat3(Vec3 a, Vec3 b, Vec3 c) : c0(a), c1(b), c2(c) {}
    Vec3 operator*(Vec3 v) const { return c0 * v.x + c1 * v.y + c2 * v.z; }
};

struct Mat4 {
    // row-major [4][4]
    float m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

    static Mat4 identity() { return Mat4{}; }

    static Mat4 translate(Vec3 t) {
        Mat4 r;
        r.m[0][3] = t.x; r.m[1][3] = t.y; r.m[2][3] = t.z;
        return r;
    }
    static Mat4 scale(Vec3 s) {
        Mat4 r;
        r.m[0][0] = s.x; r.m[1][1] = s.y; r.m[2][2] = s.z;
        return r;
    }
    static Mat4 rotateY(float a) {
        Mat4 r;
        const float c = std::cos(a), s = std::sin(a);
        r.m[0][0] = c; r.m[0][2] = s; r.m[2][0] = -s; r.m[2][2] = c;
        return r;
    }
    static Mat4 rotateX(float a) {
        Mat4 r;
        const float c = std::cos(a), s = std::sin(a);
        r.m[1][1] = c; r.m[1][2] = -s; r.m[2][1] = s; r.m[2][2] = c;
        return r;
    }
    static Mat4 rotateZ(float a) {
        Mat4 r;
        const float c = std::cos(a), s = std::sin(a);
        r.m[0][0] = c; r.m[0][1] = -s; r.m[1][0] = s; r.m[1][1] = c;
        return r;
    }
    // Right-handed look-at (camera looks down -Z).
    static Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
        const Vec3 f = normalize(target - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);
        Mat4 r;
        r.m[0][0] = s.x; r.m[0][1] = s.y; r.m[0][2] = s.z; r.m[0][3] = -dot(s, eye);
        r.m[1][0] = u.x; r.m[1][1] = u.y; r.m[1][2] = u.z; r.m[1][3] = -dot(u, eye);
        r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
        return r;
    }
    // Vulkan-style perspective: depth in [0,1], y-down not applied here.
    static Mat4 perspective(float fovYRadians, float aspect, float zn, float zf) {
        const float t = 1.0f / std::tan(fovYRadians * 0.5f);
        Mat4 r;
        for (auto& row : r.m) for (float& v : row) v = 0.0f;
        r.m[0][0] = t / aspect;
        r.m[1][1] = -t;  // flip Y for screen-space (y grows downward)
        r.m[2][2] = zf / (zn - zf);
        r.m[2][3] = (zn * zf) / (zn - zf);
        r.m[3][2] = -1.0f;
        return r;
    }
    static Mat4 ortho(float l, float r_, float b, float t, float zn, float zf) {
        Mat4 r;
        for (auto& row : r.m) for (float& v : row) v = 0.0f;
        r.m[0][0] = 2.0f / (r_ - l);
        r.m[1][1] = -2.0f / (t - b);
        r.m[2][2] = 1.0f / (zn - zf);
        r.m[0][3] = -(r_ + l) / (r_ - l);
        r.m[1][3] = (t + b) / (t - b);
        r.m[2][3] = zn / (zn - zf);
        r.m[3][3] = 1.0f;
        return r;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                float s = 0.0f;
                for (int k = 0; k < 4; ++k) s += m[i][k] * o.m[k][j];
                r.m[i][j] = s;
            }
        return r;
    }

    // Transforms a point and returns (clip.xyz, clip.w) in out.
    void transformPoint(Vec3 p, Vec3* xyz, float* w) const {
        *xyz = Vec3{m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
                    m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
                    m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]};
        *w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
    }

    // Inverse of an affine matrix (last row 0,0,0,1).
    Mat4 inverseAffine() const {
        Mat4 r;
        const float a = m[0][0], b = m[0][1], c = m[0][2];
        const float d = m[1][0], e = m[1][1], f = m[1][2];
        const float g = m[2][0], h = m[2][1], i = m[2][2];
        const float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        const float inv = det != 0.0f ? 1.0f / det : 0.0f;
        r.m[0][0] = (e * i - f * h) * inv;
        r.m[0][1] = (c * h - b * i) * inv;
        r.m[0][2] = (b * f - c * e) * inv;
        r.m[1][0] = (f * g - d * i) * inv;
        r.m[1][1] = (a * i - c * g) * inv;
        r.m[1][2] = (c * d - a * f) * inv;
        r.m[2][0] = (d * h - e * g) * inv;
        r.m[2][1] = (b * g - a * h) * inv;
        r.m[2][2] = (a * e - b * d) * inv;
        const Vec3 t{m[0][3], m[1][3], m[2][3]};
        r.m[0][3] = -(r.m[0][0] * t.x + r.m[0][1] * t.y + r.m[0][2] * t.z);
        r.m[1][3] = -(r.m[1][0] * t.x + r.m[1][1] * t.y + r.m[1][2] * t.z);
        r.m[2][3] = -(r.m[2][0] * t.x + r.m[2][1] * t.y + r.m[2][2] * t.z);
        return r;
    }

    Vec3 transformPoint(Vec3 p) const {
        Vec3 xyz; float w;
        transformPoint(p, &xyz, &w);
        return w != 1.0f && w != 0.0f ? xyz * (1.0f / w) : xyz;
    }
};

// --- small deterministic hashing / RNG (no <random> dependency on GPU parity) ---
inline uint32_t pcgHash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float hashToFloat(uint32_t h) {
    return (h >> 8) * (1.0f / 16777216.0f);  // [0,1)
}
// Same construction as the GLSL `hash11`, so CPU validation matches the shader bit-for-bit
// for the styling random offsets.
inline float hash11(float p) {
    p = std::fmod(p * 0.1031f, 1.0f);
    p *= p + 33.33f;
    p *= p + p;
    return std::fmod(p, 1.0f);
}
inline float hash12(float x, float y) {
    float p = std::fmod(x * 0.1031f + y * 0.3323f, 1.0f);
    p *= p + 33.33f;
    p *= p + p;
    return std::fmod(p + std::fmod(x + y, 1.0f), 1.0f);
}
inline Vec3 hash31(float x, float y, float z) {
    const float a = hash12(x, y + z * 17.13f);
    const float b = hash12(y + 3.7f, z + x * 11.9f);
    const float c = hash12(z + 7.1f, x + y * 23.3f);
    return {a, b, c};
}

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float saturate(float v) { return clampf(v, 0.0f, 1.0f); }
inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
inline float deg2rad(float d) { return d * (kPi / 180.0f); }
inline float smoothstepf(float e0, float e1, float x) {
    const float t = saturate((x - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}

}  // namespace silk
