#include "silkcrowd/catmull_rom.h"

#include <cassert>

namespace silk {

Vec3 CubicBezier::eval(float t) const {
    const float u = 1.0f - t;
    return p0 * (u * u * u) + b0plus * (3.0f * u * u * t) + b5minus * (3.0f * u * t * t) +
           p5 * (t * t * t);
}

Vec3 evalQuadratic(Vec3 a, Vec3 b, Vec3 c, float u) {
    const float v = 1.0f - u;
    return a * (v * v) + b * (2.0f * v * u) + c * (u * u);
}

void quadraticDeCasteljau(Vec3 a, Vec3 b, Vec3 c, float u, Vec3* ab, Vec3* bc) {
    *ab = lerp(a, b, u);   // first  trilinear sample
    *bc = lerp(b, c, u);   // second trilinear sample
}

Vec3 QuadraticPair::eval(float t) const {
    if (t <= 0.5f) return evalQuadratic(p0, q0plus, q05, t * 2.0f);
    return evalQuadratic(q05, q5minus, p5, (t - 0.5f) * 2.0f);
}

std::vector<float> centripetalKnots(const std::vector<Vec3>& points) {
    std::vector<float> knots(points.size(), 0.0f);
    for (size_t i = 1; i < points.size(); ++i) {
        const float d = length(points[i] - points[i - 1]);
        knots[i] = knots[i - 1] + std::sqrt(std::max(d, 1e-8f));  // d^0.5
    }
    return knots;
}

Vec3 catmullRomTangent(const std::vector<float>& knots, const std::vector<Vec3>& points,
                       int i) {
    const int n = static_cast<int>(points.size());
    assert(n >= 2);
    if (n == 2) {
        const float d = std::max(knots[1] - knots[0], 1e-8f);
        return (points[1] - points[0]) / d;
    }

    int im1 = i - 1, ip1 = i + 1;
    float ti = knots[i];
    float tim1, tip1;
    if (im1 < 0) {
        // Reflect the first knot interval.
        tim1 = ti - (knots[1] - knots[0]);
        im1 = 0;
        tim1 = ti - (knots[1] - knots[0]);
    } else {
        tim1 = knots[im1];
    }
    if (ip1 > n - 1) {
        tip1 = ti + (knots[n - 1] - knots[n - 2]);
        ip1 = n - 1;
    } else {
        tip1 = knots[ip1];
    }
    if (im1 == i && ip1 == i) return Vec3(0, 0, 0);

    const float d0 = std::max(ti - tim1, 1e-8f);
    const float d1 = std::max(tip1 - ti, 1e-8f);

    const Vec3 chordPrev = im1 == i ? Vec3(0, 0, 0) : (points[i] - points[im1]) / d0;
    const Vec3 chordNext = ip1 == i ? Vec3(0, 0, 0) : (points[ip1] - points[i]) / d1;

    // Weighted blend of the two adjacent chords; reduces to (p_{i+1}-p_{i-1})/2 for
    // uniform knots.
    return chordPrev * (d1 / (d0 + d1)) + chordNext * (d0 / (d0 + d1));
}

Vec3 catmullRomEval(const std::vector<float>& knots, const std::vector<Vec3>& points, int i,
                    float u) {
    const float t0 = knots[i], t1 = knots[std::min(i + 1, static_cast<int>(knots.size()) - 1)];
    const float dt = t1 - t0;
    const Vec3 p0 = points[i], p1 = points[std::min(i + 1, static_cast<int>(points.size()) - 1)];
    const Vec3 m0 = catmullRomTangent(knots, points, i);
    const Vec3 m1 = catmullRomTangent(knots, points, std::min(i + 1, static_cast<int>(points.size()) - 1));
    // Hermite
    const float u2 = u * u, u3 = u2 * u;
    const float h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
    const float h10 = u3 - 2.0f * u2 + u;
    const float h01 = -2.0f * u3 + 3.0f * u2;
    const float h11 = u3 - u2;
    return p0 * h00 + m0 * (h10 * dt) + p1 * h01 + m1 * (h11 * dt);
}

CubicBezier catmullRomToCubicBezier(const std::vector<float>& knots,
                                    const std::vector<Vec3>& points, int i) {
    const int n = static_cast<int>(points.size());
    const int i1 = std::min(i + 1, n - 1);
    const float dt = std::max(knots[i1] - knots[i], 1e-8f);
    const Vec3 m0 = catmullRomTangent(knots, points, i);
    const Vec3 m1 = catmullRomTangent(knots, points, i1);
    CubicBezier c;
    c.p0 = points[i];
    c.p5 = points[i1];
    // Hermite -> Bezier:  b0+ = p0 + m0*dt/3 ,  b5- = p5 - m1*dt/3
    c.b0plus = c.p0 + m0 * (dt / 3.0f);
    c.b5minus = c.p5 - m1 * (dt / 3.0f);
    return c;
}

QuadraticPair cubicToTwoQuadratics(const CubicBezier& c) {
    // Truong et al. 2020 two-quadratic approximation, written in the form used by
    // Figure 3e. The cubic is split with de Casteljau at t = 1/2 into a left half
    // (L0..L3) and a right half (R0..R3); each half is replaced by the quadratic with
    // the same endpoints whose middle control point is the L2-optimal one
    //
    //     B = ( 3 (H1 + H2) - (H0 + H3) ) / 4
    //
    // For the left half this is
    //     q0+  = ( 9 p0 + 21 b0+ + 3 b5- - p5 ) / 32
    // and symmetrically q5- = (9 p5 + 21 b5- + 3 b0+ - p0)/32, with q0,5 = C(0.5).
    //
    // This construction is exactly C1 at the joint (both quadratics share the tangent
    // direction *and* magnitude there, so the hair strand has no kink at the slice
    // boundary), and it is the most accurate of the candidate 2-quadratic splits:
    // measured against the cubic it approximates, the maximum deviation is ~0.1% of
    // the segment length (see tests/tests.cpp and docs/VALIDATION.md).
    const Vec3 p01 = (c.p0 + c.b0plus) * 0.5f;                 // L1
    const Vec3 p12 = (c.b0plus + c.b5minus) * 0.5f;
    const Vec3 p23 = (c.b5minus + c.p5) * 0.5f;
    const Vec3 p012 = (p01 + p12) * 0.5f;                      // L2
    const Vec3 p123 = (p12 + p23) * 0.5f;                      // R1
    const Vec3 mid = (p012 + p123) * 0.5f;                     // L3 == R0 == C(0.5)

    QuadraticPair q;
    q.p0 = c.p0;
    q.q05 = mid;
    q.p5 = c.p5;
    q.q0plus = ((p01 + p012) * 3.0f - (c.p0 + mid)) * 0.25f;
    q.q5minus = ((p123 + p23) * 3.0f - (mid + c.p5)) * 0.25f;
    return q;
}

}  // namespace silk
