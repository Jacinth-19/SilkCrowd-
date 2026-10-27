// Curve machinery for the hair mesh (paper Section 2.2 / 3.1, Figure 3d-3e).
//
// The original hair mesh (Yuksel et al. 2009a) is a cubic Catmull-Rom spline with
// centripetal parameterization. For the GPU we must express each segment as
//   Catmull-Rom  ->  cubic Bezier (p0, b0+, b5-, p5)        [Figure 3d]
//                ->  two quadratic Beziers                  [Figure 3e, Truong et al. 2020]
// so that the 3D texture only needs 3 intermediate slices per layer interval.
#pragma once

#include <vector>

#include "silkcrowd/math.h"

namespace silk {

// A cubic Bezier segment, named as in Figure 3d: p0, b0+, b5-, p5.
struct CubicBezier {
    Vec3 p0{}, b0plus{}, b5minus{}, p5{};
    Vec3 eval(float t) const;
};

// Two quadratic Beziers approximating `CubicBezier` (Figure 3e).
// First curve:  (p0,   q0plus,  q05)
// Second curve: (q05,  q5minus, p5)
struct QuadraticPair {
    Vec3 p0{}, q0plus{}, q05{};
    Vec3 q5minus{}, p5{};
    Vec3 eval(float t) const;  // t in [0,1] over the whole cubic
};

// --- Catmull-Rom (centripetal) -------------------------------------------------

// Centripetal knot sequence: t_{i+1} = t_i + |p_{i+1} - p_i|^0.5  (Yuksel et al. 2009b)
std::vector<float> centripetalKnots(const std::vector<Vec3>& points);

// Tangent m_i of the non-uniform Catmull-Rom spline at knot i.
// m_i = (t_{i+1}-t_i)/(t_{i+1}-t_{i-1}) * (p_i - p_{i-1})/(t_i - t_{i-1})
//     + (t_i - t_{i-1})/(t_{i+1}-t_{i-1}) * (p_{i+1} - p_i)/(t_{i+1}-t_i)
// Endpoints are handled by reflecting the neighbouring knot interval.
Vec3 catmullRomTangent(const std::vector<float>& knots, const std::vector<Vec3>& points,
                       int i);

// Direct evaluation of the Catmull-Rom segment between points[i] and points[i+1]
// (Hermite form). Used as ground truth in the unit tests.
Vec3 catmullRomEval(const std::vector<float>& knots, const std::vector<Vec3>& points, int i,
                    float u);

// Converts the Catmull-Rom segment between points[i] and points[i+1] into a cubic Bezier.
CubicBezier catmullRomToCubicBezier(const std::vector<float>& knots,
                                    const std::vector<Vec3>& points, int i);

// Truong et al. 2020 two-quadratic approximation of a cubic Bezier (Figure 3e).
//
// The cubic is split with de Casteljau at t = 1/2 into a left half (L0..L3) and a right
// half (R0..R3). Each half is replaced by the quadratic with the same endpoints and the
// L2-optimal middle control point B = (3(H1+H2) - (H0+H3))/4, i.e.
//     q0+    = ( 9 p0 + 21 b0+ + 3 b5- - p5 ) / 32
//     q0,5   = C(0.5) = (p0 + 3 b0+ + 3 b5- + p5) / 8
//     q5-    = ( 9 p5 + 21 b5- + 3 b0+ - p0 ) / 32
// The two quadratics are exactly C1 at q0,5 (same tangent direction and magnitude), so
// the strand has no kink where the two texture-space quadratic segments meet.
QuadraticPair cubicToTwoQuadratics(const CubicBezier& c);

// Evaluates a quadratic Bezier with de Casteljau (this is what the texture units do).
Vec3 evalQuadratic(Vec3 a, Vec3 b, Vec3 c, float u);

// de Casteljau butterfly: the two intermediate points of a quadratic at parameter u.
void quadraticDeCasteljau(Vec3 a, Vec3 b, Vec3 c, float u, Vec3* ab, Vec3* bc);

}  // namespace silk
