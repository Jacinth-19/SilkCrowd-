#include "silkcrowd/sim.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace silk {
namespace {

inline Vec3 turbulentWind(const SimParams& p, float t, Vec3 position) {
    // Sum of three incommensurate sines -> a continuously varying gust field.
    const float f = p.windFrequency;
    const float phase = position.x * 11.0f + position.y * 7.0f + position.z * 13.0f;
    const float g = 0.6f + 0.4f * std::sin(t * f * 2.0f * kPi + phase) *
                              std::sin(t * f * 3.7f * kPi + phase * 0.7f);
    const Vec3 dir = normalize(p.windDirection + Vec3(0.0f,
                                                     0.35f * std::sin(t * f * 1.3f * kPi),
                                                     0.35f * std::cos(t * f * 0.7f * kPi)));
    const float turb = 1.0f + p.windTurbulence * 0.25f *
                                 std::sin(t * f * 5.1f * kPi + phase * 1.7f);
    return dir * (p.windStrength * g * turb);
}

}  // namespace

std::string SimStats::toString() const {
    std::ostringstream os;
    os << "sim: " << substeps << " substeps, max |v| " << maxVelocity << " m/s, kinetic "
       << totalKinetic << " J, mean stretch error " << meanStretchError
       << (stable ? " [stable]" : " [UNSTABLE]");
    return os.str();
}

void HairMeshSimulator::init(const HairMesh& restMesh, const SimParams& params) {
    numVertices_ = restMesh.numVertices();
    numLayers_ = restMesh.numLayers();
    const size_t n = static_cast<size_t>(numVertices_) * numLayers_;

    restShape_.resize(n);
    position_.resize(n);
    previous_.resize(n);
    velocity_.assign(n, Vec3(0, 0, 0));
    invMass_.resize(n);
    for (uint32_t l = 0; l < numLayers_; ++l)
        for (uint32_t v = 0; v < numVertices_; ++v) {
            const size_t k = static_cast<size_t>(l) * numVertices_ + v;
            restShape_[k] = restMesh.layerVertex(l, v);
            position_[k] = restShape_[k];
            // The root layer is pinned to the scalp; the rest of the mesh has unit mass.
            invMass_[k] = (l == 0) ? 0.0f : 1.0f;
        }

    // Constraint network (this is the "hair mesh simulation" of Wu and Yuksel 2016
    // expressed in XPBD form):
    //  * stretch constraints along every extrusion edge (root -> tip per vertex),
    //  * stretch constraints across each root face, to keep the bundles coherent,
    //  * bending constraints between the vertices two layers apart, which is what gives
    //    the hair its resistance to folding and its characteristic shape memory.
    stretch_.clear();
    bending_.clear();
    for (uint32_t v = 0; v < numVertices_; ++v) {
        for (uint32_t l = 0; l + 1 < numLayers_; ++l) {
            DistanceConstraint c;
            c.a = l * numVertices_ + v;
            c.b = (l + 1) * numVertices_ + v;
            c.rest = length(restShape_[c.b] - restShape_[c.a]);
            c.compliance = params.stretchCompliance;
            stretch_.push_back(c);
        }
        for (uint32_t l = 0; l + 2 < numLayers_; ++l) {
            DistanceConstraint c;
            c.a = l * numVertices_ + v;
            c.b = (l + 2) * numVertices_ + v;
            c.rest = length(restShape_[c.b] - restShape_[c.a]);
            c.compliance = params.bendingCompliance;
            bending_.push_back(c);
        }
    }
    for (const HairMeshFace& f : restMesh.faces()) {
        for (uint32_t l = 1; l < numLayers_; ++l) {
            for (uint32_t c = 0; c < f.count; ++c) {
                const uint32_t v0 = f.v[c];
                const uint32_t v1 = f.v[(c + 1) % f.count];
                if (v1 < v0) continue;  // one constraint per edge
                DistanceConstraint dc;
                dc.a = l * numVertices_ + v0;
                dc.b = l * numVertices_ + v1;
                dc.rest = length(restShape_[dc.b] - restShape_[dc.a]);
                dc.compliance = params.stretchCompliance * 4.0f;
                stretch_.push_back(dc);
            }
        }
    }
    stats_ = SimStats{};
    time_ = 0.0f;
}

void HairMeshSimulator::computeSagFreeRest(const HairMesh& mesh, const SimParams& params,
                                           int iterations) {
    // Sag-free initialization (Hsu et al. 2022), simplified to our constraint set: relax
    // the shape under gravity alone until it reaches equilibrium, then use that
    // equilibrium as the rest shape for the bending constraints. Simulation therefore
    // starts in a state that gravity does not perturb, which removes the initial "sag"
    // that otherwise produces a visible drop of the whole hairstyle.
    HairMeshSimulator relax;
    SimParams p = params;
    p.windStrength = 0.0f;
    p.drag = 0.6f;
    p.linearDamping = 4.0f;
    relax.init(mesh, p);
    // Start from the authored shape and let gravity settle it.
    const float dt = 1.0f / 120.0f;
    HairMesh work = mesh;
    for (int i = 0; i < iterations; ++i) {
        relax.step(&work, dt, p);
        relax.step(&work, dt, p);
    }
    // Store the settled shape as the new rest shape for this simulator.
    for (uint32_t l = 0; l < numLayers_; ++l)
        for (uint32_t v = 0; v < numVertices_; ++v) {
            const size_t k = static_cast<size_t>(l) * numVertices_ + v;
            restShape_[k] = work.layerVertex(l, v);
        }
    // Recompute the constraint rest lengths from the settled shape.
    for (DistanceConstraint& c : stretch_) c.rest = length(restShape_[c.b] - restShape_[c.a]);
    for (DistanceConstraint& c : bending_) c.rest = length(restShape_[c.b] - restShape_[c.a]);
    for (size_t k = 0; k < restShape_.size(); ++k) {
        position_[k] = restShape_[k];
        velocity_[k] = Vec3(0, 0, 0);
    }
}

void HairMeshSimulator::solve(HairMesh* mesh, const SimParams& params, float dt,
                              bool withForces) {
    const float subDt = dt / static_cast<float>(std::max(params.substeps, 1));
    const float invSubDt2 = 1.0f / (subDt * subDt);
    const int iters = std::max(params.iterations, 1);

    for (int s = 0; s < std::max(params.substeps, 1); ++s) {
        time_ += subDt;
        // --- predict with gravity, wind and drag (Wu and Yuksel 2016 force models) ---
        for (size_t k = 0; k < position_.size(); ++k) {
            if (invMass_[k] == 0.0f) {
                velocity_[k] = Vec3(0, 0, 0);
                previous_[k] = position_[k];
                continue;
            }
            Vec3 accel(0.0f, params.gravity, 0.0f);
            if (withForces) {
                // Wind is modelled as drag towards a moving air mass (the standard
                // aerodynamic model used for hair): a = k (v_air - v), so that the hair
                // reaches a deflection where wind and restoring forces balance. A pure
                // acceleration term would be swamped by gravity for such light geometry.
                const Vec3 windVel = turbulentWind(params, time_, position_[k]);
                accel += (windVel - velocity_[k]) * 6.0f;
                // Ordinary air drag against still air.
                accel -= velocity_[k] * params.drag;
            }
            velocity_[k] *= (1.0f - clampf(params.linearDamping * subDt, 0.0f, 0.95f));
            previous_[k] = position_[k];
            position_[k] += velocity_[k] * subDt + accel * (subDt * subDt);

            // --- mouse drag interaction ---
            if (params.dragging) {
                const float d = length(position_[k] - params.dragPoint);
                if (d < params.dragRadius) {
                    const float w = 1.0f - d / params.dragRadius;
                    position_[k] += params.dragDelta * (w * w);
                }
            }
        }

        // --- XPBD constraint projection ---
        auto project = [&](std::vector<DistanceConstraint>& list) {
            for (DistanceConstraint& c : list) {
                const float wSum = invMass_[c.a] + invMass_[c.b];
                if (wSum == 0.0f) continue;
                const Vec3 d = position_[c.b] - position_[c.a];
                const float len = length(d);
                if (len < 1e-9f) continue;
                const float C = len - c.rest;
                const float alphaTilde = c.compliance * invSubDt2;
                const float dLambda = (-C - alphaTilde * c.lambda) / (wSum + alphaTilde);
                c.lambda += dLambda;
                const Vec3 n = d * (1.0f / len);
                position_[c.a] -= n * (dLambda * invMass_[c.a]);
                position_[c.b] += n * (dLambda * invMass_[c.b]);
            }
        };
        for (int it = 0; it < iters; ++it) {
            for (DistanceConstraint& c : stretch_) c.lambda = 0.0f;
            for (DistanceConstraint& c : bending_) c.lambda = 0.0f;
            // Rest-shape (gravity) constraint: pull every free vertex towards its
            // sag-free rest position with a stiff compliance. This is what conserves
            // the hairstyle instead of letting the mesh collapse.
            if (params.sagFreeBlend > 0.0f) {
                // Shape-memory constraint, expressed as a rate so that its strength does
                // not depend on the number of substeps (XPBD-style dt weighting).
                const float rate = clampf(params.shapeMemory * subDt * params.sagFreeBlend, 0.0f, 1.0f);
                for (size_t k = 0; k < position_.size(); ++k) {
                    if (invMass_[k] == 0.0f) continue;
                    position_[k] += (restShape_[k] - position_[k]) * rate;
                }
            }
            project(stretch_);
            project(bending_);
            // --- collision against the head (sphere approximation) ---
            if (params.collideHead) {
                const float r = params.headRadius;
                for (size_t k = 0; k < position_.size(); ++k) {
                    if (invMass_[k] == 0.0f) continue;
                    const Vec3 rel = position_[k] - params.headCenter;
                    const float d = length(rel);
                    if (d < r && d > 1e-6f) {
                        const Vec3 n = rel * (1.0f / d);
                        position_[k] = params.headCenter + n * r;
                        // Tangential friction.
                        const Vec3 dv = position_[k] - previous_[k];
                        const Vec3 tangential = dv - n * dot(dv, n);
                        position_[k] -= tangential * params.friction;
                    }
                }
            }
        }

        // --- velocity update (XPBD) ---
        float maxV = 0.0f, kinetic = 0.0f;
        for (size_t k = 0; k < position_.size(); ++k) {
            if (invMass_[k] == 0.0f) { velocity_[k] = Vec3(0, 0, 0); continue; }
            velocity_[k] = (position_[k] - previous_[k]) * (1.0f / subDt);
            const float v = length(velocity_[k]);
            maxV = std::max(maxV, v);
            kinetic += 0.5f * v * v;
        }
        stats_.maxVelocity = maxV;
        stats_.totalKinetic = kinetic;
        stats_.substeps = params.substeps;
        // Hair that moves faster than 60 m/s or gains energy without bound means the
        // solver diverged; report it instead of silently rendering garbage.
        stats_.stable = maxV < 60.0f;
    }

    // --- write back to the hair mesh ---
    float stretchErr = 0.0f;
    for (const DistanceConstraint& c : stretch_) {
        const float len = length(position_[c.b] - position_[c.a]);
        stretchErr += std::fabs(len - c.rest) / std::max(c.rest, 1e-6f);
    }
    stats_.meanStretchError = stretch_.empty() ? 0.0f
                                               : stretchErr / static_cast<float>(stretch_.size());
    for (uint32_t l = 0; l < numLayers_; ++l)
        for (uint32_t v = 0; v < numVertices_; ++v)
            mesh->layerVertexMutable(l, v) = position_[static_cast<size_t>(l) * numVertices_ + v];
}

void HairMeshSimulator::step(HairMesh* mesh, float dt, const SimParams& params) {
    solve(mesh, params, dt, true);
}

}  // namespace silk
