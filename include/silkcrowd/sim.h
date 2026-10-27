// Hair mesh simulation (paper Section 4): extended position-based dynamics
// [Macklin et al. 2016] with the force models of [Wu and Yuksel 2016] and sag-free
// initialization [Hsu et al. 2022].
//
// Only the hair *mesh* is simulated (a few hundred vertices); the strands are never
// simulated or stored -- they are regenerated from the five textures every frame, and
// the hair mesh texture / u-texture / v-texture are the only data that is updated
// (Table 1).
#pragma once

#include <vector>

#include "silkcrowd/hair_mesh.h"
#include "silkcrowd/math.h"

namespace silk {

struct SimParams {
    float gravity = -9.81f;
    int substeps = 8;
    int iterations = 2;
    // XPBD compliance values (inverse stiffness, m/N).
    float stretchCompliance = 1.0e-7f;
    float bendingCompliance = 2.0e-6f;
    // Gravity is the dominant force in hair; a fraction of it is applied as a restoring
    // "gravity constraint" so that the rest shape is the hair's hanging shape
    // (Wu and Yuksel 2016 / sag-free initialization, Hsu et al. 2022).
    float sagFreeBlend = 1.0f;
    // Shape memory: rate (1/s) at which a vertex is pulled back towards its sag-free rest
    // position. This is the hair mesh's equivalent of bending stiffness and is what lets
    // wind visibly move the hairstyle while still recovering its shape.
    float shapeMemory = 2.5f;
    // Damping / air drag.
    float linearDamping = 1.2f;
    float drag = 0.35f;
    // Wind (turbulent, time varying).
    Vec3 windDirection{0.85f, 0.05f, 0.52f};
    float windStrength = 0.0f;
    float windTurbulence = 1.7f;
    float windFrequency = 0.9f;
    // Collision against the head.
    bool collideHead = true;
    Vec3 headCenter{0.0f, 0.02f, 0.0f};
    float headRadius = 0.098f;
    float friction = 0.35f;
    // Mouse interaction: an object-space point that drags hair (Section B).
    bool dragging = false;
    Vec3 dragPoint{0.0f, 0.1f, 0.0f};
    Vec3 dragDelta{0.0f, 0.0f, 0.0f};
    float dragRadius = 0.05f;
};

struct SimStats {
    float maxVelocity = 0.0f;
    float meanStretchError = 0.0f;
    float totalKinetic = 0.0f;
    int substeps = 0;
    bool stable = true;
    std::string toString() const;
};

// Simulates the extrusion layers of a hair mesh with the root layer pinned to the scalp.
class HairMeshSimulator {
public:
    void init(const HairMesh& restMesh, const SimParams& params);

    // Advances the simulation by dt seconds and writes the deformed mesh back.
    void step(HairMesh* mesh, float dt, const SimParams& params);

    // Sag-free initialization: relaxes the rest shape until the mesh is in equilibrium
    // under gravity, so that simulation starts without a visible drop.
    void computeSagFreeRest(const HairMesh& mesh, const SimParams& params, int iterations = 200);

    const SimStats& stats() const { return stats_; }
    const std::vector<Vec3>& restShape() const { return restShape_; }

private:
    void solve(HairMesh* mesh, const SimParams& params, float dt, bool withForces);

    uint32_t numVertices_ = 0, numLayers_ = 0;
    std::vector<Vec3> restShape_;   // sag-free rest positions (layer * nv + v)
    std::vector<Vec3> position_;
    std::vector<Vec3> previous_;
    std::vector<Vec3> velocity_;
    std::vector<float> invMass_;    // 0 for pinned (root layer)
    struct DistanceConstraint {
        uint32_t a, b;
        float rest = 0.0f;
        float compliance = 0.0f;
        float lambda = 0.0f;
    };
    std::vector<DistanceConstraint> stretch_;
    std::vector<DistanceConstraint> bending_;
    SimStats stats_;
    float time_ = 0.0f;
};

}  // namespace silk
