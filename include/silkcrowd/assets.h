// Asset pipeline: head model, scalp parameterization and hair mesh presets.
//
// The reference head is generated procedurally so the repository needs no binary
// dependencies; assets/heads/female_head.gltf (produced by tools/make_assets) uses the
// same scalp parameterization and can be swapped in without touching the renderer.
#pragma once

#include <string>
#include <vector>

#include "silkcrowd/hair_mesh.h"
#include "silkcrowd/styling.h"

namespace silk {

// Simple indexed triangle mesh used by the software renderer and the GLTF loader.
struct Mesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::vector<uint32_t> indices;  // triangle list
    std::vector<uint32_t> materialIds;
    std::string name;

    void computeNormals();
    size_t triangleCount() const { return indices.size() / 3; }
};

// Procedural female head: cranium, jaw, nose, ears, neck, shoulders and eye sockets.
// The scalp region is what the hair mesh attaches to.
class HeadModel {
public:
    HeadModel();

    // Surface point / normal for the scalp parameterization (theta = azimuth,
    // phi = polar distance from the crown).
    Vec3 surfacePoint(float theta, float phi) const;
    Vec3 surfaceNormal(float theta, float phi) const;
    // Hairline polar angle for a given azimuth: larger at the back, smaller at the
    // forehead, which is what makes the hairline read as a hairline.
    float hairlinePhi(float theta) const;

    const Mesh& headMesh() const { return head_; }
    const Mesh& eyesMesh() const { return eyes_; }
    const Mesh& bodyMesh() const { return body_; }
    Vec3 center() const { return center_; }
    float radius() const { return radius_; }

    // "Scalp pieces": the hair mesh root layer is a rectangular patch of the scalp
    // (theta in [theta0,theta1], phi in [phiMin, hairlinePhi]), mapped onto the root
    // slice lattice (Section 3.1: scalps that are not a suitable lattice are split into
    // pieces, each with its own patch).
    struct ScalpPatch {
        float theta0 = -kPi, theta1 = kPi;
        float phiMin = 0.05f;
        float thetaCenter = 0.0f;
    };
    ScalpPatch defaultPatch() const;

private:
    void buildHead();
    void buildBody();
    void buildEyes();
    Mesh head_, body_, eyes_;
    Vec3 center_{0.0f, 0.02f, 0.0f};
    float radius_ = 0.13f;
};

// One of the ten hairstyles of Figure 10.
struct HairStyleSpec {
    const char* name = "style";
    int gridU = 16;        // root faces around the head
    int gridV = 12;        // root faces from crown to hairline
    int numLayers = 4;     // extrusion layers (root + numLayers-1)
    float length = 0.20f;  // world-space hair length
    float fall = 0.85f;    // 0 = sticks straight out, 1 = hangs down
    float volume = 0.012f; // outward puffing
    float backBias = 0.35f;// longer hair towards the back of the head
    float hairlineDrop = 0.0f;  // extra hairline extension (bangs vs. slicked back)
    uint32_t totalStrands = 100000;
    uint32_t seed = 1;
    StylingParams styling;

    // Figure 10 resolutions (vertices / reported size) for the ten reference styles.
    static HairStyleSpec fromFigure10(int index, uint32_t seed);
    static int numPresets() { return 10; }
};

// Builds the hair mesh (root layer on the scalp + extrusion layers) for a style.
HairMesh buildHairMesh(const HeadModel& head, const HairStyleSpec& style);

// Uniform scalp patch hair mesh used by the teapot test (T key).
HairMesh buildTeapotHairMesh(int gridU, int gridV, int numLayers, float length, uint32_t seed);

// Procedural teapot proxy (surface of revolution body + handle + spout).
Mesh buildTeapot();

// Bounding-sphere helper for LOD (Eq. 7 uses the bundle's scalp face size).
void bundleScalpMetrics(const HairMesh& mesh, std::vector<float>* bundleSize,
                        std::vector<Vec3>* bundleCenter);

}  // namespace silk
