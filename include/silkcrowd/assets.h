// Asset pipeline: head model, scalp parameterization and hair mesh presets.
//
// Two heads are supported by the same interface:
//
//   * `assets/heads/head_scan.glb` - a photogrammetry bust (Infinite-Realities scan,
//     CC-BY 3.0, see assets/heads/LICENSE-head_scan.txt). This is what the app loads by
//     default: real cranium, brow, lids, lips, ears, neck and shoulders, and a 1K
//     4096^-derived skin texture with pores. `tools/prep_head.py` repacks and fits it.
//   * the procedural head below - a parametric surface that needs no binary assets, used
//     when no scan is present (unit tests, `--procedural-head`, CI on a bare checkout).
//
// Both answer the same scalp parameterization (theta = azimuth with 0 at the face,
// phi = polar angle from the crown), which is what the hair mesh root layer, the hairline
// and Section 3.1's scalp patches are defined in. With a scan loaded, surfacePoint() is
// evaluated from the scan itself (a radial depth map around the head centre), so hair
// roots sit on the real skull rather than on an idealised ellipsoid.
#pragma once

#include <string>
#include <vector>

#include "silkcrowd/hair_mesh.h"
#include "silkcrowd/image.h"
#include "silkcrowd/styling.h"

namespace silk {

// Simple indexed triangle mesh used by the software renderer and the GLTF loader.
struct Mesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::vector<uint32_t> indices;  // triangle list
    std::vector<uint32_t> materialIds;
    // Optional per-vertex "this is scalp, not face" factor (1 above the hairline). Filled
    // for scanned heads so the textured skin can go dark under the parting, exactly as the
    // procedural head's FaceMasks::scalp does.
    std::vector<float> scalpMask;
    std::string name;

    void computeNormals();
    size_t triangleCount() const { return indices.size() / 3; }
};

// Human head: either the scanned bust or the procedural surface. The scalp region is what
// the hair mesh attaches to.
class HeadModel {
public:
    HeadModel();

    // Loads a scanned bust (GLB with POSITION/NORMAL/TEXCOORD_0 and a base-colour
    // texture) and takes over the surface. Returns false and leaves the procedural head in
    // place if the file is missing or unsupported, so a bare checkout still renders.
    bool loadScan(const std::string& glbPath, std::string* error = nullptr);
    bool isScan() const { return scan_; }
    // Linear-light skin albedo of the scan (empty when procedural).
    const Image& skinAlbedo() const { return skinAlbedo_; }
    // Hairline mask in the head's own space: 1 on the scalp, 0 on the face. Used to keep
    // the scanned scalp dark under the parting instead of showing a bald head.
    float scalpMask(const Vec3& localPos) const;

    // Surface point / normal for the scalp parameterization (theta = azimuth,
    // phi = polar distance from the crown).
    Vec3 surfacePoint(float theta, float phi) const;
    Vec3 surfaceNormal(float theta, float phi) const;
    // Hairline polar angle for a given azimuth: larger at the back, smaller at the
    // forehead, which is what makes the hairline read as a hairline.
    float hairlinePhi(float theta) const;

    const Mesh& headMesh() const { return head_; }
    // Coarse copy of the same surface for the crowd, where each face is a few pixels tall.
    const Mesh& headMeshLow() const { return headLow_; }
    // Eyeballs. The scan has closed lids and a painted lash line, so this is empty then.
    const Mesh& eyesMesh() const { return eyes_; }
    // Neck and shoulders. Empty when the scan already contains them.
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
    Vec3 surfacePointProcedural(float theta, float phi) const;
    void buildHead();
    void buildHeadLow();
    void buildBody();
    void buildEyes();
    // Radial depth map of the scan around center_ (theta x phi): the hair mesh is fitted
    // against the real skull by evaluating it, not by re-deriving an ellipsoid.
    void buildScanDepthMap();
    float scanRadius(float theta, float phi) const;

    Mesh head_, headLow_, body_, eyes_;
    Vec3 center_{0.0f, 0.02f, 0.0f};
    float radius_ = 0.13f;

    bool scan_ = false;
    Image skinAlbedo_;
    int depthU_ = 0, depthV_ = 0;          // grid resolution (theta, phi)
    std::vector<float> depth_;             // radius per cell, 0 = no scan surface there
};

// Analytic face masks for shading, evaluated from the same anatomy as the geometry: a
// point on the head is mapped back to (theta, phi) and the landmark Gaussians are reused,
// so the brows, the lips, the lash line and the creases stay glued to the surface no
// matter how the head is tessellated.
struct FaceMasks {
    float brow = 0.0f;        // eyebrow
    float lipUpper = 0.0f;    // vermilion of the upper lip
    float lipLower = 0.0f;
    float mouthLine = 0.0f;   // the dark seam between the lips
    float lashUpper = 0.0f;   // lash line along the upper lid
    float lashLower = 0.0f;
    float nostril = 0.0f;
    float noseBridge = 0.0f;  // the dorsum of the nose, between the brows and the tip
    float noseSide = 0.0f;    // the side walls and the alar creases beside them
    float blush = 0.0f;       // cheeks, nose tip, ears
    float ao = 0.0f;          // sockets, nasolabial folds, under the chin, lip seam
    float scalp = 0.0f;       // 1 above the hairline: hair, not skin, is what shows
                              // through if a lock separates and the scalp peeks out
};
// |localPos| is in the head's own space (world position minus HeadModel::center()).
FaceMasks headFaceMasks(const Vec3& localPos);

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

// ---------------------------------------------------------------------------
//  Asset paths
// ---------------------------------------------------------------------------
// Resolves a path relative to the repository's `assets/` directory: the
// SILKCROWD_ASSET_DIR environment variable wins, then ./assets, ../assets, ../../assets and
// finally the source directory recorded by CMake. Returns "" when nothing is found.
std::string assetPath(const std::string& relative);

// The scanned head (`assets/heads/head_scan.glb`). Empty environment-independent default;
// setHeadScanAsset("") forces the procedural head (the CLI's --procedural-head).
std::string headScanAsset();
void setHeadScanAsset(const std::string& path);

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
