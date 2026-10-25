// Software reference renderer.
//
// It rasterises the same primitives the GPU path produces: camera-facing triangle strips
// with 2*ns+2 vertices and 2*ns triangles per strand (Section 3.2), regenerated every
// frame from the five hair mesh textures -- no strand buffer is ever stored.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "silkcrowd/assets.h"
#include "silkcrowd/image.h"
#include "silkcrowd/lod.h"
#include "silkcrowd/root_placement.h"
#include "silkcrowd/strand_generator.h"

namespace silk {

struct Camera {
    Vec3 position{0.0f, 0.06f, 0.62f};
    Vec3 target{0.0f, 0.02f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovY = 0.72f;  // radians
    float nearZ = 0.01f, farZ = 60.0f;
    float aspect = 16.0f / 9.0f;

    Mat4 view() const { return Mat4::lookAt(position, target, up); }
    Mat4 proj() const { return Mat4::perspective(fovY, aspect, nearZ, farZ); }
    Vec3 forward() const { return normalize(target - position); }
};

struct RenderOptions {
    int width = 1280;
    int height = 720;
    int ssaa = 2;                // supersampling factor (software MSAA substitute)
    int msaa = 1;                // hardware MSAA sample count (Vulkan path)
    bool enableLod = true;       // Figure 8: LOD on/off comparison
    bool strandLod = true;
    bool vertexLod = true;
    bool applyStyling = true;
    FrameMode frameMode = FrameMode::PerVertexTextures;
    LodParams lod;
    float exposure = 1.05f;
    float hairAlpha = 0.42f;
    bool drawHair = true;
    bool drawHead = true;
    bool drawGround = true;
    bool drawSky = true;
    bool drawShadows = true;
    int fixedVertexExponent = -1;  // >=0 forces a vertex LOD level (Figure 6 tests)
    float lodBias = 1.0f;          // multiplies alpha of both LOD types
};

struct RenderStats {
    double frameMs = 0.0;
    double generationMs = 0.0;
    double rasterMs = 0.0;
    uint64_t strandsRendered = 0;
    uint64_t strandsTotal = 0;
    uint64_t strandVertices = 0;
    uint64_t triangles = 0;
    uint64_t headTriangles = 0;
    int minVertexExponent = 99;
    int maxVertexExponent = -1;
    float averageDetailLevel = 0.0f;
    int bundlesRendered = 0;
    std::string toString() const;
};

// One hair model instance (unique textures, no instancing of hair data).
struct HairInstance {
    HairMesh mesh;
    StylingParams styling;
    HairMeshTextureSet textures;
    Mat4 transform;
    Vec3 center{0, 0, 0};
    float radius = 0.2f;
    float baseRadius = 0.00045f;
    uint32_t seed = 0;
    std::vector<float> bundleSize;    // scalp face size (Eq. 7, strand LOD)
    std::vector<float> bundleLength;  // strand length per bundle (Eq. 7, vertex LOD)
    std::vector<Vec3> bundleCenter;
    std::vector<uint8_t> bundleVisible;  // optional per-bundle culling
    TexelFormat format = TexelFormat::Float16;
    uint64_t textureBytes = 0;
};

struct Scene {
    std::vector<std::shared_ptr<HairInstance>> hair;
    Mesh head;
    Mesh body;
    Mesh eyes;
    Mesh teapot;
    Vec3 sunDirection = normalize(Vec3(0.35f, 0.75f, 0.45f));
    Vec3 sunColor{1.0f, 0.93f, 0.82f};
    float sunIntensity = 2.2f;
    Vec3 skyZenith{0.20f, 0.36f, 0.62f};
    Vec3 skyHorizon{0.62f, 0.70f, 0.82f};
    Vec3 groundColor{0.16f, 0.15f, 0.14f};
    bool useTeapots = false;
};

// Builds the five textures for one hair model and caches the LOD helper data.
std::shared_ptr<HairInstance> makeHairInstance(const HairMesh& mesh, const StylingParams& styling,
                                               uint32_t seed = 0,
                                               TexelFormat format = TexelFormat::Float16);

class SoftwareRenderer {
public:
    void render(const Scene& scene, const Camera& camera, const RenderOptions& options,
                Image* out, RenderStats* stats);

private:
    struct Sample {
        Vec3 color{0, 0, 0};
        float depth = 1e30f;
    };
    std::vector<Sample> buffer_;
    std::vector<float> opaqueDepth_;  // depth of the opaque geometry only
    int bw_ = 0, bh_ = 0;
    SampleSet2D sampleSetQuad_;
    SampleSet2D sampleSetTriangle_;
};

}  // namespace silk
