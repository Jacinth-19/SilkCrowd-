#include "scenes.h"

#include <cmath>
#include <string>

#include "silkcrowd/assets.h"

namespace silk {
namespace {
constexpr float kDeg2Rad = 3.14159265358979323846f / 180.0f;
}  // namespace

Scene buildHeadScene(int styleIndex, uint32_t strandCount, uint32_t seed) {
    Scene scene;
    HeadModel head;
    scene.head = head.headMesh();
    scene.body = head.bodyMesh();
    scene.eyes = head.eyesMesh();

    HairStyleSpec spec = HairStyleSpec::fromFigure10(styleIndex, seed);
    spec.totalStrands = strandCount;
    spec.styling = StylingParams::fromPreset(styleIndex, seed);
    HairMesh mesh = buildHairMesh(head, spec);
    auto inst = makeHairInstance(mesh, spec.styling, seed, TexelFormat::Float16);
    inst->baseRadius = 0.00042f;
    scene.hair.push_back(inst);
    return scene;
}

// Compact hair meshes for the crowd (13-21 KB of textures each, as in Figure 1).
HairStyleSpec crowdStyleSpec(int i, uint32_t seed) {
    HairStyleSpec s = HairStyleSpec::fromFigure10(i % 10, seed);
    // The crowd models of Figure 1 are small: they only need 13-21 KB of textures.
    // Root face counts of 35-48 keep every model at 14-19 KB of textures (Figure 1's
    // 13-21 KB range) while still giving the model a recognisable hairstyle silhouette.
    const int gu[] = {8, 9, 8, 9, 8, 9, 9, 8, 8, 9};
    const int gv[] = {6, 6, 7, 7, 6, 7, 6, 7, 6, 7};
    s.gridU = gu[i % 10];
    s.gridV = gv[i % 10];
    s.numLayers = 3;
    s.styling = StylingParams::fromPreset(i % 10, seed + 17u);
    return s;
}

Scene buildCrowdScene(int count) {
    Scene scene;
    HeadModel head;
    // One shared head/body mesh, drawn once per crowd instance (see the renderers), so that
    // every character in Figure 1's crowd has a face under its own hairstyle.
    scene.head = head.headMesh();
    scene.body = head.bodyMesh();
    scene.eyes = head.eyesMesh();

    const int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count))));
    const float spacing = 0.56f;
    for (int i = 0; i < count; ++i) {
        const int cx = i % cols, cy = i / cols;
        const float x = (cx - (cols - 1) * 0.5f) * spacing;
        const float z = (cy - (cols - 1) * 0.5f) * spacing;
        HairStyleSpec spec = crowdStyleSpec(i, 100u + i);
        spec.totalStrands = 100000;
        HairMesh mesh = buildHairMesh(head, spec);
        auto inst = makeHairInstance(mesh, spec.styling, 500u + i, TexelFormat::Float16);
        const float scale = 0.92f + 0.16f * hashToFloat(pcgHash(static_cast<uint32_t>(i) * 7u + 3u));
        const float yaw = (hashToFloat(pcgHash(static_cast<uint32_t>(i) * 13u + 5u)) - 0.5f) * 0.7f;
        inst->transform = Mat4::translate(Vec3(x, 0.0f, z)) * Mat4::rotateY(yaw) * Mat4::scale(Vec3(scale));
        scene.hair.push_back(inst);
    }
    return scene;
}

Scene buildTeapotScene(int count) {
    Scene scene;
    scene.useTeapots = true;
    scene.teapot = buildTeapot();
    const int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count))));
    const float spacing = 0.30f;
    for (int i = 0; i < count; ++i) {
        const int cx = i % cols, cy = i / cols;
        const float x = (cx - (cols - 1) * 0.5f) * spacing;
        const float z = (cy - (cols - 1) * 0.5f) * spacing;
        // Figure 9: each teapot grows a short hair layer on its lid (about a third of the
        // pot's height), so the body stays readable at crowd distance.
        HairMesh mesh = buildTeapotHairMesh(6, 5, 3, 0.030f + 0.010f * (i % 3), 900u + i);
        StylingParams sp = StylingParams::fromPreset(i % 10, 900u + i);
        auto inst = makeHairInstance(mesh, sp, 900u + i, TexelFormat::Float16);
        inst->transform = Mat4::translate(Vec3(x, 0.0f, z)) * Mat4::rotateY(0.4f * i);
        inst->baseRadius = 0.00035f;
        scene.hair.push_back(inst);
    }
    return scene;
}

Camera headCamera(int width, int height, float fovYDeg) {
    Camera c;
    c.fovY = deg2rad(fovYDeg);
    c.aspect = static_cast<float>(width) / static_cast<float>(height);
    c.position = Vec3(0.135f, 0.055f, 0.52f);
    c.target = Vec3(0.0f, 0.015f, 0.0f);
    c.nearZ = 0.05f;
    return c;
}

Camera crowdCamera(int width, int height) {
    // Frames the whole 10x10 crowd of Figure 1 (spacing 0.56 m) from slightly above eye
    // level, so the grid reads as a crowd rather than a row of heads.
    Camera c;
    c.fovY = deg2rad(42.0f);
    c.aspect = static_cast<float>(width) / static_cast<float>(height);
    c.position = Vec3(0.0f, 2.55f, 6.10f);
    c.target = Vec3(0.0f, 0.42f, 0.0f);
    c.nearZ = 0.05f;
    return c;
}

Camera teapotCamera(int width, int height) {
    // 200 teapots on a 15x14 grid with 0.30 m spacing (Figure 9).
    Camera c;
    c.fovY = deg2rad(40.0f);
    c.aspect = static_cast<float>(width) / static_cast<float>(height);
    c.position = Vec3(0.0f, 1.85f, 4.30f);
    c.target = Vec3(0.0f, 0.22f, 0.0f);
    c.nearZ = 0.05f;
    return c;
}

Camera cameraForScene(const std::string& scene, int width, int height) {
    if (scene == "crowd") return crowdCamera(width, height);
    if (scene == "teapot") return teapotCamera(width, height);
    return headCamera(width, height);
}

}  // namespace silk
