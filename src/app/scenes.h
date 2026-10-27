// Scene construction shared by the CLI (main.cpp), the Vulkan screenshot mode and the
// Qt6 editor, so every entry point renders exactly the same content.
#pragma once

#include <cstdint>
#include <memory>

#include "silkcrowd/renderer.h"

namespace silk {

// The ten Figure 10 styles. |strandCount| is the total root strand budget of the model.
Scene buildHeadScene(int styleIndex, uint32_t strandCount, uint32_t seed = 1);

// Figure 1 crowd: |count| unique heads, each with a unique style, unique styling
// parameters and its own five texture sets (no instancing of hair data).
Scene buildCrowdScene(int count);

// Figure 9: |count| teapots, each with a hair mesh and 50k strands.
Scene buildTeapotScene(int count);

// Compact style spec used by the crowd (keeps every model in the 13-21 KB range).
HairStyleSpec crowdStyleSpec(int i, uint32_t seed);

Camera headCamera(int width, int height, float fovYDeg = 34.0f);
Camera crowdCamera(int width, int height);
Camera teapotCamera(int width, int height);

// Picks the camera that belongs to a scene name ("head", "crowd", "teapot").
Camera cameraForScene(const std::string& scene, int width, int height);

}  // namespace silk
