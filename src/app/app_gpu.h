// Entry points shared by the CLI (main.cpp) and the Qt6 editor (qt_app.cpp).
#pragma once

#include <string>

namespace silk {

// Everything the GPU paths need from the command line.
struct GpuAppOptions {
    std::string screenshotPath;   // empty: no readback
    std::string scene = "head";   // head | crowd | teapot
    int width = 1280;
    int height = 720;
    int crowd = 100;              // crowd / teapot instance count
    int style = 0;                // Figure 10 preset index
    int strands = 40000;          // root strands for the single-head scene
    int msaa = 1;                 // 1, 2, 4 or 8
    int frames = 1;               // frames to render before the screenshot
    bool taa = true;              // temporal anti-aliasing
    bool lod = true;
    bool useMeshShaders = true;   // false forces the compute fallback
    bool validation = false;
    bool shadows = true;
    bool headless = true;
    float exposure = 1.05f;
    float wind = 0.0f;            // m/s, drives the XPBD simulation
};

// Headless Vulkan render. Returns 0 on success; on failure |error| explains why so the
// caller can fall back to the software renderer.
int runGpuScreenshot(const GpuAppOptions& options, std::string* error);

// Qt6 QVulkanWindow editor. Returns the process exit code; returns 2 when this build has
// no Qt6 support (the CLI then prints the help text).
int runGpuWindow(int argc, char** argv);

}  // namespace silk
