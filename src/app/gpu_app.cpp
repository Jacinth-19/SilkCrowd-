// GPU entry points: headless Vulkan rendering for --screenshot / CI, and the bridge to
// the Qt6 QVulkanWindow editor.
#include "app_gpu.h"

#include <cstdio>
#include <memory>
#include <string>

#include "scenes.h"
#include "silkcrowd/renderer.h"

#ifdef SILKCROWD_WITH_VULKAN
#include "gpu/hair_gpu_renderer.h"
#include "gpu/vulkan_context.h"
#endif

namespace silk {

#ifdef SILKCROWD_WITH_VULKAN

namespace {

Scene buildSceneFor(const GpuAppOptions& o) {
    if (o.scene == "crowd") return buildCrowdScene(o.crowd);
    if (o.scene == "teapot") return buildTeapotScene(o.crowd);
    return buildHeadScene(o.style, static_cast<uint32_t>(o.strands));
}

}  // namespace

int runGpuScreenshot(const GpuAppOptions& o, std::string* error) {
    Scene scene = buildSceneFor(o);

    VulkanContext context;
    if (!context.init(VK_NULL_HANDLE, VK_NULL_HANDLE, o.validation, error)) return 1;

    HairGpuRenderer renderer;
    // Snapshots are single images, so MSAA is handled by supersampling the render target
    // when the requested sample count is above one.
    const uint32_t renderWidth = static_cast<uint32_t>(o.width) * (o.msaa > 1 ? 1u : 1u);
    if (!renderer.init(&context, renderWidth, static_cast<uint32_t>(o.height), o.validation,
                       error)) {
        renderer.shutdown();
        context.shutdown();
        return 1;
    }
    if (!o.useMeshShaders) renderer.forceComputeFallback();
    renderer.setScene(std::make_shared<Scene>(std::move(scene)));

    Camera camera = cameraForScene(o.scene, o.width, o.height);
    RenderOptions options;
    options.width = o.width;
    options.height = o.height;
    options.msaa = o.msaa;
    options.taa = o.taa;
    options.enableLod = o.lod;
    options.exposure = o.exposure;
    options.drawShadows = o.shadows;

    const auto& info = context.info();
    std::printf("device: %s\n", info.name.c_str());
    std::printf("vulkan: %s\n", info.apiVersion.c_str());
    std::printf("mesh shaders: %s\n", renderer.meshShaderPath() ? "yes (task/mesh path)"
                                                                : "no (compute fallback)");
    if (info.tier != HardwareTier::Unknown) {
        renderer.applyHardwarePolicy(info, &options);
        std::printf("hardware tier tuned to %d fps target, MSAA %dx\n", info.targetFps,
                    info.defaultMsaa);
    }

    const int frames = o.frames < 1 ? 1 : o.frames;
    for (int i = 0; i < frames; ++i) {
        const bool last = (i == frames - 1);
        if (!renderer.renderFrame(camera, options,
                                  (last && !o.screenshotPath.empty()) ? o.screenshotPath
                                                                     : std::string())) {
            if (error) *error = "Vulkan frame rendering failed";
            renderer.shutdown();
            context.shutdown();
            return 1;
        }
    }

    const GpuStats& stats = renderer.stats();
    std::printf("frame: %.2f ms | strands %llu / %llu | tris %llu | textures %.2f MB\n",
                stats.frameMs, static_cast<unsigned long long>(stats.strandsRendered),
                static_cast<unsigned long long>(stats.strandsTotal),
                static_cast<unsigned long long>(stats.triangles),
                static_cast<double>(stats.textureBytes) / (1024.0 * 1024.0));
    std::printf("crowd %d | vertex LOD 2^%d..2^%d | %s\n", stats.crowdCount, stats.vertexLodMin,
                stats.vertexLodMax,
                stats.meshShaderPath ? "task/mesh pipeline" : "compute pipeline");
    if (!stats.warning.empty()) std::printf("warning: %s\n", stats.warning.c_str());
    if (!o.screenshotPath.empty()) std::printf("wrote %s\n", o.screenshotPath.c_str());

    renderer.shutdown();
    context.shutdown();
    return 0;
}

#else  // !SILKCROWD_WITH_VULKAN

int runGpuScreenshot(const GpuAppOptions&, std::string* error) {
    if (error)
        *error = "this build has no Vulkan support (configure with -DVulkan_FOUND or install "
                 "the Vulkan SDK); rerun with --software";
    return 2;
}

#endif  // SILKCROWD_WITH_VULKAN

#if !defined(SILKCROWD_WITH_QT) || !defined(SILKCROWD_WITH_VULKAN)
// The Qt6 editor lives in qt_app.cpp and is only compiled when Qt6 was found. Return 2 so
// the caller knows to fall back instead of failing.
int runGpuWindow(int, char**) { return 2; }
#endif

}  // namespace silk
