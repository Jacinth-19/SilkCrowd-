// Vulkan renderer for SilkCrowd.
//
// Implements the paper's GPU pipeline:
//   * the five hair mesh textures (Table 1) as sampled 3D/2D images,
//   * a task/mesh shader pipeline (bundle -> strand parallelism, Section 3.2),
//   * a compute shader fallback with a CPU dispatch pre-pass ("Option A") for devices
//     without mesh shaders (Adreno, Intel UHD 630),
//   * PCF shadow map, HDR target, bloom, tone mapping, procedural sky and grid,
//   * ImGui statistics overlay.
#pragma once

#include <vulkan/vulkan.h>

#include <memory>
#include <string>
#include <vector>

#include "silkcrowd/renderer.h"
#include "../gpu/vulkan_context.h"

namespace silk {

struct GpuHairModel {
    // The five textures of Table 1.
    VkImage images[5] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory memories[5] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView views[5] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageLayout layouts[5] = {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t dims[5][3] = {};
    VkDeviceMemory bundleMemory = VK_NULL_HANDLE;
    VkBuffer bundleBuffer = VK_NULL_HANDLE;
    // set 0 = the five textures, set 1 = model uniforms, set 2 = bundle data + meta
    VkDescriptorSet descriptorSets[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformMemory = VK_NULL_HANDLE;
    void* uniformMapped = nullptr;
    uint64_t textureBytes = 0;
    uint32_t numFaces = 0;
    uint32_t maxStrandsPerBundle = 1;
    uint32_t chunksPerBundle = 1;
    const HairInstance* instance = nullptr;   // owned by the Scene
};

// Mirrors the layout in shaders/composite.frag (set 0, binding 2).
struct CompositeUniforms {
    float invViewProj[16];
    float cameraPos[4];
    float sunDirection[4];
    float sunColor[4];
    float skyZenith[4];
    float skyHorizon[4];
    float groundColor[4];
    float params[4];    // x = exposure, y = bloom strength, z = grid fade, w = draw sky/ground
    float viewport[4];  // x = width, y = height, z = 1/width, w = 1/height
};

// Mirrors shaders/taa.frag.
struct TaaUniforms {
    float prevViewProj[16];
    float invViewProj[16];
    float jitter[4];   // xy = current frame, zw = previous frame
    float params[4];   // x = blend weight, y = reset flag, z/w reserved
};

struct GpuStats {
    double frameMs = 0.0;
    double gpuHairMs = 0.0;
    uint64_t strandsRendered = 0;
    uint64_t strandsTotal = 0;
    uint64_t triangles = 0;
    uint64_t textureBytes = 0;
    int taskShaderCount = 0;
    int meshShaderRuns = 0;
    int vertexLodMin = 0, vertexLodMax = 0;
    std::string deviceName;
    std::string vulkanVersion;
    bool meshShaderPath = false;
    bool taa = false;
    int msaa = 1;
    int crowdCount = 0;
    std::string warning;
};

class HairGpuRenderer {
public:
    bool init(VulkanContext* context, uint32_t width, uint32_t height, bool enableValidation,
              std::string* error);
    void shutdown();

    // Uploads one hair model's five textures. Called once per model (no instancing of
    // hair data: every model owns its own textures, as in the paper's crowd test).
    bool uploadHairModel(const HairInstance& instance);

    // Re-uploads only the dynamic textures after the hair mesh moved (Table 1).
    bool updateDynamicTextures(const HairInstance& instance);

    void setScene(std::shared_ptr<Scene> scene);
    void resize(uint32_t width, uint32_t height);

    // Renders one frame into an offscreen HDR target and resolves to |outImage|.
    // If |screenshotPath| is non-empty the frame is also written to that PNG.
    bool renderFrame(const Camera& camera, const RenderOptions& options,
                     const std::string& screenshotPath);

    // Renders the frame directly into an externally owned image (the Qt6 QVulkanWindow
    // swapchain image). The image is transitioned to COLOR_ATTACHMENT_OPTIMAL and left in
    // PRESENT_SRC_KHR, ready to be presented by the caller.
    bool renderToSwapchainImage(VkImage image, uint32_t width, uint32_t height,
                                const Camera& camera, const RenderOptions& options);

    // The view/format of the image that renderToSwapchainImage() writes into. Must be set
    // whenever the swapchain is (re)created.
    void setCompositeTarget(VkImageView targetView, VkFormat targetFormat);

    // Forces the compute-shader fallback even on hardware with mesh shaders (used to
    // exercise the Adreno / Intel UHD 630 code path on a desktop GPU).
    void forceComputeFallback() { forceCompute_ = true; }

    const GpuStats& stats() const { return stats_; }
    bool meshShaderPath() const { return hasMeshShaders_; }

    // Auto-tuning for the detected hardware tier (Section B).
    void applyHardwarePolicy(const DeviceInfo& info, RenderOptions* options) const;

private:
    bool createPipelines(std::string* error);
    // The head/hair pipelines render into the HDR pass and therefore carry the current
    // sample count; they are rebuilt when the 8x MSAA toggle changes.
    bool createMainPassPipelines(std::string* error);
    void destroyMainPassPipelines();
    bool setSampleCount(int wanted);
    bool createMsaaTargets();
    int msaaSamples_ = 1;
    bool createGraphicsPipeline(VkPipeline* out, const char* task, const char* mesh,
                                const char* vertex, const char* fragment, bool hairBlend,
                                bool depthOnly, std::string* error);
    void uploadTextureImage(GpuHairModel* model, int index, const float* data, uint32_t w,
                            uint32_t h, uint32_t d, uint32_t components);
    void writeModelUniforms(GpuHairModel* model, const HairInstance& instance,
                            const RenderOptions& options, const Camera& camera, uint32_t height);
    VkDescriptorSet allocateModelDescriptors(GpuHairModel* model);
    bool createDescriptorLayouts(std::string* error);
    bool createRenderTargets();
    bool createComputeFallbackBuffers(uint32_t maxStrands);
    bool writeSampleDescriptors();
    bool uploadBundleBuffer(GpuHairModel* model, const std::vector<Vec3>& bundleCenter,
                            const std::vector<float>& bundleSize,
                            const std::vector<float>& bundleLength,
                            const std::vector<uint32_t>& strandCounts, uint32_t seed);
    bool uploadStaticMeshes();
    void recordComputeFallback(VkCommandBuffer cmd, const RenderOptions& opt);
    void recordHairDraw(VkCommandBuffer cmd, const Camera& cam, const RenderOptions& opt,
                        bool shadowPass);
    void recordShadowPass(VkCommandBuffer cmd, const Camera& cam);
    void recordBloomPass(VkCommandBuffer cmd, const RenderOptions& opt);
    bool createTaaResources();
    void recordTaaPass(VkCommandBuffer cmd, bool reset);
    void advanceJitter();
    Mat4 jitteredProj(const Camera& cam) const;
    void bindPresentationImages(bool useResolved);
    void recordComposite(VkCommandBuffer cmd, VkImage target, uint32_t width, uint32_t height,
                         const RenderOptions& opt);
    bool prepareComputeWorkItems(const Camera& cam, const RenderOptions& opt,
                                 bool uploadGeometry);
    bool readbackToPng(const std::string& path);
    VkShaderModule loadShader(const char* path, std::string* error) const;

    VulkanContext* ctx_ = nullptr;
    uint32_t width_ = 0, height_ = 0;
    bool hasMeshShaders_ = false;
    bool forceCompute_ = false;
    bool validation_ = false;

    VkDescriptorSetLayout hairTextureLayout_ = VK_NULL_HANDLE;   // set 0: 5 textures
    VkDescriptorSetLayout modelLayout_ = VK_NULL_HANDLE;         // set 1: model uniforms
    VkDescriptorSetLayout bundleLayout_ = VK_NULL_HANDLE;        // set 2: bundle data + meta
    VkDescriptorSetLayout sampleLayout_ = VK_NULL_HANDLE;        // set 3: root samples etc.
    VkDescriptorSetLayout frameLayout_ = VK_NULL_HANDLE;          // set 0: camera/shadow/light
    VkDescriptorSet frameDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeLayout_ = VK_NULL_HANDLE;     // hdr + bloom + camera UBO
    VkDescriptorSetLayout bloomLayout_ = VK_NULL_HANDLE;         // single input image
    VkDescriptorSet compositeSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomSet_ = VK_NULL_HANDLE;
    VkBuffer compositeUniform_ = VK_NULL_HANDLE;
    VkDeviceMemory compositeUniformMemory_ = VK_NULL_HANDLE;
    void* compositeUniformMapped_ = nullptr;
    VkPipelineLayout hairPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;

    VkPipeline hairMeshPipeline_ = VK_NULL_HANDLE;
    VkPipeline hairComputePipeline_ = VK_NULL_HANDLE;
    VkPipeline hairDrawPipeline_ = VK_NULL_HANDLE;     // vertex path for the compute output
    VkPipeline hairShadowDrawPipeline_ = VK_NULL_HANDLE;  // depth-only variant of the above
    VkPipeline shadowHairPipeline_ = VK_NULL_HANDLE;
    VkPipeline headPipeline_ = VK_NULL_HANDLE;
    VkPipeline compositePipeline_ = VK_NULL_HANDLE;      // BGRA8 (swapchain)
    VkPipeline compositePipelineRgba_ = VK_NULL_HANDLE;  // RGBA8 (offscreen)
    VkPipeline bloomPipeline_ = VK_NULL_HANDLE;
    VkPipeline taaPipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout taaLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet taaSet_ = VK_NULL_HANDLE;
    VkBuffer taaUniform_ = VK_NULL_HANDLE;
    VkDeviceMemory taaUniformMemory_ = VK_NULL_HANDLE;
    void* taaUniformMapped_ = nullptr;
    VkImage resolvedImage_ = VK_NULL_HANDLE;      // TAA output
    VkDeviceMemory resolvedMemory_ = VK_NULL_HANDLE;
    VkImageView resolvedView_ = VK_NULL_HANDLE;
    VkImage historyImage_ = VK_NULL_HANDLE;       // previous resolved frame
    VkDeviceMemory historyMemory_ = VK_NULL_HANDLE;
    VkImageView historyView_ = VK_NULL_HANDLE;
    bool taaEnabled_ = true;
    bool taaHistoryValid_ = false;
    uint32_t frameCounter_ = 0;
    Mat4 lightViewProj_;   // the shadow pass's orthographic light matrix
    float prevViewProj_[16] = {0};
    float currentInvViewProj_[16] = {0};
    float currentViewProj_[16] = {0};
    float jitter_[2] = {0.0f, 0.0f};
    float prevJitter_[2] = {0.0f, 0.0f};
    bool usingResolved_ = false;
    bool taaResetRequested_ = true;
    uint32_t resolvedWidth_ = 0, resolvedHeight_ = 0;

    // Targets
    VkImage hdrImage_ = VK_NULL_HANDLE;
    VkDeviceMemory hdrMemory_ = VK_NULL_HANDLE;
    VkImageView hdrView_ = VK_NULL_HANDLE;
    // Multisampled attachments for the main pass (only when msaaSamples_ > 1); the HDR
    // image stays single-sampled and receives the resolve.
    VkImage msaaImage_ = VK_NULL_HANDLE;
    VkDeviceMemory msaaMemory_ = VK_NULL_HANDLE;
    VkImageView msaaView_ = VK_NULL_HANDLE;
    VkImage msaaDepthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory msaaDepthMemory_ = VK_NULL_HANDLE;
    VkImageView msaaDepthView_ = VK_NULL_HANDLE;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkImage ldrImage_ = VK_NULL_HANDLE;      // tonemapped output for screenshots
    VkDeviceMemory ldrMemory_ = VK_NULL_HANDLE;
    VkImageView ldrView_ = VK_NULL_HANDLE;
    VkImage bloomImage_ = VK_NULL_HANDLE;
    VkDeviceMemory bloomMemory_ = VK_NULL_HANDLE;
    VkImageView bloomView_ = VK_NULL_HANDLE;
    VkImage shadowImage_ = VK_NULL_HANDLE;
    VkDeviceMemory shadowMemory_ = VK_NULL_HANDLE;
    VkImageView shadowView_ = VK_NULL_HANDLE;
    VkSampler samplerLinear_ = VK_NULL_HANDLE;
    VkSampler samplerShadow_ = VK_NULL_HANDLE;
    VkBuffer readbackBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory readbackMemory_ = VK_NULL_HANDLE;
    void* readbackMapped_ = nullptr;
    VkBuffer cameraUniform_ = VK_NULL_HANDLE;
    VkDeviceMemory cameraUniformMemory_ = VK_NULL_HANDLE;
    void* cameraUniformMapped_ = nullptr;
    VkBuffer lightUniform_ = VK_NULL_HANDLE;
    VkDeviceMemory lightUniformMemory_ = VK_NULL_HANDLE;
    void* lightUniformMapped_ = nullptr;

    // Root sample tables and compute fallback buffers.
    VkBuffer sampleBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sampleMemory_ = VK_NULL_HANDLE;
    VkBuffer sampleTriangleBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sampleTriangleMemory_ = VK_NULL_HANDLE;
    VkBuffer workItemBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory workItemMemory_ = VK_NULL_HANDLE;
    VkBuffer strandVertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory strandVertexMemory_ = VK_NULL_HANDLE;
    VkBuffer strandTangentBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory strandTangentMemory_ = VK_NULL_HANDLE;
    VkBuffer strandIndexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory strandIndexMemory_ = VK_NULL_HANDLE;
    VkBuffer strandMetaBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory strandMetaMemory_ = VK_NULL_HANDLE;
    VkDescriptorSet sampleDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet headDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet compositeDescriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout headPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout compositePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout drawPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout taaPipelineLayout_ = VK_NULL_HANDLE;
    VkImageView compositeTargetView_ = VK_NULL_HANDLE;
    VkFormat compositeTargetFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
    uint32_t computeWorkItems_ = 0;
    uint32_t computeIndexCount_ = 0;

    // Static geometry (head / body / teapot) uploads.
    struct StaticMeshGpu {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        uint32_t indexCount = 0;
    };
    StaticMeshGpu headMesh_;
    StaticMeshGpu bodyMesh_;
    StaticMeshGpu teapotMesh_;
    StaticMeshGpu headMeshIndex_;
    StaticMeshGpu bodyMeshIndex_;
    StaticMeshGpu teapotMeshIndex_;
    uint32_t maxStrands_ = 0;
    uint32_t computeMaxSegments_ = 8;
    VkDeviceSize computeVertexCapacity_ = 0;

    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkSemaphore renderDone_ = VK_NULL_HANDLE;
    VkFence frameFence_ = VK_NULL_HANDLE;

    std::shared_ptr<Scene> scene_;
    std::vector<GpuHairModel> models_;
    GpuStats stats_;
};

}  // namespace silk
