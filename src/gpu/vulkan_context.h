// Vulkan 1.3 context for SilkCrowd.
//
// Creates the instance / device used by both the Qt6 editor (which passes the instance
// and surface of its QVulkanWindow) and the headless screenshot mode used by CI.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace silk {

// Hardware tiers used by the automatic quality control (Section B of the requirements).
enum class HardwareTier { Discrete, Integrated, Mobile, Software, Unknown };

struct DeviceInfo {
    std::string name;
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    HardwareTier tier = HardwareTier::Unknown;
    bool meshShaders = false;         // VK_EXT_mesh_shader
    bool dynamicRendering = false;
    bool descriptorIndexing = false;
    bool samplersAnisotropy = false;
    uint32_t maxTexture3DSize = 0;
    uint64_t memoryMB = 0;
    std::string apiVersion;

    // Recommended defaults for this class of hardware.
    int targetFps = 60;
    int defaultMsaa = 1;
    int defaultCrowd = 100;
    float lodBias = 1.0f;
};

// Automatic tier detection: Android/Adreno, Intel UHD, discrete, software.
DeviceInfo detectDevice(const VkPhysicalDeviceProperties& props, uint64_t memoryMB,
                        bool meshShaders);

class VulkanContext {
public:
    bool init(VkInstance instance, VkSurfaceKHR surface, bool enableValidation,
              std::string* error);

    // Adopts a device that someone else created (used by the Qt6 QVulkanWindow editor,
    // which owns its instance, device and swapchain).
    bool adopt(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
               VkQueue graphicsQueue, VkQueue presentQueue, uint32_t graphicsFamily,
               uint32_t presentFamily, VkCommandPool commandPool, bool meshShaders,
               std::string* error);
    void shutdown();

    VkInstance instance() const { return instance_; }
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    VkQueue graphicsQueue() const { return graphicsQueue_; }
    VkQueue presentQueue() const { return presentQueue_; }
    uint32_t graphicsFamily() const { return graphicsFamily_; }
    uint32_t presentFamily() const { return presentFamily_; }
    const DeviceInfo& info() const { return info_; }
    bool hasFragmentStoresAndAtomics() const { return fragmentStoresAndAtomics_; }
    bool ownsInstance() const { return ownsInstance_; }

    // ---- memory helpers ----
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags props, VkDeviceMemory* memory) const;
    void destroyBuffer(VkBuffer buffer, VkDeviceMemory memory) const;

    // ---- image helpers ----
    VkImage createImage2D(uint32_t width, uint32_t height, VkFormat format,
                          VkImageUsageFlags usage, VkImageAspectFlags aspect,
                          VkDeviceMemory* memory, VkImageView* view,
                          VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) const;
    VkImage createImage3D(uint32_t width, uint32_t height, uint32_t depth, VkFormat format,
                          VkImageUsageFlags usage, VkDeviceMemory* memory,
                          VkImageView* view) const;
    void destroyImage(VkImage image, VkDeviceMemory memory, VkImageView view) const;

    // ---- one-shot command helpers ----
    VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer cmd) const;
    void imageLayoutTransition(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
                               VkImageLayout to, VkImageAspectFlags aspect) const;
    void copyBufferToImage3D(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                             uint32_t width, uint32_t height, uint32_t depth) const;
    void copyBufferToImage2D(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                             uint32_t width, uint32_t height) const;

    const std::string& error() const { return error_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    bool ownsInstance_ = false;
    bool ownsDevice_ = false;
    bool ownsCommandPool_ = false;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    uint32_t graphicsFamily_ = 0, presentFamily_ = 0;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    DeviceInfo info_;
    bool fragmentStoresAndAtomics_ = false;
    std::string error_;
};

// Checks a VkResult and logs a message on failure.
#define SILK_VK_CHECK(expr)                                                                \
    do {                                                                                   \
        const VkResult _r = (expr);                                                        \
        if (_r != VK_SUCCESS) {                                                            \
            std::fprintf(stderr, "[vulkan] %s failed with VkResult %d at %s:%d\n", #expr,  \
                         static_cast<int>(_r), __FILE__, __LINE__);                        \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

}  // namespace silk
