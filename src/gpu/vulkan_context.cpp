#include "vulkan_context.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace silk {
namespace {

const char* tierName(HardwareTier t) {
    switch (t) {
        case HardwareTier::Discrete: return "discrete GPU";
        case HardwareTier::Integrated: return "integrated GPU";
        case HardwareTier::Mobile: return "mobile GPU";
        case HardwareTier::Software: return "software rasteriser";
        default: return "unknown";
    }
}

std::string versionString(uint32_t v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u", VK_API_VERSION_MAJOR(v),
                  VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
    return buf;
}

}  // namespace

DeviceInfo detectDevice(const VkPhysicalDeviceProperties& props, uint64_t memoryMB,
                        bool meshShaders) {
    DeviceInfo info;
    info.name = props.deviceName;
    info.vendorId = props.vendorID;
    info.deviceId = props.deviceID;
    info.meshShaders = meshShaders;
    info.apiVersion = versionString(props.apiVersion);
    info.maxTexture3DSize = props.limits.maxImageDimension3D;
    info.memoryMB = memoryMB;
    info.dynamicRendering = true;  // checked by the caller against the enabled features

    const uint32_t v = props.vendorID;
    const std::string n(props.deviceName);
    const bool isIntel = (v == 0x8086);
    const bool isArm = (v == 0x13B5);
    const bool isQualcomm = (v == 0x5143);
    const bool isMoltenVK = n.find("Apple") != std::string::npos ||
                            n.find("MoltenVK") != std::string::npos;
    const bool isSoftware = n.find("llvmpipe") != std::string::npos ||
                            n.find("SwiftShader") != std::string::npos ||
                            n.find("lavapipe") != std::string::npos ||
                            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;

    if (isSoftware) info.tier = HardwareTier::Software;
    else if (isIntel || isArm || isQualcomm || isMoltenVK) info.tier = HardwareTier::Integrated;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) info.tier = HardwareTier::Discrete;
    else info.tier = HardwareTier::Integrated;

    // Android / mobile: devices with little memory or an ARM/Qualcomm GPU.
    if ((isArm || isQualcomm) && memoryMB < 8192) info.tier = HardwareTier::Mobile;

    // Automatic quality policy (Section B): keep the frame time inside the target.
    switch (info.tier) {
        case HardwareTier::Discrete:
            info.targetFps = 120;
            info.defaultMsaa = 8;
            info.defaultCrowd = 100;
            info.lodBias = 1.0f;
            break;
        case HardwareTier::Integrated:
            info.targetFps = 60;
            info.defaultMsaa = 4;
            info.defaultCrowd = 40;
            info.lodBias = 1.6f;   // bias the LOD towards fewer strands/vertices
            break;
        case HardwareTier::Mobile:
            info.targetFps = 60;
            info.defaultMsaa = 1;
            info.defaultCrowd = 12;
            info.lodBias = 2.6f;
            break;
        case HardwareTier::Software:
            info.targetFps = 10;
            info.defaultMsaa = 1;
            info.defaultCrowd = 1;
            info.lodBias = 4.0f;
            break;
        default:
            break;
    }
    return info;
}

bool VulkanContext::init(VkInstance instance, VkSurfaceKHR surface, bool enableValidation,
                         std::string* error) {
    (void)enableValidation;
    instance_ = instance;
    ownsInstance_ = false;

    if (instance_ == VK_NULL_HANDLE) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "SilkCrowd";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName = "SilkCrowd hair mesh renderer";
        app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion = VK_API_VERSION_1_3;

        std::vector<const char*> layers;
        std::vector<const char*> extensions;
        if (enableValidation) layers.push_back("VK_LAYER_KHRONOS_validation");

        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
        ci.ppEnabledLayerNames = layers.data();
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
        const VkResult r = vkCreateInstance(&ci, nullptr, &instance_);
        if (r != VK_SUCCESS) {
            error_ = "vkCreateInstance failed";
            if (error) *error = error_;
            return false;
        }
        ownsInstance_ = true;
    }

    // ---- physical device selection ----
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        error_ = "no Vulkan physical device";
        if (error) *error = error_;
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int bestScore = -1;
    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        // Queue families.
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, queues.data());
        bool hasGraphics = false, hasPresent = surface == VK_NULL_HANDLE;
        uint32_t gfx = 0, present = 0;
        for (uint32_t i = 0; i < qCount; ++i) {
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                if (!hasGraphics) { hasGraphics = true; gfx = i; }
                if (surface != VK_NULL_HANDLE) {
                    VkBool32 supported = VK_FALSE;
                    vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &supported);
                    if (supported) { hasPresent = true; present = i; }
                }
            }
        }
        if (!hasGraphics || !hasPresent) continue;

        // Device extensions.
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());
        bool mesh = false;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0) mesh = true;
        }

        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(dev, &mem);
        uint64_t totalMB = 0;
        for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
            if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                totalMB += mem.memoryHeaps[i].size / (1024 * 1024);

        int score = static_cast<int>(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 500);
        score += mesh ? 200 : 0;
        score += static_cast<int>(std::min<uint64_t>(totalMB / 512, 200));
        if (score > bestScore) {
            bestScore = score;
            best = dev;
            graphicsFamily_ = gfx;
            presentFamily_ = present;
            physical_ = dev;
            VkPhysicalDeviceProperties bestProps = props;
            info_ = detectDevice(bestProps, totalMB, mesh);
        }
    }
    if (best == VK_NULL_HANDLE) {
        error_ = "no suitable Vulkan device (graphics + present queue required)";
        if (error) *error = error_;
        return false;
    }

    // ---- device creation ----
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = graphicsFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    queueInfos.push_back(qci);
    if (presentFamily_ != graphicsFamily_) {
        VkDeviceQueueCreateInfo qci2 = qci;
        qci2.queueFamilyIndex = presentFamily_;
        queueInfos.push_back(qci2);
    }

    std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                 VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME};
    if (info_.meshShaders) deviceExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy = VK_TRUE;
    features.fragmentStoresAndAtomics = VK_TRUE;
    features.fillModeNonSolid = VK_TRUE;

    VkPhysicalDeviceDynamicRenderingFeatures dynamicRendering{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    dynamicRendering.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    meshFeatures.meshShader = info_.meshShaders ? VK_TRUE : VK_FALSE;
    meshFeatures.taskShader = info_.meshShaders ? VK_TRUE : VK_FALSE;
    meshFeatures.pNext = &dynamicRendering;

    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.features = features;
    features2.pNext = info_.meshShaders ? static_cast<void*>(&meshFeatures)
                                        : static_cast<void*>(&dynamicRendering);

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &features2;
    dci.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    dci.pQueueCreateInfos = queueInfos.data();
    dci.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    dci.ppEnabledExtensionNames = deviceExtensions.data();

    const VkResult dr = vkCreateDevice(best, &dci, nullptr, &device_);
    if (dr != VK_SUCCESS) {
        error_ = "vkCreateDevice failed (missing VK_KHR_dynamic_rendering or mesh shaders?)";
        if (error) *error = error_;
        return false;
    }
    vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
    fragmentStoresAndAtomics_ = features.fragmentStoresAndAtomics == VK_TRUE;

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = graphicsFamily_;
    SILK_VK_CHECK(vkCreateCommandPool(device_, &pci, nullptr, &commandPool_));
    ownsCommandPool_ = true;
    ownsDevice_ = true;

    std::printf("[vulkan] %s (%s, API %s, %llu MB)%s\n", info_.name.c_str(),
                tierName(info_.tier), info_.apiVersion.c_str(),
                static_cast<unsigned long long>(info_.memoryMB),
                info_.meshShaders ? ", mesh shaders: yes (compute fallback available)"
                                  : ", mesh shaders: no -> compute fallback");
    return true;
}

bool VulkanContext::adopt(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                          VkQueue graphicsQueue, VkQueue presentQueue, uint32_t graphicsFamily,
                          uint32_t presentFamily, VkCommandPool commandPool, bool meshShaders,
                          std::string* error) {
    instance_ = instance;
    physical_ = physical;
    device_ = device;
    graphicsQueue_ = graphicsQueue;
    presentQueue_ = presentQueue;
    graphicsFamily_ = graphicsFamily;
    presentFamily_ = presentFamily;
    commandPool_ = commandPool;
    ownsInstance_ = false;
    ownsDevice_ = false;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(physical, &mem);
    uint64_t totalMB = 0;
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
        if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            totalMB += mem.memoryHeaps[i].size / (1024 * 1024);
    info_ = detectDevice(props, totalMB, meshShaders);
    if (device_ == VK_NULL_HANDLE) {
        error_ = "adopt() called without a device";
        if (error) *error = error_;
        return false;
    }
    return true;
}

void VulkanContext::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        if (commandPool_ != VK_NULL_HANDLE && ownsCommandPool_)
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        if (ownsDevice_) vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (ownsInstance_ && instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(physical_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return 0;
}

VkBuffer VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags props, VkDeviceMemory* memory) const {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &bci, nullptr, &buffer) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device_, &mai, nullptr, memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return VK_NULL_HANDLE;
    }
    vkBindBufferMemory(device_, buffer, *memory, 0);
    return buffer;
}

void VulkanContext::destroyBuffer(VkBuffer buffer, VkDeviceMemory memory) const {
    if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device_, memory, nullptr);
}

VkImage VulkanContext::createImage2D(uint32_t width, uint32_t height, VkFormat format,
                                     VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                     VkDeviceMemory* memory, VkImageView* view,
                                     VkSampleCountFlagBits samples) const {
    VkImage image = VK_NULL_HANDLE;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.format = format;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage = usage;
    ici.samples = samples;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device_, &ici, nullptr, &image) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device_, &mai, nullptr, memory) != VK_SUCCESS) return VK_NULL_HANDLE;
    vkBindImageMemory(device_, image, *memory, 0);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCreateImageView(device_, &vci, nullptr, view);
    return image;
}

VkImage VulkanContext::createImage3D(uint32_t width, uint32_t height, uint32_t depth,
                                     VkFormat format, VkImageUsageFlags usage,
                                     VkDeviceMemory* memory, VkImageView* view) const {
    VkImage image = VK_NULL_HANDLE;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_3D;
    ici.extent = {width, height, depth};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.format = format;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage = usage;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device_, &ici, nullptr, &image) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device_, &mai, nullptr, memory) != VK_SUCCESS) return VK_NULL_HANDLE;
    vkBindImageMemory(device_, image, *memory, 0);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device_, &vci, nullptr, view);
    return image;
}

void VulkanContext::destroyImage(VkImage image, VkDeviceMemory memory, VkImageView view) const {
    if (view != VK_NULL_HANDLE) vkDestroyImageView(device_, view, nullptr);
    if (image != VK_NULL_HANDLE) vkDestroyImage(device_, image, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(device_, memory, nullptr);
}

VkCommandBuffer VulkanContext::beginSingleTimeCommands() const {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue_);
    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
}

void VulkanContext::imageLayoutTransition(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
                                          VkImageLayout to, VkImageAspectFlags aspect) const {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void VulkanContext::copyBufferToImage3D(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                                        uint32_t width, uint32_t height, uint32_t depth) const {
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, depth};
    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void VulkanContext::copyBufferToImage2D(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                                        uint32_t width, uint32_t height) const {
    copyBufferToImage3D(cmd, buffer, image, width, height, 1);
}

}  // namespace silk
