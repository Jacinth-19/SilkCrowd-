// Synthetic Vulkan loader used by tools/gpu_probe_selftest.sh.
//
// It provides only the loader surface that `SilkCrowd --gpu-info` touches, so the device
// enumeration, the version handling and the classification can be exercised on a machine
// with no GPU and no driver at all. Behaviour is switched with environment variables:
//
//   STUB_DEVICES=0|1|2   how many physical devices to report (default 1)
//   STUB_REFUSE_13=1     refuse Vulkan 1.3 instances (driver supports 1.2 only)
//   STUB_FAIL_INSTANCE=1 fail vkCreateInstance outright (no driver answered)
//
// Device 1 is an llvmpipe-like CPU device with Vulkan 1.3, mesh shaders and a graphics
// queue (usable); device 2 is an old integrated device with Vulkan 1.1 (not usable).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

static uint32_t deviceCount(void) {
    const char* e = getenv("STUB_DEVICES");
    return e ? (uint32_t)atoi(e) : 1u;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t* version) {
    if (version) *version = VK_MAKE_API_VERSION(0, 1, 3, 289);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* ci,
                                               const VkAllocationCallbacks* alloc,
                                               VkInstance* instance) {
    (void)alloc;
    if (getenv("STUB_FAIL_INSTANCE")) return VK_ERROR_INITIALIZATION_FAILED;
    if (getenv("STUB_REFUSE_13") && ci->pApplicationInfo->apiVersion >= VK_API_VERSION_1_3)
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    *instance = (VkInstance)(uintptr_t)1;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance,
                                            const VkAllocationCallbacks* alloc) {
    (void)instance;
    (void)alloc;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance instance, uint32_t* count,
                                                          VkPhysicalDevice* devices) {
    (void)instance;
    const uint32_t n = deviceCount();
    if (!devices) {
        *count = n;
        return VK_SUCCESS;
    }
    const uint32_t write = *count < n ? *count : n;
    for (uint32_t i = 0; i < write; ++i) devices[i] = (VkPhysicalDevice)(uintptr_t)(i + 1);
    *count = write;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice device,
                                                         VkPhysicalDeviceProperties* props) {
    memset(props, 0, sizeof(*props));
    if ((uintptr_t)device == 1) {
        strcpy(props->deviceName, "llvmpipe (LLVM 17.0.6, 256 bits)");
        props->deviceType = VK_PHYSICAL_DEVICE_TYPE_CPU;
        props->apiVersion =
            getenv("STUB_REFUSE_13") ? VK_API_VERSION_1_2 : VK_API_VERSION_1_3;
    } else {
        strcpy(props->deviceName, "Stub Old Driver");
        props->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        props->apiVersion = VK_API_VERSION_1_1;
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice device, uint32_t* count, VkQueueFamilyProperties* queues) {
    (void)device;
    if (!queues) {
        *count = 1;
        return;
    }
    memset(queues, 0, sizeof(*queues));
    queues->queueFlags = VK_QUEUE_GRAPHICS_BIT;
    queues->queueCount = 1;
    *count = 1;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice device, const char* layer, uint32_t* count, VkExtensionProperties* props) {
    (void)device;
    (void)layer;
    if (!props) {
        *count = 1;
        return VK_SUCCESS;
    }
    memset(props, 0, sizeof(*props));
    strcpy(props->extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME);
    props->specVersion = 1;
    *count = 1;
    return VK_SUCCESS;
}
