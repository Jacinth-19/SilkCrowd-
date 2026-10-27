// Vulkan entry points that the Android loader does not export.
//
// Dynamic rendering was promoted to Vulkan 1.3 core, so the renderer calls
// vkCmdBeginRendering / vkCmdEndRendering. On desktop the loader exports those symbols and
// a direct call links fine - but Android's libvulkan.so stops at the Vulkan 1.1 API surface,
// so a direct reference fails at link time inside the APK build (ld.lld: undefined symbol:
// vkCmdBeginRendering) and, even if it linked, the shim would have nothing to bind to.
//
// They are therefore resolved through vkGetDeviceProcAddr at device-creation time: core name
// first, then the VK_KHR_dynamic_rendering aliases that Vulkan 1.1 Android drivers expose.
// Both spellings take the same VkRenderingInfo struct (the sType values are the same number),
// so one pointer type covers either path.
#pragma once

#include <vulkan/vulkan.h>

namespace silk {
namespace vk13 {

extern PFN_vkCmdBeginRendering cmdBeginRendering;  // nullptr if the driver offers neither
extern PFN_vkCmdEndRendering cmdEndRendering;

// Call once per VkDevice, after vkCreateDevice (VulkanContext::init / adopt do this).
// Returns false when dynamic rendering entry points are unavailable on this device.
bool resolve(VkDevice device);

// Convenience wrappers: the two call sites read exactly like the core API.
inline void beginRendering(VkCommandBuffer cmd, const VkRenderingInfo* info) {
    if (cmdBeginRendering) cmdBeginRendering(cmd, info);
}
inline void endRendering(VkCommandBuffer cmd) {
    if (cmdEndRendering) cmdEndRendering(cmd);
}

}  // namespace vk13
}  // namespace silk
