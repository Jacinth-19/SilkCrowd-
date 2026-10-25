// SilkCrowd - Android entry point.
//
// NativeActivity -> ANativeWindow -> VkSurfaceKHR -> swapchain, and then exactly the same
// HairGpuRenderer the desktop and CI builds use. The renderer owns everything after the
// swapchain image is handed to renderToSwapchainImage(): task/mesh hair pass (or the
// compute fallback of Section 3.2), PCF shadows, bloom and the composite into the
// acquired image, which is left in PRESENT_SRC_KHR.
//
// Headless CI mode:  adb shell am start -n com.silkcrowd.renderer/.MainActivity \
//                        --ez headless true --ei seconds 15
// renders the crowd scene without drawing to the window and logs the statistics line.
#include <android/log.h>
#include <android_native_app_glue.h>
#include <jni.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gpu/hair_gpu_renderer.h"
#include "gpu/vulkan_context.h"
#include "scenes.h"
#include "silkcrowd/assets.h"
#include "silkcrowd/renderer.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SilkCrowd", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "SilkCrowd", __VA_ARGS__)

using namespace silk;

namespace {

// ---------------------------------------------------------------- assets
// Copies assets/shaders/*.spv into the app's private directory and chdir()s there, so
// loadShader()'s relative "shaders/<name>.spv" lookup works on Android (assets are not
// accessible through fopen()).
bool stageAssets(android_app* app) {
    AAssetManager* mgr = app->activity->assetManager;
    AAssetDir* dir = AAssetManager_openDir(mgr, "shaders");
    if (!dir) return false;
    const char* outDir = app->activity->internalDataPath;
    if (!outDir) return false;
    if (chdir(outDir) != 0) {
        LOGE("chdir(%s) failed", outDir);
        return false;
    }
    mkdir("shaders", 0755);
    int copied = 0;
    while (const char* name = AAssetDir_getNextFileName(dir)) {
        AAsset* asset = AAssetManager_open(mgr, (std::string("shaders/") + name).c_str(),
                                          AASSET_MODE_BUFFER);
        if (!asset) continue;
        const void* data = AAsset_getBuffer(asset);
        const off_t size = AAsset_getLength(asset);
        const std::string path = std::string("shaders/") + name;
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            if (data) std::fwrite(data, 1, static_cast<size_t>(size), f);
            std::fclose(f);
            ++copied;
        }
        AAsset_close(asset);
    }
    AAssetDir_close(dir);
    LOGI("staged %d shader modules", copied);
    return copied > 0;
}

// ---------------------------------------------------------------- intent extras
bool intentBool(android_app* app, const char* key, bool fallback) {
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    jobject activity = app->activity->clazz;
    jclass cls = env->GetObjectClass(activity);
    jmethodID getIntent = env->GetMethodID(cls, "getIntent", "()Landroid/content/Intent;");
    jobject intent = getIntent ? env->CallObjectMethod(activity, getIntent) : nullptr;
    bool value = fallback;
    if (intent) {
        jclass icls = env->GetObjectClass(intent);
        jmethodID getBool = env->GetMethodID(icls, "getBooleanExtra", "(Ljava/lang/String;Z)Z");
        if (getBool) {
            jstring k = env->NewStringUTF(key);
            value = env->CallBooleanMethod(intent, getBool, k, fallback) == JNI_TRUE;
            env->DeleteLocalRef(k);
        }
    }
    app->activity->vm->DetachCurrentThread();
    return value;
}

int intentInt(android_app* app, const char* key, int fallback) {
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    jobject activity = app->activity->clazz;
    jclass cls = env->GetObjectClass(activity);
    jmethodID getIntent = env->GetMethodID(cls, "getIntent", "()Landroid/content/Intent;");
    jobject intent = getIntent ? env->CallObjectMethod(activity, getIntent) : nullptr;
    int value = fallback;
    if (intent) {
        jclass icls = env->GetObjectClass(intent);
        jmethodID getInt = env->GetMethodID(icls, "getIntExtra", "(Ljava/lang/String;I)I");
        if (getInt) {
            jstring k = env->NewStringUTF(key);
            value = env->CallIntMethod(intent, getInt, k, fallback);
            env->DeleteLocalRef(k);
        }
    }
    app->activity->vm->DetachCurrentThread();
    return value;
}

// ---------------------------------------------------------------- app state
struct AppState {
    VulkanContext context;
    std::unique_ptr<HairGpuRenderer> renderer;
    std::shared_ptr<Scene> scene;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{0, 0};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    VkSemaphore acquireSem = VK_NULL_HANDLE;
    VkSemaphore presentSem = VK_NULL_HANDLE;
    bool headless = false;
    float seconds = 0.0f;
    float time = 0.0f;
    float dragX = 0.0f, dragY = 0.0f;
    bool dragging = false;
};

AppState* g_app = nullptr;

bool createSwapchain(AppState* s, uint32_t w, uint32_t h) {
    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(s->context.physicalDevice(), s->surface,
                                                  &caps) != VK_SUCCESS)
        return false;
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(s->context.physicalDevice(), s->surface, &formatCount,
                                         nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(s->context.physicalDevice(), s->surface, &formatCount,
                                         formats.data());
    VkSurfaceFormatKHR chosen = formats.empty()
                                    ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM,
                                                         VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
                                    : formats[0];
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) chosen = f;
    s->swapchainFormat = chosen.format;

    VkExtent2D extent{caps.currentExtent.width, caps.currentExtent.height};
    if (extent.width == 0xFFFFFFFFu) {
        extent.width = std::min(std::max(w, caps.minImageExtent.width), caps.maxImageExtent.width);
        extent.height = std::min(std::max(h, caps.minImageExtent.height), caps.maxImageExtent.height);
    }
    s->extent = extent;

    const uint32_t imageCount = std::min(caps.minImageCount + 1,
                                         caps.maxImageCount ? caps.maxImageCount : 3);
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = s->surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    if (vkCreateSwapchainKHR(s->context.device(), &ci, nullptr, &s->swapchain) != VK_SUCCESS)
        return false;

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(s->context.device(), s->swapchain, &count, nullptr);
    s->images.resize(count);
    vkGetSwapchainImagesKHR(s->context.device(), s->swapchain, &count, s->images.data());
    s->views.resize(count, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = s->images[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = s->swapchainFormat;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(s->context.device(), &vi, nullptr, &s->views[i]);
    }
    LOGI("swapchain %ux%u, %u images, format %d", extent.width, extent.height, count,
         static_cast<int>(s->swapchainFormat));
    return true;
}

void destroySwapchain(AppState* s) {
    if (!s->context.device()) return;
    for (VkImageView v : s->views)
        if (v) vkDestroyImageView(s->context.device(), v, nullptr);
    s->views.clear();
    s->images.clear();
    if (s->swapchain) vkDestroySwapchainKHR(s->context.device(), s->swapchain, nullptr);
    s->swapchain = VK_NULL_HANDLE;
}

bool initVulkan(android_app* app, AppState* s) {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "SilkCrowd";
    ai.apiVersion = VK_API_VERSION_1_3;
    const char* extensions[] = {"VK_KHR_surface", "VK_KHR_android_surface"};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = extensions;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        LOGE("vkCreateInstance failed");
        return false;
    }
    VkAndroidSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    sci.window = app->window;
    if (vkCreateAndroidSurfaceKHR(instance, &sci, nullptr, &s->surface) != VK_SUCCESS) {
        LOGE("vkCreateAndroidSurfaceKHR failed");
        return false;
    }
    std::string error;
    if (!s->context.init(instance, s->surface, false, &error)) {
        LOGE("VulkanContext::init failed: %s", error.c_str());
        return false;
    }
    VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(s->context.device(), &sem, nullptr, &s->acquireSem);
    vkCreateSemaphore(s->context.device(), &sem, nullptr, &s->presentSem);
    return true;
}

bool initRenderer(android_app* app, AppState* s) {
    s->renderer = std::make_unique<HairGpuRenderer>();
    const uint32_t w = static_cast<uint32_t>(ANativeWindow_getWidth(app->window));
    const uint32_t h = static_cast<uint32_t>(ANativeWindow_getHeight(app->window));
    std::string error;
    if (!s->renderer->init(&s->context, w, h, false, &error)) {
        LOGE("renderer init failed: %s", error.c_str());
        return false;
    }
    // The Android tier is chosen by detectDevice(): the crowd size and MSAA are reduced on
    // mobile GPUs to keep the 120 fps target.
    RenderOptions probe;
    s->renderer->applyHardwarePolicy(s->context.info(), &probe);
    const int crowd = s->context.info().tier == HardwareTier::Mobile ? 60 : 100;
    s->scene = std::make_shared<Scene>(buildCrowdScene(crowd));
    s->renderer->setScene(s->scene);
    LOGI("device: %s, tier crowd=%d msaa=%d", s->context.info().name.c_str(), crowd,
         probe.msaa);
    return true;
}

void renderFrame(AppState* s, float dt, bool headless) {
    if (!s->renderer || !s->scene) return;
    Camera cam = crowdCamera(1280, 720);
    cam.position = Vec3(1.15f * std::cos(s->time * 0.15f), 1.1f,
                        1.15f * std::sin(s->time * 0.15f) + 1.9f);
    cam.target = Vec3(0.0f, 0.45f, 0.0f);
    RenderOptions opt;
    opt.width = static_cast<int>(s->extent.width ? s->extent.width : 1280);
    opt.height = static_cast<int>(s->extent.height ? s->extent.height : 720);
    opt.enableLod = true;
    opt.drawShadows = true;
    s->renderer->applyHardwarePolicy(s->context.info(), &opt);
    (void)dt;

    if (headless) {
        s->renderer->renderFrame(cam, opt, std::string());
        return;
    }
    uint32_t index = 0;
    if (vkAcquireNextImageKHR(s->context.device(), s->swapchain, UINT64_MAX, s->acquireSem,
                              VK_NULL_HANDLE, &index) != VK_SUCCESS)
        return;
    if (!s->renderer->renderToSwapchainImage(s->images[index], s->extent.width,
                                             s->extent.height, cam, opt))
        return;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &s->presentSem;
    pi.swapchainCount = 1;
    pi.pSwapchains = &s->swapchain;
    pi.pImageIndices = &index;
    vkQueuePresentKHR(s->context.presentQueue(), &pi);
}

}  // namespace

void android_main(android_app* app) {
    AppState state;
    g_app = &state;
    app->onAppCmd = [](android_app* a, int32_t cmd) {
        switch (cmd) {
            case APP_CMD_INIT_WINDOW:
                if (a->window && g_app->context.device() == VK_NULL_HANDLE) {
                    if (!stageAssets(a)) LOGE("shader staging failed");
                    if (!initVulkan(a, g_app)) return;
                    if (!initRenderer(a, g_app)) return;
                    createSwapchain(g_app, ANativeWindow_getWidth(a->window),
                                    ANativeWindow_getHeight(a->window));
                    g_app->renderer->setCompositeTarget(VK_NULL_HANDLE, g_app->swapchainFormat);
                }
                break;
            case APP_CMD_TERM_WINDOW:
                break;
            case APP_CMD_DESTROY:
                if (g_app->renderer) g_app->renderer->shutdown();
                destroySwapchain(g_app);
                g_app->context.shutdown();
                break;
            default:
                break;
        }
    };
    app->onInputEvent = [](android_app* a, AInputEvent* event) -> int32_t {
        if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
            const int32_t action = AMotionEvent_getAction(event);
            const float x = AMotionEvent_getX(event, 0);
            const float y = AMotionEvent_getY(event, 0);
            if ((action & AMOTION_EVENT_ACTION_MASK) == AMOTION_EVENT_ACTION_DOWN) {
                g_app->dragging = true;
                g_app->dragX = x;
                g_app->dragY = y;
            } else if ((action & AMOTION_EVENT_ACTION_MASK) == AMOTION_EVENT_ACTION_MOVE &&
                       g_app->dragging) {
                // Touch drag orbits the camera; the hair simulation reacts to the drag
                // point in the same way the desktop editor does.
                g_app->dragX = x;
                g_app->dragY = y;
            } else {
                g_app->dragging = false;
            }
            (void)a;
            return 1;
        }
        return 0;
    };

    state.headless = intentBool(app, "headless", false);
    state.seconds = static_cast<float>(intentInt(app, "seconds", 0));
    LOGI("SilkCrowd starting (headless=%d, seconds=%.0f)", state.headless ? 1 : 0, state.seconds);

    auto last = std::chrono::high_resolution_clock::now();
    int frame = 0;
    while (!app->destroyRequested) {
        int events = 0;
        android_poll_source* source = nullptr;
        const int timeout = state.headless ? 0 : 16;
        while (ALooper_pollAll(timeout, nullptr, &events, reinterpret_cast<void**>(&source)) >= 0) {
            if (source) source->process(app, source);
            if (app->destroyRequested) break;
        }
        if (!state.renderer) continue;
        const auto now = std::chrono::high_resolution_clock::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        state.time += dt;
        renderFrame(&state, dt, state.headless);
        if (++frame % 60 == 0) {
            const GpuStats& st = state.renderer->stats();
            LOGI("frame %d: %.2f ms | strands %llu/%llu | tris %llu | crowd %d | %s", frame,
                 st.frameMs, static_cast<unsigned long long>(st.strandsRendered),
                 static_cast<unsigned long long>(st.strandsTotal),
                 static_cast<unsigned long long>(st.triangles), st.crowdCount,
                 st.meshShaderPath ? "task/mesh" : "compute");
        }
        if (state.headless && state.seconds > 0.0f && state.time >= state.seconds) break;
    }
    if (state.renderer) state.renderer->shutdown();
    destroySwapchain(&state);
    state.context.shutdown();
    LOGI("SilkCrowd exiting");
}
