// Fills GpuFacts for the Vulkan diagnostics: loader probe, ICD manifest scan, and - when
// this binary was built with the GPU renderer - a real instance/device enumeration.
//
// Deliberately dependency-free apart from the platform's dynamic loader, so the same code
// answers the question in a build *without* Vulkan ("would a rebuild even help?").
#include "gpu_probe.h"

#include "silkcrowd/gpu_diag.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__linux__) || defined(__APPLE__)
#include <dirent.h>
#include <sys/stat.h>
#endif

#ifdef SILKCROWD_WITH_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace silk {
namespace {

// ---- tiny platform shims -------------------------------------------------------------

void* openLibrary(const char* name) {
#if defined(_WIN32)
    return static_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void closeLibrary(void* handle) {
    if (!handle) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

void* librarySymbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

bool fileExists(const std::string& path) {
    if (path.empty()) return false;
#if defined(_WIN32)
    const DWORD attr = GetFileAttributesA(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

bool directoryExists(const std::string& path) {
    if (path.empty()) return false;
#if defined(_WIN32)
    const DWORD attr = GetFileAttributesA(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

std::vector<std::string> listJsonFiles(const std::string& dir) {
    std::vector<std::string> out;
#if defined(_WIN32)
    WIN32_FIND_DATAA data;
    const std::string pattern = dir + "\\*.json";
    HANDLE h = FindFirstFileA(pattern.c_str(), &data);
    if (h == INVALID_HANDLE_VALUE) return out;
    do {
        if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) out.push_back(dir + "/" + data.cFileName);
    } while (FindNextFileA(h, &data));
    FindClose(h);
#else
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    while (const dirent* e = readdir(d)) {
        const std::string name(e->d_name);
        if (name.size() > 5 && name.compare(name.size() - 5, 5, ".json") == 0)
            out.push_back(dir + "/" + name);
    }
    closedir(d);
#endif
    struct less {
        bool operator()(const std::string& a, const std::string& b) const { return a < b; }
    };
    std::sort(out.begin(), out.end(), less());
    return out;
}

std::string envValue(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::string readFile(const std::string& path) {
    std::string out;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        char buf[4096];
        std::size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
        std::fclose(f);
    }
    return out;
}

// The library names the loader is known as on this platform, most specific first.
void addLoaderCandidates(GpuFacts& f) {
#if defined(_WIN32)
    f.loaderCandidates = {"vulkan-1.dll"};
#elif defined(__APPLE__)
    f.loaderCandidates = {"libvulkan.1.dylib", "libvulkan.dylib",
                          "/opt/homebrew/lib/libvulkan.1.dylib",
                          "/usr/local/lib/libvulkan.1.dylib"};
#else
    f.loaderCandidates = {"libvulkan.so.1", "libvulkan.so"};
#endif
}

// Every directory the loader is allowed to read ICD manifests from, plus the SDK paths.
void addIcdSearchDirs(GpuFacts& f) {
#if defined(_WIN32)
    // Windows driver discovery is registry based (HKLM\SOFTWARE\Khronos\Vulkan\Drivers);
    // the environment override and the SDK layout are what we can check without COM.
    const std::string sdk = envValue("VULKAN_SDK");
    if (!sdk.empty()) f.icdSearchDirs.push_back(sdk + "\\Bin");
#elif defined(__APPLE__)
    f.icdSearchDirs.push_back("/opt/homebrew/share/vulkan/icd.d");
    f.icdSearchDirs.push_back("/usr/local/share/vulkan/icd.d");
    const std::string home = envValue("HOME");
    if (!home.empty()) f.icdSearchDirs.push_back(home + "/.local/share/vulkan/icd.d");
#else
    f.icdSearchDirs.push_back("/usr/share/vulkan/icd.d");
    f.icdSearchDirs.push_back("/etc/vulkan/icd.d");
    f.icdSearchDirs.push_back("/usr/local/share/vulkan/icd.d");
    const std::string sdk = envValue("VULKAN_SDK");
    if (!sdk.empty()) f.icdSearchDirs.push_back(sdk + "/share/vulkan/icd.d");
    // Env-derived directories only when they are actually set: "/vulkan/icd.d" is not a path
    // the loader ever looks at, and printing it just adds noise to the report.
    const std::string xdgData = envValue("XDG_DATA_HOME");
    if (!xdgData.empty()) f.icdSearchDirs.push_back(xdgData + "/vulkan/icd.d");
    const std::string home = envValue("HOME");
    if (!home.empty()) f.icdSearchDirs.push_back(home + "/.local/share/vulkan/icd.d");
#endif
}

// Reads a manifest and records whether its library_path exists. Returns nothing; the
// manifest is only a hint, and a stale manifest with a missing .so is a real failure mode
// (driver uninstalled, container image trimmed) worth naming explicitly.
void inspectManifest(GpuFacts& f, const std::string& manifest) {
    const std::string json = readFile(manifest);
    if (json.empty()) return;
    f.icdManifests.push_back(manifest);
    const std::string lib = resolveIcdLibraryPath(manifest, jsonStringField(json, "library_path"));
    if (lib.empty()) {
        f.icdMissing.push_back(manifest + " (no library_path)");
        return;
    }
    if (fileExists(lib))
        f.icdLibraries.push_back(lib);
    else
        f.icdMissing.push_back(lib);
}

void probeLoader(GpuFacts& f) {
    void* handle = nullptr;
    for (const std::string& name : f.loaderCandidates) {
        handle = openLibrary(name.c_str());
        if (handle) {
            f.loaderFound = true;
            f.loaderPath = name;
            break;
        }
    }
    if (!handle) return;

    // vkEnumerateInstanceVersion needs no instance, so dlsym is enough even when the binary
    // was built without Vulkan headers - that is what makes "would a rebuild help?" answerable.
    using EnumerateVersionFn = int (*)(unsigned*);
    if (auto fn = reinterpret_cast<EnumerateVersionFn>(
            librarySymbol(handle, "vkEnumerateInstanceVersion"))) {
        unsigned version = 0;
        if (fn(&version) == 0) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%u.%u.%u", (version >> 22) & 0x7Fu,
                          (version >> 12) & 0x3FFu, version & 0xFFFu);
            f.loaderVersion = buf;
        }
    }
    closeLibrary(handle);
}

void probeIcds(GpuFacts& f) {
    // VK_DRIVER_FILES (Vulkan 1.3.207+) replaced VK_ICD_FILENAMES; both override the search.
    for (const char* var : {"VK_DRIVER_FILES", "VK_ICD_FILENAMES"}) {
        const std::string value = envValue(var);
        if (value.empty()) continue;
        f.icdOverride = std::string(var) + "=" + value;
        for (const std::string& entry : splitPathList(value))
            if (fileExists(entry)) inspectManifest(f, entry);
        return;  // an override replaces the default search, exactly like the loader's own rule
    }
    for (const std::string& dir : f.icdSearchDirs) {
        if (!directoryExists(dir)) continue;
        for (const std::string& manifest : listJsonFiles(dir)) inspectManifest(f, manifest);
    }
}

bool shaderBlobsPresent() {
    std::vector<std::string> candidates;
#ifdef SILKCROWD_SHADER_DIR
    candidates.push_back(std::string(SILKCROWD_SHADER_DIR) + "/hair.frag.spv");
#endif
#ifdef SILKCROWD_SOURCE_DIR
    candidates.push_back(std::string(SILKCROWD_SOURCE_DIR) + "/build/shaders/hair.frag.spv");
    candidates.push_back(std::string(SILKCROWD_SOURCE_DIR) + "/shaders/hair.frag.spv");
#endif
    candidates.push_back("shaders/hair.frag.spv");
    for (const std::string& path : candidates)
        if (fileExists(path)) return true;
    return false;
}

#ifdef SILKCROWD_WITH_VULKAN

std::string apiVersionString(uint32_t v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v),
                  VK_API_VERSION_PATCH(v));
    return buf;
}

const char* deviceTypeName(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

void probeVulkanDevices(GpuFacts& f) {
    // Zero-init + assignment rather than {sType}: the app target is built with -Werror in
    // CI and does not carry -Wno-missing-field-initializers.
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "SilkCrowd --gpu-info";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "SilkCrowd";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, nullptr, &instance);
    if (r == VK_ERROR_INCOMPATIBLE_DRIVER) {
        // The loader is fine, the driver just cannot do 1.3. Ask for 1.0 so the device list
        // can name it and say which version it does support.
        app.apiVersion = VK_API_VERSION_1_0;
        r = vkCreateInstance(&ci, nullptr, &instance);
        if (r == VK_SUCCESS) f.note = "the installed driver refuses Vulkan 1.3 instances";
    }
    if (r != VK_SUCCESS) {
        f.note = std::string("vkCreateInstance failed with VkResult ") + std::to_string(static_cast<int>(r)) +
                 " (no driver answered; VK_ERROR_INCOMPATIBLE_DRIVER=4294967292)";
        return;
    }

    f.instanceCreated = true;
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) count = 0;
    f.deviceEnumerated = true;
    std::vector<VkPhysicalDevice> devices(count);
    if (count) vkEnumeratePhysicalDevices(instance, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, queues.data());
        bool graphics = false;
        for (const VkQueueFamilyProperties& q : queues)
            if (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics = true;

        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());
        bool mesh = false;
        for (const VkExtensionProperties& e : exts)
            if (std::strcmp(e.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0) mesh = true;

        const bool v13 = props.apiVersion >= VK_API_VERSION_1_3;
        const bool usable = v13 && graphics;
        char line[320];
        std::snprintf(line, sizeof(line), "%s [%s] api %s - %s, %s, %s", props.deviceName,
                      deviceTypeName(props.deviceType), apiVersionString(props.apiVersion).c_str(),
                      mesh ? "mesh shaders" : "no mesh shaders",
                      graphics ? "graphics queue" : "NO graphics queue",
                      usable ? "usable" : "not usable (needs Vulkan 1.3 + graphics)");
        f.deviceLines.push_back(line);

        int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 500;
        if (mesh) score += 200;
        if (!usable) score -= 1000;
        if (score > bestScore) {
            bestScore = score;
            f.deviceName = props.deviceName;
            f.apiVersion = apiVersionString(props.apiVersion);
            f.meshShaders = mesh;
            f.deviceUsable = usable;
        }
    }
    vkDestroyInstance(instance, nullptr);
}

#else  // !SILKCROWD_WITH_VULKAN

void probeVulkanDevices(GpuFacts&) {
    // No Vulkan headers at configure time, so there is nothing to enumerate with. The
    // loader/ICD probe above still answers whether a rebuild would find a device.
}

#endif  // SILKCROWD_WITH_VULKAN

}  // namespace

GpuFacts probeGpu() {
    GpuFacts f;
#ifdef SILKCROWD_WITH_VULKAN
    f.builtWithVulkan = true;
#endif
    f.shaderBlobsPresent = shaderBlobsPresent();

    addLoaderCandidates(f);
    addIcdSearchDirs(f);
    probeLoader(f);
    probeIcds(f);
    if (f.builtWithVulkan) {
        probeVulkanDevices(f);
    } else if (f.loaderFound && !f.icdLibraries.empty()) {
        f.note = "the loader and an ICD are installed - a Vulkan build would have a device to "
                 "render on";
    }
    return f;
}

void printGpuReport(const GpuFacts& facts, FILE* out) {
    for (const std::string& line : gpuReportLines(facts)) std::fprintf(out, "%s\n", line.c_str());
}

bool gpuUsable(const GpuFacts& facts) { return classify(facts) == GpuIssue::Ready; }

}  // namespace silk
