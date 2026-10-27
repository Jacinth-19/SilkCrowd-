// Classification and reporting for the Vulkan availability diagnostics.
//
// Pure functions only: no dlopen, no Vulkan API. gpu_probe.cpp fills a GpuFacts and these
// turn it into the text the CLI prints (and into the exit-relevant verdict).
#include "silkcrowd/gpu_diag.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace silk {
namespace {

// Right-pads a label to the width the report uses for its left column.
std::string label(const char* text) {
    const std::size_t width = 17;
    std::string out(text);
    while (out.size() < width) out.push_back(' ');
    return out;
}

std::string join(const std::vector<std::string>& items, const char* sep) {
    std::string out;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out += sep;
        out += items[i];
    }
    return out;
}

// The platform-correct install hint. Compile-time: the binary only ever runs on the OS it
// was built for, and guessing at run time would just add a source of wrong advice.
std::vector<std::string> installHints(GpuIssue issue) {
    std::vector<std::string> hints;
#if defined(__linux__)
    // The README targets Debian/Ubuntu; other distros map these package names directly.
    if (issue == GpuIssue::BuiltWithoutVulkan || issue == GpuIssue::NoLoader) {
        hints.push_back("sudo apt-get install -y libvulkan-dev mesa-vulkan-drivers "
                        "glslang-tools libgl1-mesa-dri");
        hints.push_back("then reconfigure: cmake -B build -DCMAKE_BUILD_TYPE=Release && "
                        "cmake --build build -j");
    }
    if (issue == GpuIssue::NoDriver) {
        hints.push_back("sudo apt-get install -y mesa-vulkan-drivers   # lavapipe (CPU) or your "
                        "GPU vendor's ICD");
        hints.push_back("verify with: vulkaninfo --summary");
    }
    if (issue == GpuIssue::NoDevice) {
        hints.push_back("let the container see the GPU: docker run --device /dev/dri ...");
        hints.push_back("or force the CPU driver: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/"
                        "lvp_icd.x86_64.json");
        hints.push_back("check what the loader sees with: vulkaninfo --summary");
    }
    if (issue == GpuIssue::UnsupportedDevice) {
        hints.push_back("SilkCrowd needs Vulkan 1.3 with a graphics queue; update the driver or "
                        "use the software rasteriser");
    }
#elif defined(_WIN32)
    if (issue == GpuIssue::BuiltWithoutVulkan || issue == GpuIssue::NoLoader) {
        hints.push_back("install the LunarG Vulkan SDK (sets VULKAN_SDK and vulkan-1.dll)");
        hints.push_back("then reconfigure: cmake -B build && cmake --build build --config Release");
    }
    if (issue == GpuIssue::NoDriver) {
        hints.push_back("install your GPU vendor's driver (it ships the Vulkan ICD)");
        hints.push_back("verify with: vulkaninfo --summary");
    }
    if (issue == GpuIssue::NoDevice) {
        hints.push_back("a remote-desktop session may hide the GPU; run locally or use the "
                        "software rasteriser");
    }
#elif defined(__APPLE__)
    if (issue == GpuIssue::BuiltWithoutVulkan || issue == GpuIssue::NoLoader) {
        hints.push_back("brew install vulkan-loader vulkan-headers molten-vk shaderc");
        hints.push_back("then reconfigure: cmake -B build -DCMAKE_BUILD_TYPE=Release && "
                        "cmake --build build -j");
    }
    if (issue == GpuIssue::NoDriver) {
        hints.push_back("brew install molten-vk   # provides the MoltenVK ICD");
        hints.push_back("export VK_ICD_FILENAMES=$(brew --prefix)/share/vulkan/icd.d/"
                        "MoltenVK_icd.json");
    }
#else
    if (issue != GpuIssue::Ready) hints.push_back("install the Vulkan loader and an ICD for this platform");
#endif
    if (issue != GpuIssue::Ready)
        hints.push_back("the software rasteriser needs no driver at all: SilkCrowd --software");
    return hints;
}

}  // namespace

const char* gpuIssueName(GpuIssue issue) {
    switch (issue) {
        case GpuIssue::Ready: return "ready";
        case GpuIssue::BuiltWithoutVulkan: return "build-without-vulkan";
        case GpuIssue::NoLoader: return "no-loader";
        case GpuIssue::NoDriver: return "no-driver";
        case GpuIssue::NoDevice: return "no-device";
        case GpuIssue::UnsupportedDevice: return "unsupported-device";
        default: return "unknown";
    }
}

GpuIssue classify(const GpuFacts& f) {
    // The build is checked first: if the renderer is not compiled in, nothing a driver can
    // do will make the Vulkan path available. The report still prints what the machine has,
    // so "install a driver" and "rebuild" are never confused again.
    if (!f.builtWithVulkan) return GpuIssue::BuiltWithoutVulkan;
    if (!f.loaderFound) return GpuIssue::NoLoader;
    // No instance means no driver answered - a missing ICD, a stale manifest, or a driver
    // that refuses the requested API version. Never report that as "no device".
    if (!f.instanceCreated) return GpuIssue::NoDriver;
    // A device that enumerated is the strongest evidence there is: drivers installed outside
    // the standard manifest directories (SDK builds, container images, LD_LIBRARY_PATH) still
    // answer the loader, and calling that "no driver" would be wrong.
    if (f.deviceEnumerated && !f.deviceLines.empty())
        return f.deviceUsable ? GpuIssue::Ready : GpuIssue::UnsupportedDevice;
    // No device: the manifest bookkeeping explains why.
    if (f.icdLibraries.empty()) return GpuIssue::NoDriver;  // nothing, or stale manifests
    return GpuIssue::NoDevice;
}

std::string gpuHeadline(const GpuFacts& f) {
    switch (classify(f)) {
        case GpuIssue::Ready:
            return "Vulkan device ready: " + (f.deviceName.empty() ? std::string("unknown") : f.deviceName);
        case GpuIssue::BuiltWithoutVulkan: {
            std::string s = "this build has no Vulkan renderer (configured without libvulkan headers)";
            if (f.loaderFound && !f.icdLibraries.empty())
                s += ", but this machine does have a loader and a driver: rebuilding enables it";
            return s;
        }
        case GpuIssue::NoLoader:
            return "this machine has no Vulkan loader (libvulkan.so.1); install libvulkan1";
        case GpuIssue::NoDriver:
            return "no installable Vulkan driver (ICD) found; install mesa-vulkan-drivers or the "
                   "GPU vendor driver";
        case GpuIssue::NoDevice:
            return "a Vulkan instance was created but no physical device was listed; the "
                   "container may not expose /dev/dri";
        case GpuIssue::UnsupportedDevice:
            return "no Vulkan 1.3 device with a graphics queue";
        default:
            return "the Vulkan probe could not complete";
    }
}

std::vector<std::string> gpuReportLines(const GpuFacts& f) {
    const GpuIssue issue = classify(f);
    std::vector<std::string> out;

    out.push_back("SilkCrowd Vulkan diagnostics");
    out.push_back("  " + label("build") + std::string(f.builtWithVulkan
                          ? "SILKCROWD_WITH_VULKAN=1 (GPU renderer compiled in)"
                          : "SILKCROWD_WITH_VULKAN undefined (CMake did not find libvulkan)"));
    out.push_back("  " + label("shaders") +
                  (f.shaderBlobsPresent ? "SPIR-V blobs found next to the binary"
                                        : "no SPIR-V blobs found (not needed by the software path)"));

    if (f.loaderFound) {
        out.push_back("  " + label("loader") + f.loaderPath);
        if (!f.loaderVersion.empty())
            out.push_back("  " + label("loader api") + f.loaderVersion);
    } else {
        out.push_back("  " + label("loader") + "not found (tried " +
                      (f.loaderCandidates.empty() ? std::string("nothing")
                                                  : join(f.loaderCandidates, ", ")) +
                      ")");
    }

    if (!f.icdOverride.empty()) out.push_back("  " + label("icd override") + f.icdOverride);

    if (f.icdLibraries.empty() && f.icdMissing.empty()) {
        out.push_back("  " + label("driver (ICD)") + "no manifest found (searched " +
                      (f.icdSearchDirs.empty() ? std::string("the default locations")
                                               : join(f.icdSearchDirs, ", ")) +
                      ")");
    } else {
        out.push_back("  " + label("driver (ICD)") + std::to_string(f.icdManifests.size()) +
                      " manifest(s), " + std::to_string(f.icdLibraries.size()) + " usable");
        for (const std::string& m : f.icdManifests) out.push_back("                  " + m);
        for (const std::string& lib : f.icdLibraries)
            out.push_back("                  -> " + lib);
        for (const std::string& lib : f.icdMissing)
            out.push_back("                  -> " + lib + "  (missing)");
    }

    if (!f.deviceLines.empty()) {
        out.push_back("  " + label("device") + f.deviceLines.front());
        for (std::size_t i = 1; i < f.deviceLines.size(); ++i)
            out.push_back("                  " + f.deviceLines[i]);
    } else if (!f.deviceEnumerated) {
        out.push_back("  " + label("device") + "not queried (no usable loader/driver)");
    } else {
        out.push_back("  " + label("device") + "none (vkEnumeratePhysicalDevices returned 0)");
    }

    if (!f.note.empty()) out.push_back("  " + label("note") + f.note);

    out.push_back("  " + label("verdict") + std::string(gpuIssueName(issue)) + " - " +
                  gpuHeadline(f));
    for (const std::string& hint : installHints(issue)) out.push_back("  " + label("remedy") + hint);
    return out;
}

std::string jsonStringField(const std::string& json, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    std::size_t pos = 0;
    while ((pos = json.find(quoted, pos)) != std::string::npos) {
        std::size_t i = pos + quoted.size();
        // Only a *key* counts: the next non-space character must be ':'.
        while (i < json.size() && std::isspace(static_cast<unsigned char>(json[i]))) ++i;
        if (i >= json.size() || json[i] != ':') {
            pos += quoted.size();
            continue;
        }
        ++i;
        while (i < json.size() && std::isspace(static_cast<unsigned char>(json[i]))) ++i;
        if (i >= json.size() || json[i] != '"') return std::string();
        ++i;
        std::string value;
        while (i < json.size() && json[i] != '"') {
            if (json[i] == '\\' && i + 1 < json.size()) {
                // ICD paths can contain escaped separators (Windows manifests use \\ and /).
                const char esc = json[i + 1];
                switch (esc) {
                    case 'n': value.push_back('\n'); break;
                    case 't': value.push_back('\t'); break;
                    case 'r': value.push_back('\r'); break;
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    case '/': value.push_back('/'); break;
                    default: value.push_back(esc); break;
                }
                i += 2;
                continue;
            }
            value.push_back(json[i++]);
        }
        return value;
    }
    return std::string();
}

std::string parentDirectory(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return std::string();
    if (slash == 0) return path.substr(0, 1);  // "/file"
    return path.substr(0, slash);
}

std::string resolveIcdLibraryPath(const std::string& manifestPath, const std::string& libraryPath) {
    if (libraryPath.empty()) return std::string();
    const bool absolute = libraryPath[0] == '/' || libraryPath[0] == '\\' ||
                          (libraryPath.size() > 1 && libraryPath[1] == ':');
    if (absolute) return libraryPath;
    const std::string dir = parentDirectory(manifestPath);
    if (dir.empty()) return libraryPath;
    return dir + "/" + libraryPath;
}

std::vector<std::string> splitPathList(const std::string& value) {
    std::vector<std::string> out;
    std::string current;
    for (char c : value) {
        const bool sep = c == ':' || c == ';';
        if (sep) {
            if (!current.empty()) out.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) out.push_back(current);
    return out;
}

}  // namespace silk
