// Vulkan availability diagnostics.
//
// The Vulkan renderer is an optional part of the build: CMake only compiles it when
// libvulkan headers are found, and even then the machine needs a loader, an installable
// client driver (ICD) and a physical device that speaks Vulkan 1.3. When any of those is
// missing the CLI silently fell back to the software rasteriser, which made a missing
// driver look like a rendering bug.
//
// This module separates the pure bookkeeping from the probing so the classification and
// the wording of the report can be unit tested without a GPU:
//
//   * gpu_diag.cpp (core)  - GpuFacts, classify(), the report formatter, JSON helpers.
//   * gpu_probe.cpp (app)  - fills GpuFacts: dlopen probe, ICD manifest scan, and the real
//                            Vulkan enumeration when the binary was built with it.
//
// "No driver" is a machine problem, not a paper problem: the software rasteriser always
// works, so the report always says which of the five blockers was hit and what fixes it.
#pragma once

#include <string>
#include <vector>

namespace silk {

// The first thing that stops the Vulkan path, in the order the app hits it.
enum class GpuIssue {
    Ready,               // loader + driver + a Vulkan 1.3 device: the GPU path can run
    BuiltWithoutVulkan,  // this binary has no Vulkan renderer (headers missing at configure)
    NoLoader,            // libvulkan.so / vulkan-1.dll not present
    NoDriver,            // loader present but no ICD manifest, or its library is missing
    NoDevice,            // loader and ICD present, but vkEnumeratePhysicalDevices is empty
    UnsupportedDevice,   // a device exists but has no Vulkan 1.3 / graphics queue
    Unknown,             // the probe could not finish
};

const char* gpuIssueName(GpuIssue issue);

// Everything the probe learned. Keep it plain data: the report is a pure function of it.
struct GpuFacts {
    // ---- build ----
    bool builtWithVulkan = false;              // SILKCROWD_WITH_VULKAN was defined
    bool shaderBlobsPresent = false;           // SPIR-V next to the binary / in the source tree

    // ---- loader (libvulkan.so.1, vulkan-1.dll, libvulkan.dylib) ----
    bool loaderFound = false;
    std::string loaderPath;                    // what dlopen actually succeeded on
    std::vector<std::string> loaderCandidates;  // every name that was tried, in order
    std::string loaderVersion;                  // from vkEnumerateInstanceVersion, if reachable

    // ---- driver (ICD) ----
    std::vector<std::string> icdSearchDirs;    // directories that were scanned
    std::vector<std::string> icdManifests;     // *.json manifests that exist
    std::vector<std::string> icdLibraries;     // manifest library_path that resolves to a file
    std::vector<std::string> icdMissing;       // manifest library_path that does not exist
    std::string icdOverride;                   // VK_DRIVER_FILES / VK_ICD_FILENAMES, if set

    // ---- device ----
    bool instanceCreated = false;              // a driver answered vkCreateInstance
    bool deviceEnumerated = false;             // the loader answered vkEnumeratePhysicalDevices
    std::vector<std::string> deviceLines;      // one human-readable line per physical device
    std::string deviceName;                    // the device the renderer would pick
    std::string apiVersion;                    // its apiVersion, e.g. "1.3.290"
    bool meshShaders = false;
    bool deviceUsable = false;                 // Vulkan 1.3 + a graphics queue

    std::string note;                          // anything unusual worth printing verbatim
};

// The first blocker, in the order the app hits it (see GpuIssue).
GpuIssue classify(const GpuFacts& facts);

// One line, suitable for the "falling back to the software renderer" message.
std::string gpuHeadline(const GpuFacts& facts);

// The full multi-line report, indented the way the CLI prints it. Ends with the remedy.
std::vector<std::string> gpuReportLines(const GpuFacts& facts);

// ---- pure helpers (unit tested in tests/tests.cpp) ----

// Reads a top-level or nested `"key": "value"` string out of JSON text. Returns "" when the
// key is absent or its value is not a string. Tolerates whitespace and escaped quotes.
std::string jsonStringField(const std::string& json, const std::string& key);

// Vulkan ICD manifests store library_path either absolute or relative to the manifest's
// own directory. Returns the joined path; absolute input is returned unchanged.
std::string resolveIcdLibraryPath(const std::string& manifestPath, const std::string& libraryPath);

// Directory part of a path (no trailing separator), or "" when it has none.
std::string parentDirectory(const std::string& path);

// Splits a ':' separated list (on Windows ';' as well) into its non-empty entries.
std::vector<std::string> splitPathList(const std::string& value);

}  // namespace silk
