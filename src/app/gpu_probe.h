// App-side Vulkan probing: fills GpuFacts and prints the diagnostics report.
#pragma once

#include <cstdio>

#include "silkcrowd/gpu_diag.h"

namespace silk {

// Probes the build, the loader, the ICD manifests and (when compiled in) the physical
// devices. Cheap: a handful of stat() calls plus one instance create/destroy.
GpuFacts probeGpu();

// Prints gpuReportLines() to |out| (stdout for --gpu-info, stderr for warnings).
void printGpuReport(const GpuFacts& facts, FILE* out);

// True when the Vulkan path would actually run (GpuIssue::Ready).
bool gpuUsable(const GpuFacts& facts);

}  // namespace silk
