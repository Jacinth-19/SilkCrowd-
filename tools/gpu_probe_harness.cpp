// Drives probeGpu()/printGpuReport() directly, so tools/gpu_probe_selftest.sh can exercise
// the Vulkan branch of the diagnostics without a GPU. Built with SILKCROWD_WITH_VULKAN=1.
#include <cstdio>

#include "gpu_probe.h"

int main() {
    const silk::GpuFacts facts = silk::probeGpu();
    silk::printGpuReport(facts, stdout);
    return silk::gpuUsable(facts) ? 0 : 3;
}
