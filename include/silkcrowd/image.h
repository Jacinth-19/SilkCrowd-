// Minimal image output (PNG + PPM) with no external dependencies, used by the
// headless screenshot mode and the CI workflows.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "silkcrowd/math.h"

namespace silk {

struct Image {
    int width = 0, height = 0;
    std::vector<Vec3> pixels;  // linear RGB

    void resize(int w, int h) {
        width = w; height = h;
        pixels.assign(static_cast<size_t>(w) * h, Vec3(0, 0, 0));
    }
    Vec3& at(int x, int y) { return pixels[static_cast<size_t>(y) * width + x]; }
    const Vec3& at(int x, int y) const { return pixels[static_cast<size_t>(y) * width + x]; }
};

// ACES-ish filmic tone curve + sRGB encoding.
Vec3 tonemap(Vec3 linear, float exposure = 1.0f);
uint8_t encodeSrgb(float linear);

// Writes a PNG (8-bit RGB). Uses stored deflate blocks so no zlib is required.
bool writePng(const std::string& path, const Image& image, float exposure = 1.0f);
bool writePpm(const std::string& path, const Image& image);

}  // namespace silk
