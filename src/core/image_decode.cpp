// PNG/JPEG decoding for the asset pipeline.
//
// stb_image is vendored in third_party/stb (public domain / MIT, see
// third_party/stb/LICENSE-stb_image.txt) and compiled here in its own translation unit so
// that the ~9000-line implementation is exposed through one small function and never
// leaks into the rest of the build.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "stb_image.h"

#include <vector>

#include "silkcrowd/gltf.h"

namespace silk {

bool decodeImageRgb(const unsigned char* data, size_t size, int* width, int* height,
                    std::vector<unsigned char>* rgb) {
    if (!data || size == 0) return false;
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load_from_memory(data, static_cast<int>(size), &w, &h,
                                                  &channels, 3);
    if (!pixels) return false;
    if (width) *width = w;
    if (height) *height = h;
    if (rgb) rgb->assign(pixels, pixels + static_cast<size_t>(w) * h * 3);
    stbi_image_free(pixels);
    return true;
}

}  // namespace silk
