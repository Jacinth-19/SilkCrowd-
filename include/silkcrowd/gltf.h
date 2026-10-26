// Minimal glTF 2.0 / GLB reader for scanned head assets.
//
// The repository ships `assets/heads/head_scan.glb`: a photogrammetry bust (geometry +
// a single base-colour texture) repacked by `tools/prep_head.py`. Everything the renderer
// needs from it is one indexed triangle mesh with UVs and one sRGB texture, so this loader
// implements exactly that subset of the specification: the GLB container, the JSON chunk,
// accessors with an optional byteStride, and the base-colour texture of the first
// material that has one. Skins, animations, morph targets, sparse accessors, external
// buffers and .gltf+base64 are rejected with an error string rather than silently
// half-loaded -- a head that loads wrong is worse than a head that does not load.
#pragma once

#include <string>

#include "silkcrowd/assets.h"
#include "silkcrowd/image.h"

namespace silk {

struct GltfAsset {
    Mesh mesh;
    Image albedo;            // linear RGB, mip level 0
    bool hasAlbedo = false;
    std::string generator;
};

// Returns false and fills |error| if the file cannot be read or uses an unsupported
// feature. `asset` is left untouched on failure.
bool loadGlb(const std::string& path, GltfAsset* asset, std::string* error = nullptr);

// Decodes a PNG/JPEG from memory into 8-bit RGB (used for the base-colour image and by
// the texture pipeline). Implemented in src/core/image_decode.cpp on top of stb_image.
bool decodeImageRgb(const unsigned char* data, size_t size, int* width, int* height,
                    std::vector<unsigned char>* rgb);

}  // namespace silk
