#include "silkcrowd/image.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace silk {
namespace {

uint32_t crcTable[256];
bool crcInit = false;

void initCrc() {
    for (uint32_t n = 0; n < 256; ++n) {
        uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crcTable[n] = c;
    }
    crcInit = true;
}

uint32_t crc32Buf(const uint8_t* buf, size_t len, uint32_t crc = 0xFFFFFFFFu) {
    if (!crcInit) initCrc();
    for (size_t i = 0; i < len; ++i) crc = crcTable[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

void writeChunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    put32(out, static_cast<uint32_t>(data.size()));
    const uint8_t* t = reinterpret_cast<const uint8_t*>(type);
    out.insert(out.end(), t, t + 4);
    out.insert(out.end(), data.begin(), data.end());
    uint32_t crc = crc32Buf(t, 4);
    if (!data.empty()) crc = crc32Buf(data.data(), data.size(), crc ^ 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
    else crc ^= 0xFFFFFFFFu;
    // crc32 over type+data starting from 0: recompute cleanly (the incremental form
    // above is equivalent, but recomputing avoids subtle state bugs).
    std::vector<uint8_t> all(t, t + 4);
    all.insert(all.end(), data.begin(), data.end());
    crc = crc32Buf(all.data(), all.size()) ^ 0xFFFFFFFFu;
    put32(out, crc);
}

uint32_t adler32(const uint8_t* data, size_t len) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

}  // namespace

uint8_t encodeSrgb(float linear) {
    const float v = saturate(linear);
    const float s = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    return static_cast<uint8_t>(clampf(s * 255.0f + 0.5f, 0.0f, 255.0f));
}

Vec3 tonemap(Vec3 c, float exposure) {
    c = c * exposure;
    // ACES fitted (Narkowicz)
    auto curve = [](float x) {
        const float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
        return saturate((x * (a * x + b)) / (x * (cc * x + d) + e));
    };
    return Vec3(curve(c.x), curve(c.y), curve(c.z));
}

bool writePng(const std::string& path, const Image& image, float exposure) {
    if (image.width <= 0 || image.height <= 0) return false;
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(image.height) * (image.width * 3 + 1));
    for (int y = 0; y < image.height; ++y) {
        raw.push_back(0);  // filter type 0 (None)
        for (int x = 0; x < image.width; ++x) {
            const Vec3 c = tonemap(image.at(x, y), exposure);
            raw.push_back(encodeSrgb(c.x));
            raw.push_back(encodeSrgb(c.y));
            raw.push_back(encodeSrgb(c.z));
        }
    }

    // zlib stream with stored deflate blocks.
    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    const size_t maxBlock = 65535;
    for (size_t off = 0; off < raw.size(); off += maxBlock) {
        const size_t n = std::min(maxBlock, raw.size() - off);
        const bool last = (off + n >= raw.size());
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n & 0xFF));
        z.push_back(static_cast<uint8_t>(n >> 8));
        const uint16_t nlen = static_cast<uint16_t>(~n);
        z.push_back(static_cast<uint8_t>(nlen & 0xFF));
        z.push_back(static_cast<uint8_t>(nlen >> 8));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
    }
    const uint32_t ad = adler32(raw.data(), raw.size());
    z.push_back(static_cast<uint8_t>(ad >> 24));
    z.push_back(static_cast<uint8_t>(ad >> 16));
    z.push_back(static_cast<uint8_t>(ad >> 8));
    z.push_back(static_cast<uint8_t>(ad));

    std::vector<uint8_t> out;
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    out.insert(out.end(), sig, sig + 8);

    std::vector<uint8_t> ihdr;
    put32(ihdr, static_cast<uint32_t>(image.width));
    put32(ihdr, static_cast<uint32_t>(image.height));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // colour type RGB
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    writeChunk(out, "IHDR", ihdr);
    writeChunk(out, "IDAT", z);
    writeChunk(out, "IEND", {});

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return true;
}

bool writePpm(const std::string& path, const Image& image) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", image.width, image.height);
    for (int y = 0; y < image.height; ++y)
        for (int x = 0; x < image.width; ++x) {
            const Vec3 c = tonemap(image.at(x, y));
            const uint8_t px[3] = {encodeSrgb(c.x), encodeSrgb(c.y), encodeSrgb(c.z)};
            std::fwrite(px, 1, 3, f);
        }
    std::fclose(f);
    return true;
}

}  // namespace silk
