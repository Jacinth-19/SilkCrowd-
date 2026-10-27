// Asset export: GLTF head model + the ten hair mesh presets with their 5 textures.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "scenes.h"

#include "silkcrowd/assets.h"
#include "silkcrowd/hair_mesh_texture.h"
#include "silkcrowd/styling.h"

namespace silk {
namespace {

// float32 -> IEEE 754 binary16 (the storage format of Table 1; it is what keeps a whole
// hair model's five textures in the 13-21 KB range of Figure 1).
uint16_t toHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t m = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) ++m;   // round to nearest
        return static_cast<uint16_t>(sign | m);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) ++m;                   // round to nearest
    if (m == 0x400u) { m = 0; ++exp; }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | m);
}

void appendHalfs(std::vector<uint8_t>& out, const std::vector<float>& v) {
    const size_t off = out.size();
    out.resize(off + v.size() * sizeof(uint16_t));
    uint16_t* p = reinterpret_cast<uint16_t*>(out.data() + off);
    for (size_t i = 0; i < v.size(); ++i) p[i] = toHalf(v[i]);
}

void appendVec3(std::vector<uint8_t>& out, const std::vector<Vec3>& v) {
    const size_t off = out.size();
    out.resize(off + v.size() * sizeof(float) * 3);
    float* p = reinterpret_cast<float*>(out.data() + off);
    for (size_t i = 0; i < v.size(); ++i) {
        p[i * 3 + 0] = v[i].x;
        p[i * 3 + 1] = v[i].y;
        p[i * 3 + 2] = v[i].z;
    }
}

}  // namespace

void writeGlif(const std::string& dir) {
    const std::string base = dir + "/female_head";
    HeadModel head;
    struct Part {
        const Mesh* mesh;
        const char* name;
    };
    const Part parts[2] = {{&head.headMesh(), "head"}, {&head.bodyMesh(), "body"}};

    std::vector<uint8_t> bin;
    std::string accessors, meshJson, bufferViews;

    int accIndex = 0, viewIndex = 0, meshIndex = 0;
    std::string meshEntries;
    for (const Part& part : parts) {
        const Mesh& m = *part.mesh;
        // positions
        std::vector<Vec3> pos = m.positions;
        Vec3 lo(1e30f), hi(-1e30f);
        for (const Vec3& p : pos) { lo = minv(lo, p); hi = maxv(hi, p); }
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "{\"bufferView\":%d,\"componentType\":5126,\"count\":%zu,\"type\":\"VEC3\","
                      "\"min\":[%.6f,%.6f,%.6f],\"max\":[%.6f,%.6f,%.6f]}",
                      viewIndex, pos.size(), lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
        accessors += (accIndex ? "," : "") + std::string(buf);
        const int posAcc = accIndex++;
        std::snprintf(buf, sizeof(buf), "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu}",
                      bin.size(), pos.size() * 12);
        bufferViews += (viewIndex ? "," : "") + std::string(buf);
        ++viewIndex;
        appendVec3(bin, pos);

        // normals
        std::snprintf(buf, sizeof(buf),
                      "{\"bufferView\":%d,\"componentType\":5126,\"count\":%zu,\"type\":\"VEC3\"}",
                      viewIndex, m.normals.size());
        accessors += std::string(",") + buf;
        const int nrmAcc = accIndex++;
        std::snprintf(buf, sizeof(buf), "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu}",
                      bin.size(), m.normals.size() * 12);
        bufferViews += std::string(",") + buf;
        ++viewIndex;
        appendVec3(bin, m.normals);

        // indices
        const size_t idxBytes = m.indices.size() * sizeof(uint32_t);
        std::snprintf(buf, sizeof(buf),
                      "{\"bufferView\":%d,\"componentType\":5125,\"count\":%zu,\"type\":\"SCALAR\"}",
                      viewIndex, m.indices.size());
        accessors += std::string(",") + buf;
        const int idxAcc = accIndex++;
        std::snprintf(buf, sizeof(buf), "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu}",
                      bin.size(), idxBytes);
        bufferViews += std::string(",") + buf;
        ++viewIndex;
        const size_t off = bin.size();
        bin.resize(off + idxBytes);
        std::memcpy(bin.data() + off, m.indices.data(), idxBytes);

        std::snprintf(buf, sizeof(buf),
                      "%s{\"name\":\"%s\",\"primitives\":[{\"attributes\":{\"POSITION\":%d,"
                      "\"NORMAL\":%d},\"indices\":%d,\"material\":0}]}",
                      meshEntries.empty() ? "" : ",", part.name, posAcc, nrmAcc, idxAcc);
        meshEntries += buf;
        ++meshIndex;
    }

    // Write the binary buffer.
    FILE* fb = std::fopen((base + ".bin").c_str(), "wb");
    if (fb) {
        std::fwrite(bin.data(), 1, bin.size(), fb);
        std::fclose(fb);
    }

    char json[8192];
    std::snprintf(json, sizeof(json),
                  "{\n  \"asset\": {\"version\": \"2.0\", \"generator\": \"SilkCrowd asset pipeline\"},\n"
                  "  \"scene\": 0,\n"
                  "  \"scenes\": [{\"nodes\": [0]}],\n"
                  "  \"nodes\": [{\"name\": \"female_head\", \"mesh\": 0}, {\"name\": \"body\", \"mesh\": 1}],\n"
                  "  \"materials\": [{\"name\": \"skin\", \"pbrMetallicRoughness\": "
                  "{\"baseColorFactor\": [0.66, 0.47, 0.39, 1.0], \"roughnessFactor\": 0.55, "
                  "\"metallicFactor\": 0.0}}],\n"
                  "  \"meshes\": [%s],\n"
                  "  \"bufferViews\": [%s],\n"
                  "  \"buffers\": [{\"uri\": \"female_head.bin\", \"byteLength\": %zu}],\n"
                  "  \"accessors\": [%s]\n}\n",
                  meshEntries.c_str(), bufferViews.c_str(), bin.size(), accessors.c_str());
    FILE* fj = std::fopen((base + ".gltf").c_str(), "wb");
    if (fj) {
        std::fwrite(json, 1, std::strlen(json), fj);
        std::fclose(fj);
    }
    std::printf("wrote %s.gltf / .bin (%zu bytes of geometry)\n", base.c_str(), bin.size());
}

// .skhm container: header + the five hair mesh textures as IEEE binary16 texels
// (the 13-21 KB per model storage of Figure 1).
//   magic "SKHM", version 1, numLayers, numVertices, numFaces, layout blocksX/blocksY,
//   hairMesh (w,h,d,3), uv (w,h,2), w (w,h,d,1), u (w,h,d,3), v (w,h,d,3)
void writeHairMeshAssets(const std::string& dir) {
    HeadModel head;
    double total = 0.0;
    double crowdTotal = 0.0;
    int crowdMin = 1 << 30, crowdMax = 0;
    // Two quality levels per style:
    //   <dir>/<NN>_<name>.skhm       hero quality (root grid of the Figure 10 preset)
    //   <dir>/crowd/<NN>_<name>.skhm the Figure 1 crowd budget (13-21 KB per model)
    // Both are written with float16 texels, so the second set is exactly the storage the
    // paper reports for its crowd.
    for (int i = 0; i < HairStyleSpec::numPresets(); ++i) {
        for (int quality = 0; quality < 2; ++quality) {
        HairStyleSpec spec = quality == 0 ? HairStyleSpec::fromFigure10(i, 1u + i)
                                          : crowdStyleSpec(i, 1u + i);
        HairMesh mesh = buildHairMesh(head, spec);
        const StylingFrames frames = computeStylingFrames(mesh);
        HairMeshTextureSet tex;
        tex.build(mesh, frames, TexelFormat::Float16);
        const auto size = tex.sizeReport(TexelFormat::Float16);
        total += size.totalKB();

        const auto& hm = tex.hairMeshTexture();
        const auto& uv = tex.uvTexture();
        const auto& wt = tex.wTexture();
        const auto& ut = tex.uTexture();
        const auto& vt = tex.vTexture();

        std::vector<uint8_t> out;
        auto putU32 = [&](uint32_t v) {
            const size_t o = out.size();
            out.resize(o + 4);
            std::memcpy(out.data() + o, &v, 4);
        };
        auto putHalfs = [&](const std::vector<float>& v) { appendHalfs(out, v); };
        out.insert(out.end(), {'S', 'K', 'H', 'M'});
        putU32(2u);          // version 2: float16 texel storage
        putU32(1u);          // 1 = IEEE 754 binary16, 0 = float32
        putU32(mesh.numLayers());
        putU32(mesh.numVertices());
        putU32(mesh.numFaces());
        putU32(tex.layout().blocksX);
        putU32(tex.layout().blocksY);
        putU32(hm.width); putU32(hm.height); putU32(hm.depth);
        putU32(uv.width); putU32(uv.height); putU32(1u);
        putU32(wt.width); putU32(wt.height); putU32(wt.depth);
        putU32(ut.width); putU32(ut.height); putU32(ut.depth);
        putU32(vt.width); putU32(vt.height); putU32(vt.depth);
        putHalfs(hm.data);
        putHalfs(uv.data);
        putHalfs(wt.data);
        putHalfs(ut.data);
        putHalfs(vt.data);

        char name[512];
        if (quality == 0) {
            std::snprintf(name, sizeof(name), "%s/%02d_%s.skhm", dir.c_str(), i, spec.name);
        } else {
            std::snprintf(name, sizeof(name), "%s/crowd/%02d_%s.skhm", dir.c_str(), i, spec.name);
        }
        FILE* f = std::fopen(name, "wb");
        if (f) {
            std::fwrite(out.data(), 1, out.size(), f);
            std::fclose(f);
        }
        const double kb = static_cast<double>(out.size()) / 1024.0;
        std::printf("  %-52s %6u verts  %6.2f KB (fp16 textures)\n", name,
                    mesh.numVertices(), kb);
        if (quality == 1) {
            crowdTotal += kb;
            const int rounded = static_cast<int>(kb + 0.5);
            crowdMin = std::min(crowdMin, rounded);
            crowdMax = std::max(crowdMax, rounded);
        }
        }
    }
    std::printf("total hair mesh storage for the 10 hero presets: %.1f KB\n", total);
    std::printf("crowd presets: %.1f KB total, %d-%d KB per model (Figure 1 reports 13-21 KB)\n",
                crowdTotal, crowdMin, crowdMax);
}

}  // namespace silk
