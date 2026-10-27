#include "silkcrowd/gltf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

namespace silk {
namespace {

// =============================================================================
//  A small JSON reader.  glTF's JSON chunk is the only JSON we ever parse, so this
//  handles the value types the specification uses and nothing more: objects, arrays,
//  strings (with the four escapes that appear in practice), numbers, booleans, null.
// =============================================================================
struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> members;

    bool isObject() const { return type == Type::Object; }
    bool isArray() const { return type == Type::Array; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }

    const Json* find(const char* key) const {
        for (const auto& m : members)
            if (m.first == key) return &m.second;
        return nullptr;
    }
    const Json* at(size_t i) const { return i < items.size() ? &items[i] : nullptr; }
    double num(double fallback = 0.0) const { return isNumber() ? number : fallback; }
    int integer(int fallback = -1) const {
        return isNumber() ? static_cast<int>(std::lround(number)) : fallback;
    }
    std::string str(const char* fallback = "") const { return isString() ? text : fallback; }
};

class JsonParser {
public:
    JsonParser(const char* begin, const char* end) : p_(begin), end_(end) {}

    bool parse(Json* out, std::string* error) {
        skipSpace();
        if (!value(out, error)) return false;
        skipSpace();
        // Trailing NUL padding is legal in a GLB JSON chunk; anything else is not.
        while (p_ < end_ && (*p_ == '\0' || *p_ == ' ' || *p_ == '\n' || *p_ == '\r' ||
                             *p_ == '\t'))
            ++p_;
        if (p_ != end_) {
            *error = "trailing bytes after the glTF JSON";
            return false;
        }
        return true;
    }

private:
    void skipSpace() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
    }

    bool literal(const char* word) {
        const size_t n = std::strlen(word);
        if (static_cast<size_t>(end_ - p_) < n || std::strncmp(p_, word, n) != 0) return false;
        p_ += n;
        return true;
    }

    bool string(std::string* out, std::string* error) {
        if (p_ >= end_ || *p_ != '"') {
            *error = "expected a JSON string";
            return false;
        }
        ++p_;
        out->clear();
        while (p_ < end_) {
            const char c = *p_++;
            if (c == '"') return true;
            if (c != '\\') {
                out->push_back(c);
                continue;
            }
            if (p_ >= end_) break;
            const char e = *p_++;
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    if (end_ - p_ < 4) {
                        *error = "truncated \\u escape";
                        return false;
                    }
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = *p_++;
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                        else {
                            *error = "bad \\u escape";
                            return false;
                        }
                    }
                    // The generator name is the only non-ASCII string we read; encode as
                    // UTF-8 and let the renderer ignore anything it cannot display.
                    if (code < 0x80) {
                        out->push_back(static_cast<char>(code));
                    } else if (code < 0x800) {
                        out->push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out->push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default:
                    *error = "unsupported escape in JSON string";
                    return false;
            }
        }
        *error = "unterminated JSON string";
        return false;
    }

    bool value(Json* out, std::string* error) {
        if (p_ >= end_) {
            *error = "unexpected end of JSON";
            return false;
        }
        const char c = *p_;
        if (c == '{') {
            ++p_;
            out->type = Json::Type::Object;
            skipSpace();
            if (p_ < end_ && *p_ == '}') {
                ++p_;
                return true;
            }
            while (true) {
                skipSpace();
                std::string key;
                if (!string(&key, error)) return false;
                skipSpace();
                if (p_ >= end_ || *p_ != ':') {
                    *error = "expected ':' in JSON object";
                    return false;
                }
                ++p_;
                skipSpace();
                Json child;
                if (!value(&child, error)) return false;
                out->members.emplace_back(std::move(key), std::move(child));
                skipSpace();
                if (p_ < end_ && *p_ == ',') {
                    ++p_;
                    continue;
                }
                if (p_ < end_ && *p_ == '}') {
                    ++p_;
                    return true;
                }
                *error = "expected ',' or '}' in JSON object";
                return false;
            }
        }
        if (c == '[') {
            ++p_;
            out->type = Json::Type::Array;
            skipSpace();
            if (p_ < end_ && *p_ == ']') {
                ++p_;
                return true;
            }
            while (true) {
                skipSpace();
                Json child;
                if (!value(&child, error)) return false;
                out->items.push_back(std::move(child));
                skipSpace();
                if (p_ < end_ && *p_ == ',') {
                    ++p_;
                    continue;
                }
                if (p_ < end_ && *p_ == ']') {
                    ++p_;
                    return true;
                }
                *error = "expected ',' or ']' in JSON array";
                return false;
            }
        }
        if (c == '"') {
            out->type = Json::Type::String;
            return string(&out->text, error);
        }
        if (literal("true")) {
            out->type = Json::Type::Bool;
            out->boolean = true;
            return true;
        }
        if (literal("false")) {
            out->type = Json::Type::Bool;
            out->boolean = false;
            return true;
        }
        if (literal("null")) {
            out->type = Json::Type::Null;
            return true;
        }
        // -?digits[.digits][e[+-]digits]
        const char* start = p_;
        if (p_ < end_ && (*p_ == '-' || *p_ == '+')) ++p_;
        bool digits = false;
        while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
            ++p_;
            digits = true;
        }
        if (p_ < end_ && *p_ == '.') {
            ++p_;
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
                ++p_;
                digits = true;
            }
        }
        if (digits && p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
            ++p_;
            if (p_ < end_ && (*p_ == '-' || *p_ == '+')) ++p_;
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (!digits) {
            *error = "invalid JSON number";
            return false;
        }
        out->type = Json::Type::Number;
        out->number = std::strtod(std::string(start, static_cast<size_t>(p_ - start)).c_str(),
                                  nullptr);
        return true;
    }

    const char* p_ = nullptr;
    const char* end_ = nullptr;
};

// =============================================================================
//  GLB container
// =============================================================================
struct GlbChunks {
    const char* json = nullptr;
    size_t jsonSize = 0;
    const unsigned char* binary = nullptr;
    size_t binarySize = 0;
};

bool readFile(const std::string& path, std::vector<unsigned char>* out, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *error = "cannot open " + path;
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size <= 0) {
        *error = path + " is empty";
        return false;
    }
    out->resize(static_cast<size_t>(size));
    if (!in.read(reinterpret_cast<char*>(out->data()), size)) {
        *error = "short read on " + path;
        return false;
    }
    return true;
}

uint32_t readU32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

bool splitGlb(const std::vector<unsigned char>& data, GlbChunks* chunks, std::string* error) {
    if (data.size() < 20) {
        *error = "file is too small to be a GLB";
        return false;
    }
    if (readU32(data.data()) != 0x46546C67u) {  // 'glTF'
        *error = "not a binary glTF (missing 'glTF' magic): .gltf + external .bin is not "
                 "supported, use a GLB";
        return false;
    }
    const uint32_t version = readU32(data.data() + 4);
    if (version != 2) {
        *error = "only GLB version 2 is supported";
        return false;
    }
    const uint32_t total = readU32(data.data() + 8);
    const size_t limit = std::min<size_t>(total, data.size());
    size_t off = 12;
    bool gotJson = false;
    while (off + 8 <= limit) {
        const uint32_t chunkLen = readU32(data.data() + off);
        const uint32_t chunkType = readU32(data.data() + off + 4);
        off += 8;
        if (off + chunkLen > data.size()) {
            *error = "GLB chunk overruns the file";
            return false;
        }
        if (chunkType == 0x4E4F534Au) {  // 'JSON'
            chunks->json = reinterpret_cast<const char*>(data.data() + off);
            chunks->jsonSize = chunkLen;
            gotJson = true;
        } else if (chunkType == 0x004E4942u) {  // 'BIN\0'
            chunks->binary = data.data() + off;
            chunks->binarySize = chunkLen;
        }
        off += chunkLen;
    }
    if (!gotJson) {
        *error = "GLB has no JSON chunk";
        return false;
    }
    return true;
}

// =============================================================================
//  Accessors
// =============================================================================
int componentSize(int componentType) {
    switch (componentType) {
        case 5120: case 5121: return 1;   // BYTE, UNSIGNED_BYTE
        case 5122: case 5123: return 2;   // SHORT, UNSIGNED_SHORT
        case 5125: case 5126: return 4;   // UNSIGNED_INT, FLOAT
        default: return 0;
    }
}

int componentCount(const std::string& type) {
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    if (type == "MAT4") return 16;
    return 0;
}

// Reads an accessor into floats (positions/normals/UVs) or uint32s (indices).
class AccessorReader {
public:
    AccessorReader(const Json& root, const GlbChunks& chunks) : root_(root), chunks_(chunks) {}

    bool view(const Json& accessor, const unsigned char** base, size_t* stride,
              int* count, int* components, int* componentType, std::string* error) const {
        *count = accessor.find("count") ? accessor.find("count")->integer() : -1;
        if (*count < 0) {
            *error = "accessor has no count";
            return false;
        }
        *componentType = accessor.find("componentType")
                             ? accessor.find("componentType")->integer()
                             : 0;
        *components = componentCount(accessor.find("type") ? accessor.find("type")->str() : "");
        if (componentSize(*componentType) == 0 || *components == 0) {
            *error = "unsupported accessor componentType/type";
            return false;
        }
        if (!accessor.find("bufferView")) {
            *error = "sparse or bufferless accessors are not supported";
            return false;
        }
        const int viewIndex = accessor.find("bufferView")->integer(-1);
        const Json* views = root_.find("bufferViews");
        const Json* bv = views ? views->at(static_cast<size_t>(viewIndex)) : nullptr;
        if (!bv) {
            *error = "accessor references a missing bufferView";
            return false;
        }
        const int buffer = bv->find("buffer") ? bv->find("buffer")->integer() : -1;
        if (buffer != 0) {
            *error = "only the GLB's embedded buffer is supported";
            return false;
        }
        const size_t viewOffset = static_cast<size_t>(bv->find("byteOffset")
                                                          ? bv->find("byteOffset")->integer(0)
                                                          : 0);
        const size_t offset = viewOffset + static_cast<size_t>(
                                              accessor.find("byteOffset")
                                                  ? accessor.find("byteOffset")->integer(0)
                                                  : 0);
        const size_t elementSize = static_cast<size_t>(componentSize(*componentType)) *
                                   static_cast<size_t>(*components);
        *stride = bv->find("byteStride") && bv->find("byteStride")->isNumber()
                      ? static_cast<size_t>(bv->find("byteStride")->integer())
                      : elementSize;
        if (offset + elementSize * static_cast<size_t>(*count - 1) + elementSize >
            chunks_.binarySize) {
            *error = "accessor overruns the binary chunk";
            return false;
        }
        *base = chunks_.binary + offset;
        return true;
    }

    bool readVec2(const Json& accessor, std::vector<Vec2>* out, std::string* error) const {
        const unsigned char* base = nullptr;
        size_t stride = 0;
        int count = 0, comps = 0, ctype = 0;
        if (!view(accessor, &base, &stride, &count, &comps, &ctype, error)) return false;
        if (comps != 2) {
            *error = "TEXCOORD_0 is not a VEC2";
            return false;
        }
        out->resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            (*out)[static_cast<size_t>(i)] = Vec2(fetch(base + i * stride, 0, ctype),
                                                  fetch(base + i * stride, 1, ctype));
        return true;
    }

    bool readVec3(const Json& accessor, std::vector<Vec3>* out, std::string* error) const {
        const unsigned char* base = nullptr;
        size_t stride = 0;
        int count = 0, comps = 0, ctype = 0;
        if (!view(accessor, &base, &stride, &count, &comps, &ctype, error)) return false;
        if (comps != 3) {
            *error = "expected a VEC3 accessor";
            return false;
        }
        out->resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const unsigned char* e = base + i * stride;
            (*out)[static_cast<size_t>(i)] = Vec3(fetch(e, 0, ctype), fetch(e, 1, ctype),
                                                  fetch(e, 2, ctype));
        }
        return true;
    }

    bool readIndices(const Json& accessor, std::vector<uint32_t>* out,
                     std::string* error) const {
        const unsigned char* base = nullptr;
        size_t stride = 0;
        int count = 0, comps = 0, ctype = 0;
        if (!view(accessor, &base, &stride, &count, &comps, &ctype, error)) return false;
        if (comps != 1) {
            *error = "indices are not SCALAR";
            return false;
        }
        out->resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const unsigned char* e = base + i * stride;
            uint32_t v = 0;
            switch (ctype) {
                case 5121: v = e[0]; break;
                case 5123: v = static_cast<uint32_t>(e[0]) | (static_cast<uint32_t>(e[1]) << 8); break;
                case 5125: v = readU32(e); break;
                default:
                    *error = "unsupported index componentType";
                    return false;
            }
            (*out)[static_cast<size_t>(i)] = v;
        }
        return true;
    }

private:
    static float fetch(const unsigned char* element, int component, int ctype) {
        switch (ctype) {
            case 5126: {
                float f = 0.0f;
                const unsigned char* p = element + component * 4;
                unsigned bits = static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8) |
                                (static_cast<unsigned>(p[2]) << 16) |
                                (static_cast<unsigned>(p[3]) << 24);
                std::memcpy(&f, &bits, 4);
                return f;
            }
            case 5123: {
                const unsigned char* p = element + component * 2;
                return static_cast<float>(static_cast<uint32_t>(p[0]) |
                                          (static_cast<uint32_t>(p[1]) << 8));
            }
            case 5121: return static_cast<float>(element[component]);
            default: return 0.0f;
        }
    }

    const Json& root_;
    const GlbChunks& chunks_;
};

}  // namespace

bool loadGlb(const std::string& path, GltfAsset* asset, std::string* errorStorage) {
    std::string localError;
    std::string* error = errorStorage ? errorStorage : &localError;
    error->clear();
    if (!asset) {
        *error = "no output asset";
        return false;
    }

    std::vector<unsigned char> file;
    if (!readFile(path, &file, error)) return false;
    GlbChunks chunks;
    if (!splitGlb(file, &chunks, error)) return false;

    Json root;
    {
        JsonParser parser(chunks.json, chunks.json + chunks.jsonSize);
        if (!parser.parse(&root, error)) {
            *error = "glTF JSON: " + *error;
            return false;
        }
    }
    if (!root.isObject()) {
        *error = "glTF JSON root is not an object";
        return false;
    }

    GltfAsset result;
    if (const Json* gen = root.find("asset"))
        if (const Json* g = gen->find("generator")) result.generator = g->str();

    const Json* meshes = root.find("meshes");
    const Json* accessors = root.find("accessors");
    if (!meshes || !meshes->isArray() || meshes->items.empty() || !accessors) {
        *error = "the GLB contains no mesh";
        return false;
    }
    AccessorReader reader(root, chunks);

    // Merge every primitive of the first mesh into one indexed triangle list; a scan is
    // normally a single primitive, but merging keeps the loader honest for split meshes.
    const Json* primitives = meshes->at(0)->find("primitives");
    if (!primitives || !primitives->isArray()) {
        *error = "the first mesh has no primitives";
        return false;
    }
    int albedoImageIndex = -1;
    for (size_t p = 0; p < primitives->items.size(); ++p) {
        const Json& prim = primitives->items[p];
        const Json* attrs = prim.find("attributes");
        if (!attrs) {
            *error = "a primitive has no attributes";
            return false;
        }
        const Json* positionAccessor = attrs->find("POSITION");
        if (!positionAccessor) {
            *error = "a primitive has no POSITION";
            return false;
        }
        std::vector<Vec3> positions, normals;
        std::vector<Vec2> uvs;
        std::vector<uint32_t> indices;
        if (!reader.readVec3(*accessors->at(static_cast<size_t>(positionAccessor->integer())),
                             &positions, error))
            return false;
        if (const Json* na = attrs->find("NORMAL")) {
            if (!reader.readVec3(*accessors->at(static_cast<size_t>(na->integer())), &normals,
                                 error))
                return false;
        }
        if (const Json* ta = attrs->find("TEXCOORD_0")) {
            if (!reader.readVec2(*accessors->at(static_cast<size_t>(ta->integer())), &uvs,
                                 error))
                return false;
        }
        if (const Json* ia = prim.find("indices")) {
            if (!reader.readIndices(*accessors->at(static_cast<size_t>(ia->integer())),
                                    &indices, error))
                return false;
        } else {
            indices.resize(positions.size());
            for (size_t i = 0; i < positions.size(); ++i)
                indices[i] = static_cast<uint32_t>(i);
        }

        const uint32_t base = static_cast<uint32_t>(result.mesh.positions.size());
        result.mesh.positions.insert(result.mesh.positions.end(), positions.begin(),
                                     positions.end());
        if (normals.size() == positions.size())
            result.mesh.normals.insert(result.mesh.normals.end(), normals.begin(), normals.end());
        else
            result.mesh.normals.resize(result.mesh.positions.size(), Vec3(0, 1, 0));
        if (uvs.size() == positions.size())
            result.mesh.uvs.insert(result.mesh.uvs.end(), uvs.begin(), uvs.end());
        else
            result.mesh.uvs.resize(result.mesh.positions.size(), Vec2(0, 0));
        for (uint32_t index : indices) result.mesh.indices.push_back(base + index);

        if (albedoImageIndex < 0 && prim.find("material")) {
            const int material = prim.find("material")->integer(-1);
            const Json* materials = root.find("materials");
            const Json* materialJson = materials ? materials->at(static_cast<size_t>(material))
                                                 : nullptr;
            if (materialJson) {
                const Json* pbr = materialJson->find("pbrMetallicRoughness");
                const Json* baseColor = pbr ? pbr->find("baseColorTexture") : nullptr;
                if (baseColor && baseColor->find("index")) {
                    const int texture = baseColor->find("index")->integer(-1);
                    const Json* textures = root.find("textures");
                    const Json* textureJson =
                        textures ? textures->at(static_cast<size_t>(texture)) : nullptr;
                    if (textureJson && textureJson->find("source"))
                        albedoImageIndex = textureJson->find("source")->integer(-1);
                }
            }
        }
    }
    if (result.mesh.positions.empty() || result.mesh.indices.size() < 3) {
        *error = "the GLB mesh is empty";
        return false;
    }
    result.mesh.name = meshes->at(0)->find("name") ? meshes->at(0)->find("name")->str() : "head";
    if (result.mesh.normals.size() != result.mesh.positions.size()) result.mesh.computeNormals();
    result.mesh.materialIds.assign(result.mesh.triangleCount(), 0);

    // ---- base-colour texture -------------------------------------------------
    if (albedoImageIndex >= 0) {
        const Json* images = root.find("images");
        const Json* imageJson = images ? images->at(static_cast<size_t>(albedoImageIndex))
                                       : nullptr;
        const Json* views = root.find("bufferViews");
        if (imageJson && imageJson->find("bufferView") && views) {
            const Json* bv = views->at(static_cast<size_t>(
                imageJson->find("bufferView")->integer(-1)));
            if (bv) {
                const size_t offset = static_cast<size_t>(bv->find("byteOffset")
                                                              ? bv->find("byteOffset")->integer(0)
                                                              : 0);
                const size_t length = static_cast<size_t>(
                    bv->find("byteLength") ? bv->find("byteLength")->integer(0) : 0);
                if (offset + length <= chunks.binarySize && length > 0) {
                    int w = 0, h = 0;
                    std::vector<unsigned char> rgb;
                    if (decodeImageRgb(chunks.binary + offset, length, &w, &h, &rgb) &&
                        w > 0 && h > 0) {
                        result.albedo.resize(w, h);
                        for (int y = 0; y < h; ++y) {
                            for (int x = 0; x < w; ++x) {
                                const unsigned char* texel =
                                    &rgb[(static_cast<size_t>(y) * w + x) * 3];
                                // glTF base-colour textures are sRGB; the renderer works in
                                // linear light, so decode here once per load.
                                result.albedo.at(x, y) =
                                    Vec3(srgbToLinear(texel[0] / 255.0f),
                                         srgbToLinear(texel[1] / 255.0f),
                                         srgbToLinear(texel[2] / 255.0f));
                            }
                        }
                        result.hasAlbedo = true;
                    }
                }
            }
        }
    }

    *asset = std::move(result);
    return true;
}

}  // namespace silk
