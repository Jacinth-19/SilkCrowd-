#!/usr/bin/env python3
"""Repacks the Infinite-Realities head scan into the compact GLB that SilkCrowd ships.

    tools/prep_head.py <source.glb> <out.glb> [--texture-size 1024]

The upstream file (BabylonJS/Assets, `meshes/Lee-Perry-Smith-Head/head.glb`) is 27 MB: two
4096^2 PNGs, one of which is a tangent-space normal map we do not use (the scan's own normals
are geometric), plus a 4-quadrillion-texel diffuse that is pointless at portrait resolution.
This tool keeps the geometry and the diffuse map only, box-filters the diffuse down to
`--texture-size`, and writes a self-contained GLB with the same primitive layout, so the
runtime loader (`src/core/gltf.cpp`) has exactly one code path for scanned heads.

Only zlib/struct are used: the PNG codec is the same dependency-free one as
`tools/make_styles_grid.py`, so the repository needs no Python packages.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

# SilkCrowd's head space: the crown sits at CROWN_Y, the midline at x = 0 and the centroid
# of the skull at z = 0. The target skull width is the one the procedural HeadModel uses
# (2 * 0.0805 m), so the hair-mesh volume constants (stand-off, crown pad, hem lengths),
# which are all in millimetres of real-world hair, keep their meaning.
CROWN_Y = 0.128
SKULL_WIDTH = 0.161

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def read_png(data: bytes):
    assert data[:8] == PNG_MAGIC, "not a PNG"
    pos, width, height, bit_depth, color_type = 8, 0, 0, 0, 0
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    assert bit_depth == 8, "only 8-bit PNGs are supported"
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = zlib.decompress(idat)
    stride = width * channels
    rows: list[bytearray] = []
    prev = bytearray(stride)
    i = 0
    for _ in range(height):
        filt = raw[i]
        i += 1
        line = bytearray(raw[i : i + stride])
        i += stride
        if filt == 1:
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif filt == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif filt == 3:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif filt == 4:
            for x in range(stride):
                a = line[x - channels] if x >= channels else 0
                b = prev[x]
                c = prev[x - channels] if x >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 0xFF
        rows.append(line)
        prev = line
    return width, height, channels, rows


def write_png(width: int, height: int, rgb: bytes) -> bytes:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride : (y + 1) * stride])

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    return (PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def downsample(width: int, height: int, channels: int, rows, factor: int):
    """Box filter by an integer factor, in gamma space (these are 8-bit sRGB texels)."""
    out_w, out_h = width // factor, height // factor
    out = bytearray(out_w * out_h * 3)
    area = factor * factor
    for oy in range(out_h):
        base = oy * factor
        for ox in range(out_w):
            r = g = b = 0
            for dy in range(factor):
                line = rows[base + dy]
                x0 = ox * factor * channels
                for dx in range(factor):
                    i = x0 + dx * channels
                    r += line[i]
                    g += line[i + 1]
                    b += line[i + 2]
            o = (oy * out_w + ox) * 3
            out[o] = r // area
            out[o + 1] = g // area
            out[o + 2] = b // area
    return out_w, out_h, bytes(out)


def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, _length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2, "not a GLB 2.0 file"
    off, js, binary = 12, None, b""
    while off < len(data):
        clen, ctype = struct.unpack("<II", data[off : off + 8])
        off += 8
        chunk = data[off : off + clen]
        off += clen
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binary = chunk
    assert js is not None, "no JSON chunk"
    return js, binary


def accessor_bytes(js, binary, index: int) -> bytes:
    acc = js["accessors"][index]
    bv = js["bufferViews"][acc["bufferView"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    comp = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}[acc["componentType"]]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    return binary[start : start + acc["count"] * comp * ncomp]


def fit_to_silk_space(positions: bytes, vertex_count: int) -> tuple[bytes, dict]:
    """Fits the bust into SilkCrowd's head space using anatomical landmarks only.

    The scan arrives in arbitrary scale and origin (this bust is ~1.24x life size and
    floats with its shoulders at y = 0). Four landmarks are measured on the mesh itself:

      * the neck:     the narrowest horizontal slab between the shoulders and the jaw,
      * the crown:    the highest vertex,
      * the menton:   the lowest midline vertex in front of the neck,
      * the skull:    the widest horizontal slab above the ears, and its z centroid.

    The mesh is then uniformly scaled so the skull is SKULL_WIDTH wide, its midline is
    moved to x = 0, the skull centroid to z = 0 and the crown to CROWN_Y. Uniform scale -
    no anisotropic squashing - keeps the face's proportions intact, which matters because
    the same numbers are what the hair-mesh root layer is fitted against.
    """
    pts = struct.unpack(f"<{vertex_count * 3}f", positions)
    verts = [(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]) for i in range(vertex_count)]
    ys = sorted(v[1] for v in verts)
    y_lo, y_hi = ys[0], ys[-1]

    # ---- width profile, 5 mm slabs -------------------------------------------------
    slab = 0.005
    n_slabs = max(8, int((y_hi - y_lo) / slab) + 1)
    widths = [0.0] * n_slabs
    for s in range(n_slabs):
        y = y_lo + (s + 0.5) * slab
        band = [abs(v[0]) for v in verts if abs(v[1] - y) < slab * 0.6]
        widths[s] = (max(band) * 2.0) if len(band) > 8 else 0.0
    peak = max(widths)
    # The neck is the first strong local minimum walking up from the shoulders.
    neck_slab = 0
    for s in range(1, n_slabs - 1):
        if widths[s] > 0.0 and widths[s] < 0.62 * peak and \
                widths[s] <= widths[s - 1] and widths[s] <= widths[s + 1]:
            neck_slab = s
            break
    neck_cut = y_lo + neck_slab * slab
    mid_x = 0.5 * (max(v[0] for v in verts) + min(v[0] for v in verts))
    head = [v for v in verts if v[1] > neck_cut + 0.012]
    crown = max(v[1] for v in head)
    # Menton: the lowest vertex on the midline, in front of the skull, above the neck.
    z_mid = 0.5 * (min(v[2] for v in head) + max(v[2] for v in head))
    midline = [v for v in head if abs(v[0] - mid_x) < 0.014 and v[2] > z_mid]
    chin = min(v[1] for v in midline) if midline else neck_cut
    height = crown - chin

    # ---- skull width and z centroid -------------------------------------------------
    top = [v for v in head if v[1] > crown - 0.55 * height]
    x99 = sorted(abs(v[0]) for v in top)[int(0.99 * (len(top) - 1))]
    skull_width = 2.0 * x99
    uppermost = [v for v in head if v[1] > crown - 0.22 * height]
    z_centroid = sum(v[2] for v in uppermost) / len(uppermost)

    s = SKULL_WIDTH / skull_width
    out = bytearray(len(positions))
    for i, (x, y, z) in enumerate(verts):
        struct.pack_into("<fff", out, i * 12,
                         (x - mid_x) * s,
                         (y - crown) * s + CROWN_Y,
                         (z - z_centroid) * s)
    info = {
        "neck_cut": neck_cut, "crown": crown, "chin": chin, "height": height,
        "skull_width": skull_width, "scale": s,
        "new_height": height * s, "mid_x": mid_x, "z_centroid": z_centroid,
    }
    return bytes(out), info


def texture_skin_space(positions: bytes, uvs: bytes, indices: bytes, vertex_count: int,
                       index_count: int, tw: int, th: int, origin: tuple, scale: float):
    """Rasterises the mesh in texture space, returning a 3D position per texel.

    A photogrammetry texture is one atlas: to edit a *part of the face* (the beard shadow)
    the tool has to know which texel belongs to which point of the head, and the only way to
    know that is to rasterise the UV triangles. Uncovered texels keep None.
    """
    pts = struct.unpack(f"<{vertex_count * 3}f", positions)
    tex = struct.unpack(f"<{vertex_count * 2}f", uvs)
    idx = struct.unpack(f"<{index_count}H", indices)
    space: list = [None] * (tw * th)
    for t in range(0, index_count, 3):
        i0, i1, i2 = idx[t], idx[t + 1], idx[t + 2]
        u = [tex[i0 * 2] * tw, tex[i1 * 2] * tw, tex[i2 * 2] * tw]
        v = [tex[i0 * 2 + 1] * th, tex[i1 * 2 + 1] * th, tex[i2 * 2 + 1] * th]
        area = (u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0])
        if abs(area) < 1e-9:
            continue
        inv = 1.0 / area
        x0 = max(0, int(min(u))); x1 = min(tw - 1, int(max(u)) + 1)
        y0 = max(0, int(min(v))); y1 = min(th - 1, int(max(v)) + 1)
        p = [(pts[i0 * 3], pts[i0 * 3 + 1], pts[i0 * 3 + 2]),
             (pts[i1 * 3], pts[i1 * 3 + 1], pts[i1 * 3 + 2]),
             (pts[i2 * 3], pts[i2 * 3 + 1], pts[i2 * 3 + 2])]
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                sx, sy = px + 0.5, py + 0.5
                b0 = ((u[1] - sx) * (v[2] - sy) - (u[2] - sx) * (v[1] - sy)) * inv
                b1 = ((u[2] - sx) * (v[0] - sy) - (u[0] - sx) * (v[2] - sy)) * inv
                b2 = 1.0 - b0 - b1
                if b0 < 0 or b1 < 0 or b2 < 0:
                    continue
                space[py * tw + px] = (b0 * p[0][0] + b1 * p[1][0] + b2 * p[2][0],
                                       b0 * p[0][1] + b1 * p[1][1] + b2 * p[2][1],
                                       b0 * p[0][2] + b1 * p[1][2] + b2 * p[2][2])
    return space


def digital_shave(texels: bytearray, space: list, tw: int, th: int) -> int:
    """Removes the five-o'clock shadow from the scan's texture, in texture space.

    The scan was taken with a day's growth on the jaw, and a beard shadow is the strongest
    single "this is a man" cue on the face. The follicle detail is kept -- it is what makes
    the skin read as skin at portrait distance -- but the blue-grey of the stubble is
    replaced by the complexion measured on the cheek, so what is left is the fine texture
    of a shaved face. Lips are protected by their saturation: they are the only strongly
    red pixels in the region.
    """
    # Complexion reference: the cheek, sampled where the mask definitely does not apply.
    ref_r = ref_g = ref_b = 0.0
    n = 0
    for i in range(0, tw * th, 7):
        p = space[i]
        if p is None:
            continue
        x, y, z = p
        if -0.01 > y > -0.035 and 0.02 < z and 0.03 < abs(x) < 0.07:
            o = i * 3
            ref_r += texels[o]; ref_g += texels[o + 1]; ref_b += texels[o + 2]
            n += 1
    if n < 50:
        return 0
    ref = (ref_r / n, ref_g / n, ref_b / n)
    touched = 0
    for i, p in enumerate(space):
        if p is None:
            continue
        x, y, z = p
        # Jaw, chin and moustache: in front of the ears, below the nose, above the chin
        # underside, and fattened out towards the cheeks by the second term.
        if not (-0.108 < y < 0.010 and z > -0.015):
            continue
        if abs(x) > 0.085:
            continue
        edge = min(1.0, (0.010 - y) / 0.012) * min(1.0, (y + 0.108) / 0.014)
        edge *= min(1.0, (z + 0.015) / 0.014)
        if edge <= 0.0:
            continue
        # The lips keep their colour: they are a *form* on the face, not a hair shadow,
        # and re-tinting them would leave a mannequin mouth.
        if abs(x) < 0.031 and -0.059 < y < -0.027 and z > 0.068:
            continue
        o = i * 3
        r, g, b = texels[o], texels[o + 1], texels[o + 2]
        # Keep the texture's detail, replace its colour: luminance is preserved at 45 %.
        luma = (0.30 * r + 0.59 * g + 0.11 * b) / 255.0
        ref_luma = (0.30 * ref[0] + 0.59 * ref[1] + 0.11 * ref[2]) / 255.0
        gain = (luma / ref_luma) if ref_luma > 1e-3 else 1.0
        detail = 1.0 + 0.45 * (gain - 1.0)
        mix = 0.90 * edge
        for c in range(3):
            target = ref[c] * detail * 1.06
            texels[o + c] = int(max(0, min(255, texels[o + c] * (1.0 - mix) + target * mix)))
        touched += 1
    return touched


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--no-shave", action="store_true",
                    help="keep the scan's stubble instead of digitally shaving it")
    ap.add_argument("--no-fit", action="store_true",
                    help="keep the scan's own scale and origin (debugging only)")
    args = ap.parse_args(argv[1:])

    js, binary = read_glb(args.source)
    prim = js["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    positions = accessor_bytes(js, binary, attrs["POSITION"])
    normals = accessor_bytes(js, binary, attrs["NORMAL"])
    uvs = accessor_bytes(js, binary, attrs["TEXCOORD_0"])
    indices = accessor_bytes(js, binary, prim["indices"])
    vertex_count = js["accessors"][attrs["POSITION"]]["count"]
    index_count = js["accessors"][prim["indices"]]["count"]

    if not args.no_fit:
        positions, info = fit_to_silk_space(positions, vertex_count)
        print("fit: scale %.3f  head height %.3f m -> %.3f m  (crown y %.3f, chin y %.3f)" % (
            info["scale"], info["height"], info["new_height"], CROWN_Y,
            CROWN_Y - info["new_height"]))
        print("     neck cut y=%.3f (scan units), skull width %.3f m, z centroid %.3f" % (
            info["neck_cut"], info["skull_width"] * info["scale"], info["z_centroid"]))

    material = js["materials"][prim.get("material", 0)]
    tex_index = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    image_index = js["textures"][tex_index]["source"]
    image = js["images"][image_index]
    bv = js["bufferViews"][image["bufferView"]]
    image_bytes = binary[bv.get("byteOffset", 0) : bv.get("byteOffset", 0) + bv["byteLength"]]

    width, height, channels, rows = read_png(image_bytes)
    factor = max(1, width // args.texture_size)
    tw, th, texels = downsample(width, height, channels, rows, factor)
    texels = bytearray(texels)
    if not args.no_shave:
        space = texture_skin_space(positions, uvs, indices, vertex_count, index_count, tw, th,
                                   None, 1.0)
        shaved = digital_shave(texels, space, tw, th)
        print(f"digital shave: {shaved} texels of stubble re-tinted to the cheek complexion")
    png = write_png(tw, th, texels)
    print(f"diffuse {width}x{height} -> {tw}x{th} ({len(png)/1e6:.2f} MB PNG)")

    # ---- repack -------------------------------------------------------------
    blobs: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []

    def add_view(payload: bytes, target: int | None = None) -> int:
        while len(blobs) and (sum(len(b) for b in blobs) % 4):
            blobs.append(b"\x00")
        views.append({"buffer": 0, "byteOffset": sum(len(b) for b in blobs),
                      "byteLength": len(payload), **({"target": target} if target else {})})
        blobs.append(payload)
        return len(views) - 1

    def add_accessor(payload: bytes, count: int, ctype: int, atype: str, target: int,
                     minmax: bool = False, kind: str = "") -> int:
        view = add_view(payload, target)
        acc = {"bufferView": view, "componentType": ctype, "count": count, "type": atype}
        if minmax:
            comps = len(payload) // (4 * count)
            values = struct.unpack(f"<{count * comps}f", payload)
            acc["min"] = [min(values[c::comps]) for c in range(comps)]
            acc["max"] = [max(values[c::comps]) for c in range(comps)]
        accessors.append(acc)
        return len(accessors) - 1

    a_pos = add_accessor(positions, vertex_count, 5126, "VEC3", 34962, True)
    a_nrm = add_accessor(normals, vertex_count, 5126, "VEC3", 34962)
    a_uv = add_accessor(uvs, vertex_count, 5126, "VEC2", 34962)
    a_idx = add_accessor(indices, index_count, 5123, "SCALAR", 34963)
    tex_view = add_view(png)

    total = sum(len(b) for b in blobs)
    out_js = {
        "asset": {"version": "2.0", "generator": "SilkCrowd tools/prep_head.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "head_scan"}],
        "meshes": [{"name": "head_scan", "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv},
            "indices": a_idx, "material": 0}]}],
        "materials": [{
            "name": "skin_scan",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.72},
            "doubleSided": False}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": tex_view, "mimeType": "image/png"}],
        "buffers": [{"byteLength": total}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js_bytes = json.dumps(out_js, separators=(",", ":")).encode("utf-8")
    while len(js_bytes) % 4:
        js_bytes += b" "
    bin_bytes = b"".join(blobs)
    while len(bin_bytes) % 4:
        bin_bytes += b"\x00"

    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js_bytes) + 8 + len(bin_bytes))
    out += struct.pack("<II", len(js_bytes), 0x4E4F534A) + js_bytes
    out += struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(bytes(out))
    print(f"wrote {args.out} ({len(out)/1e6:.2f} MB): {vertex_count} verts, {index_count//3} tris")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
