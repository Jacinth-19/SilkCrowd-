#!/usr/bin/env python3
"""Builds `assets/heads/female_head_scan.glb` from a MakeHuman (MPFB) female avatar.

    tools/prep_female.py <mpfb.glb> <out.glb> [--texture-size 1024]

Source: `met4citizen/TalkingHead` `avatars/mpfb.glb`, a MakeHuman 2.0 female exported with
MPFB. MakeHuman's mesh and its bundled textures are CC0, so - unlike the male
Infinite-Realities scan - this head can be redistributed without an attribution obligation
(credit is given anyway in assets/heads/LICENSE-female_head.txt).

What this tool does, in order:

 1. takes the body mesh (`base`) and cuts a bust out of it: everything above the neck,
    with the arms, hands, clothes and the ponytail mesh dropped;
 2. fits the bust into SilkCrowd's head space with the same landmark method as
    `prep_head.py` (crown, menton, narrowest neck slab, widest skull slab), uniform scale;
 3. **repacks the texture**: a MakeHuman body texture is a 2048^2 atlas of a whole body, in
    which the face owns perhaps 20 % of the area. The UVs of the bust are therefore
    re-mapped into a tight square and that region is cropped out of the source at full
    resolution -- so the face gets a 1024^2 texture of its own instead of a 300^2 patch;
 4. closes the bottom of the bust with a cap, so the neck does not read as a hole, and
 5. writes a self-contained GLB with POSITION/NORMAL/TEXCOORD_0 plus the base-colour map,
    which is exactly what `silk::loadGlb()` accepts.

Only zlib/struct are used (the PNG codec is shared with `tools/prep_head.py`).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_head import (CROWN_Y, SKULL_WIDTH, accessor_bytes, downsample, read_glb,
                       read_png, write_png)


def read_mesh(js, binary, name):
    """Returns (positions, normals, uvs, indices, material) for a glTF mesh by name."""
    mesh = next(m for m in js["meshes"] if m.get("name") == name)
    prim = mesh["primitives"][0]
    attrs = prim["attributes"]

    def floats(accessor_index, components):
        count = js["accessors"][accessor_index]["count"]
        raw = accessor_bytes(js, binary, accessor_index)
        values = struct.unpack(f"<{count * components}f", raw)
        return [values[i * components:(i + 1) * components] for i in range(count)]

    positions = floats(attrs["POSITION"], 3)
    normals = floats(attrs["NORMAL"], 3)
    uvs = floats(attrs["TEXCOORD_0"], 2)
    indices = list(struct.unpack(f"<{js['accessors'][prim['indices']]['count']}H",
                                 accessor_bytes(js, binary, prim["indices"])))
    return positions, normals, uvs, indices, js["materials"][prim["material"]]["name"]


def cut_bust(positions, normals, uvs, indices, y_lo, y_hi, half_width):
    """Keeps the triangles that lie inside the bust box; returns the compacted mesh."""
    keep = [False] * len(positions)
    for i, p in enumerate(positions):
        keep[i] = (y_lo <= p[1] <= y_hi) and abs(p[0]) <= half_width
    tris = []
    for t in range(0, len(indices), 3):
        i0, i1, i2 = indices[t], indices[t + 1], indices[t + 2]
        if keep[i0] and keep[i1] and keep[i2]:
            tris.append((i0, i1, i2))
    remap = {}
    out_pos, out_nrm, out_uv, out_idx = [], [], [], []
    for tri in tris:
        for i in tri:
            if i not in remap:
                remap[i] = len(out_pos)
                out_pos.append(positions[i])
                out_nrm.append(normals[i])
                out_uv.append(uvs[i])
            out_idx.append(remap[i])
    return out_pos, out_nrm, out_uv, out_idx


def cap_bottom(positions, normals, uvs, indices, y_cut, tolerance):
    """Closes the hole left by the cut with a fan, so the neck is not a see-through tube."""
    ring = [i for i, p in enumerate(positions) if abs(p[1] - y_cut) < tolerance]
    if len(ring) < 5:
        return
    cx = sum(positions[i][0] for i in ring) / len(ring)
    cz = sum(positions[i][2] for i in ring) / len(ring)
    ring.sort(key=lambda i: math.atan2(positions[i][2] - cz, positions[i][0] - cx))
    centre = len(positions)
    positions.append([cx, y_cut - 0.004, cz])
    normals.append([0.0, -1.0, 0.0])
    uvs.append([0.5, 0.02])
    for k in range(len(ring)):
        a = ring[k]
        b = ring[(k + 1) % len(ring)]
        indices.extend([centre, b, a])
    # The cap's own vertices are left with the skin normal above; they are only ever seen
    # from below, where the shading is irrelevant.


def fit(positions, label):
    """Uniform scale + translation into SilkCrowd's head space (same method as prep_head)."""
    ys = [p[1] for p in positions]
    y_lo, y_hi = min(ys), max(ys)
    slab = 0.004
    n_slabs = max(8, int((y_hi - y_lo) / slab) + 1)
    widths = []
    for s in range(n_slabs):
        y = y_lo + (s + 0.5) * slab
        band = [abs(p[0]) for p in positions if abs(p[1] - y) < slab * 0.6]
        widths.append(2.0 * max(band) if len(band) > 8 else 0.0)
    peak = max(widths)
    neck_slab = 0
    for s in range(1, n_slabs - 1):
        if 0.0 < widths[s] < 0.62 * peak and widths[s] <= widths[s - 1] and widths[s] <= widths[s + 1]:
            neck_slab = s
            break
    neck_cut = y_lo + neck_slab * slab
    head = [p for p in positions if p[1] > neck_cut + 0.01]
    crown = max(p[1] for p in head)
    mid_x = 0.5 * (max(p[0] for p in head) + min(p[0] for p in head))
    z_mid = 0.5 * (min(p[2] for p in head) + max(p[2] for p in head))
    midline = [p for p in head if abs(p[0] - mid_x) < 0.012 and p[2] > z_mid]
    chin = min(p[1] for p in midline) if midline else neck_cut
    height = crown - chin
    top = [p for p in head if p[1] > crown - 0.55 * height]
    x99 = sorted(abs(p[0]) for p in top)[int(0.99 * (len(top) - 1))]
    skull = 2.0 * x99
    upper = [p for p in head if p[1] > crown - 0.22 * height]
    z_c = sum(p[2] for p in upper) / len(upper)
    s = SKULL_WIDTH / skull
    print(f"{label}: crown {crown:.3f} chin {chin:.3f} height {height:.3f} m "
          f"skull {skull:.3f} m -> scale {s:.3f}, new height {height * s:.3f} m")
    return [( (p[0] - mid_x) * s, (p[1] - crown) * s + CROWN_Y, (p[2] - z_c) * s )
            for p in positions], s, crown, mid_x, z_c


def image_of(js, binary, material_name):
    """Decodes a material's base-colour image to (width, height, channels, rows)."""
    mat = next(m for m in js["materials"] if m["name"] == material_name)
    index = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    image = js["images"][js["textures"][index]["source"]]
    view = js["bufferViews"][image["bufferView"]]
    blob = binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
    return read_png(blob)


def uv_occupancy(uvs, indices, size):
    """Rasterises the UV triangles into a size x size grid of covered texels."""
    grid = [[False] * size for _ in range(size)]
    for t in range(0, len(indices), 3):
        u = [uvs[indices[t + k]][0] * size for k in range(3)]
        v = [uvs[indices[t + k]][1] * size for k in range(3)]
        area = (u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0])
        if abs(area) < 1e-9:
            continue
        inv = 1.0 / area
        x0 = max(0, int(min(u))); x1 = min(size - 1, int(max(u)) + 1)
        y0 = max(0, int(min(v))); y1 = min(size - 1, int(max(v)) + 1)
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                sx, sy = px + 0.5, py + 0.5
                b0 = ((u[1] - sx) * (v[2] - sy) - (u[2] - sx) * (v[1] - sy)) * inv
                b1 = ((u[2] - sx) * (v[0] - sy) - (u[0] - sx) * (v[2] - sy)) * inv
                if b0 < 0 or b1 < 0 or b0 + b1 > 1:
                    continue
                grid[py][px] = True
    return grid


def smooth(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a) if b > a else 0.0))
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))


def shells(positions, indices):
    """Connected components of the mesh, as lists of vertex indices."""
    parent = list(range(len(positions)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in range(0, len(indices), 3):
        a, b, c = indices[t], indices[t + 1], indices[t + 2]
        ra, rb, rc = find(a), find(b), find(c)
        parent[ra] = rb
        parent[find(rb)] = find(rc)
    groups = {}
    for i in range(len(positions)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def find_eyeballs(positions, indices):
    """The two eyeball shells of a MakeHuman base mesh, as lists of vertex indices.

    MakeHuman models the eyeballs as geometry: two closed, roughly spherical components of
    308 vertices each, sitting in the sockets. They are picked by shape rather than by name
    or by colour, so a different MakeHuman build with a different vertex count still works.
    """
    balls = []
    for comp in shells(positions, indices):
        if not 150 <= len(comp) <= 600:
            continue
        extents = [max(positions[i][k] for i in comp) - min(positions[i][k] for i in comp)
                   for k in range(3)]
        if max(extents) > 0.060 or max(extents) < 0.008:
            continue
        balls.append(comp)
    return balls


def drop_triangles(positions, normals, uvs, indices, doomed):
    """Removes every triangle touching `doomed` and compacts the vertex arrays."""
    keep = [not any(v in doomed for v in indices[t:t + 3]) for t in range(0, len(indices), 3)]
    used = sorted({v for t, k in enumerate(keep) if k for v in indices[t * 3:t * 3 + 3]})
    remap = {v: i for i, v in enumerate(used)}
    out_i = [remap[v] for t, k in enumerate(keep) if k for v in indices[t * 3:t * 3 + 3]]
    return ([positions[i] for i in used], [normals[i] for i in used],
            [uvs[i] for i in used], out_i)


def paint_eyeball(texels, width, height, block):
    """Paints one eyeball into its atlas block, in block-normalised coordinates.

    The eyeball mesh added by `add_eyeballs` maps the globe stereographically about the
    cornea onto the block, so (a, b) in [0, 1]^2 *is* the eyeball seen from the front: the
    iris is a disc at the centre and the sclera is everything out to the rim.
    """
    u0, u1, v0, v1 = block
    x0, x1 = int(u0 * width), int((u1) * width) + 1
    y0, y1 = int(v0 * height), int((v1) * height) + 1
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rx, ry = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    for py in range(max(0, y0), min(height, y1)):
        for px in range(max(0, x0), min(width, x1)):
            a = (px - cx) / rx
            b = (py - cy) / ry
            r = math.sqrt(a * a + b * b)
            # A 12 mm iris on a 24 mm globe is half its radius, seen from the front it
            # subtends 30 degrees; with a stereographic map of the front hemisphere that is
            # tan(30/2) = 0.27 of the rim, and the pupil (2 mm) is 0.09.
            if r < 0.09:
                col = mix((10.0, 8.0, 7.0), (32.0, 18.0, 11.0),
                          smooth(0.05, 0.09, r))
            elif r < 0.27:
                t = (r - 0.09) / 0.18
                ang = math.atan2(b, a)
                fibre = 0.88 + 0.12 * math.sin(ang * 61.0) * math.sin(ang * 27.0 + 1.3)
                limb = 1.0 - 0.42 * smooth(0.245, 0.27, r)
                k = fibre * limb
                col = (132.0 * k * (0.9 + 0.1 * t), 96.0 * k * (0.82 + 0.18 * t),
                       62.0 * k * (0.78 + 0.22 * t))
            else:
                t = smooth(0.27, 1.0, r)
                ang = math.atan2(b, a)
                vein = max(0.0, math.sin(ang * 9.0 + 0.7)) ** 4 * t
                col = (237.0 - 26.0 * t + 16.0 * vein,
                       232.0 - 44.0 * t + 5.0 * vein,
                       227.0 - 58.0 * t + 3.0 * vein)
            o = (py * width + px) * 3
            for k in range(3):
                texels[o + k] = max(0, min(255, int(col[k])))


def add_eyeballs(positions, normals, uvs, indices, spheres):
    """Replaces MakeHuman's eyeball shells with globes whose UVs this tool controls.

    MakeHuman's own eyeball spheres are unusable as they are: seen through the lid aperture
    they render inside out (a probe render showed the visible patch mapping to the *far*
    hemisphere of their UV island, i.e. the back of the globe), and their unwrap is a
    full-sphere layout whose centre is nowhere near the cornea. So the shells are dropped
    and rebuilt here with a stereographic front projection into the free atlas block their
    islands leave behind.
    """
    add_pos, add_nrm, add_uv, add_idx = [], [], [], []
    base_len = len(positions)
    for (centre, radius, block) in spheres:
        side = 1.0 if centre[0] >= 0.0 else -1.0
        # A resting eye looks along the head's forward axis with a few degrees of outward
        # divergence; more than that and the iris slides behind the inner lid.
        norm = math.sqrt(1.0 + 0.12 * 0.12 + 0.02 * 0.02)
        fwd = (0.12 * side / norm, 0.02 / norm, 1.0 / norm)
        right = (fwd[2], 0.0, -fwd[0])
        up = (0.0, 1.0, 0.0)
        u0, u1, v0, v1 = block
        nu, nv = 28, 20
        first = base_len + len(add_pos)
        for j in range(nv + 1):
            phi = math.pi * j / nv                    # 0 = the back pole
            for i in range(nu + 1):
                th = 2.0 * math.pi * i / nu
                d = (math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th))
                p = [centre[k] + d[k] * radius for k in range(3)]
                add_pos.append(p)
                add_nrm.append(d)
                df = sum(d[k] * fwd[k] for k in range(3))
                a = sum(d[k] * right[k] for k in range(3))
                b = -sum(d[k] * up[k] for k in range(3))
                # Stereographic about the cornea; the back hemisphere is folded onto the rim
                # so it can never overlap the visible front.
                if df <= -0.2:
                    length = math.sqrt(a * a + b * b) or 1.0
                    a, b = a / length, b / length
                else:
                    a, b = a / (1.0 + df), b / (1.0 + df)
                au = u0 + 0.5 * (1.0 + a) * (u1 - u0)
                av = v0 + 0.5 * (1.0 + b) * (v1 - v0)
                add_uv.append([min(u1, max(u0, au)), min(v1, max(v0, av))])
        for j in range(nv):
            for i in range(nu):
                a = first + j * (nu + 1) + i
                b = a + 1
                c = a + nu + 1
                d = c + 1
                add_idx.extend([a, c, b, b, c, d])
    return (positions + add_pos, normals + add_nrm, uvs + add_uv, indices + add_idx)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--texture-size", type=int, default=0,
                    help="unused (the atlas is kept at its authored resolution)")
    ap.add_argument("--neck-cut", type=float, default=1.452,
                    help="world y of the horizontal cut (source is metres, y up)")
    args = ap.parse_args(argv[1:])

    js, binary = read_glb(args.source)
    positions, normals, uvs, indices, material = read_mesh(js, binary, "base")
    print(f"source: {len(positions)} verts, {len(indices) // 3} triangles")
    ys = [p[1] for p in positions]
    print(f"  body y range {min(ys):.3f} .. {max(ys):.3f}")

    positions, normals, uvs, indices = cut_bust(positions, normals, uvs, indices,
                                                args.neck_cut, 10.0, 0.20)
    print(f"  bust: {len(positions)} verts, {len(indices) // 3} triangles")
    positions, scale, crown, mid_x, z_c = fit(positions, "fit")
    # The landmark fit recalculates the frame; express the cut plane in head space too.
    y_cut = (args.neck_cut - crown) * scale + CROWN_Y
    cap_bottom(positions, normals, uvs, indices, y_cut, 0.004)

    width, height, channels, rows = image_of(js, binary, "Human.body")
    print(f"  skin atlas: {width}x{height}")

    # The body's own UV layout is kept exactly as MakeHuman authored it -- the lips, the
    # brows and the lashes are painted where they belong, and re-mapping the atlas is what
    # put the mouth on the cheek in an earlier revision of this tool.
    texels = bytearray(width * height * 3)
    for y in range(height):
        line = rows[y]
        for x in range(width):
            o = x * channels
            d = (y * width + x) * 3
            texels[d] = line[o]
            texels[d + 1] = line[o + 1]
            texels[d + 2] = line[o + 2]

    # ---- eyes ---------------------------------------------------------------
    # MakeHuman's own globes are dropped and rebuilt (see `add_eyeballs`), and their atlas
    # islands -- the flat-red placeholder squares in the body texture -- become the free
    # blocks the new globes are painted into. The rest of the atlas keeps MakeHuman's
    # authored UVs exactly: re-mapping it is what once put the mouth on the cheek.
    balls = find_eyeballs(positions, indices)
    spheres = []
    for ball in balls:
        centre = [sum(positions[i][k] for i in ball) / len(ball) for k in range(3)]
        extents = [max(positions[i][k] for i in ball) - min(positions[i][k] for i in ball)
                   for k in range(3)]
        block = (min(uvs[i][0] for i in ball), max(uvs[i][0] for i in ball),
                 min(uvs[i][1] for i in ball), max(uvs[i][1] for i in ball))
        spheres.append((centre, 0.5 * sum(extents) / 3.0, block))
    doomed = {i for ball in balls for i in ball}
    positions, normals, uvs, indices = drop_triangles(positions, normals, uvs, indices, doomed)
    positions, normals, uvs, indices = add_eyeballs(positions, normals, uvs, indices, spheres)
    for _, _, block in spheres:
        paint_eyeball(texels, width, height, block)
    print(f"  eyes: {len(balls)} MakeHuman eyeball shells rebuilt as front-projected globes "
          f"({len(indices) // 3} triangles total)")
    png = write_png(width, height, bytes(texels))
    # ---- repack -------------------------------------------------------------
    blobs, views, accessors = [], [], []

    def add_view(payload, target=None):
        while blobs and (sum(len(b) for b in blobs) % 4):
            blobs.append(b"\x00")
        views.append({"buffer": 0, "byteOffset": sum(len(b) for b in blobs),
                      "byteLength": len(payload), **({"target": target} if target else {})})
        blobs.append(payload)
        return len(views) - 1

    def add_accessor(payload, count, ctype, atype, target, minmax=False):
        views.append({"buffer": 0, "byteOffset": sum(len(b) for b in blobs),
                      "byteLength": len(payload), "target": target})
        blobs.append(payload)
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": count, "type": atype}
        if minmax:
            comps = len(payload) // (4 * count)
            values = struct.unpack(f"<{count * comps}f", payload)
            acc["min"] = [min(values[c::comps]) for c in range(comps)]
            acc["max"] = [max(values[c::comps]) for c in range(comps)]
        accessors.append(acc)
        return len(accessors) - 1

    pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_bytes = b"".join(struct.pack("<3f", *n) for n in normals)
    uv_bytes = b"".join(struct.pack("<2f", *t) for t in uvs)
    idx_bytes = struct.pack(f"<{len(indices)}I", *indices)
    a_pos = add_accessor(pos_bytes, len(positions), 5126, "VEC3", 34962, True)
    a_nrm = add_accessor(nrm_bytes, len(normals), 5126, "VEC3", 34962)
    a_uv = add_accessor(uv_bytes, len(uvs), 5126, "VEC2", 34962)
    a_idx = add_accessor(idx_bytes, len(indices), 5125, "SCALAR", 34963)
    tex_view = add_view(png)

    total = sum(len(b) for b in blobs)
    out_js = {
        "asset": {"version": "2.0",
                  "generator": "SilkCrowd tools/prep_female.py (MakeHuman/MPFB, CC0)",
                  "copyright": "MakeHuman/MPFB mesh and textures, CC0"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "head_female"}],
        "meshes": [{"name": "head_female", "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv},
            "indices": a_idx, "material": 0}]}],
        "materials": [{"name": "skin_female", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
            "roughnessFactor": 0.55}, "doubleSided": False}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": tex_view, "mimeType": "image/png"}],
        "buffers": [{"byteLength": total}], "bufferViews": views, "accessors": accessors,
    }
    js_bytes = json.dumps(out_js, separators=(",", ":")).encode()
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
    print(f"wrote {args.out} ({len(out)/1e6:.2f} MB): {len(positions)} verts, "
          f"{len(indices)//3} tris, {width}x{height} skin+eye atlas")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
