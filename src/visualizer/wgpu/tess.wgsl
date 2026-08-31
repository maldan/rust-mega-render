// Packed GpuVertex is 22 u32s (88 bytes) — matches CPU `repr(C)` vertex layout,
// which WGSL vec3 storage padding cannot express.

const VERT_U32: u32 = 22u;
const OFF_POS: u32 = 0u;
const OFF_NRM: u32 = 3u;
const OFF_UV: u32 = 6u;
const OFF_TAN: u32 = 8u;
const OFF_JOINTS: u32 = 12u;
const OFF_WEIGHTS: u32 = 14u;
const OFF_COLOR: u32 = 18u;

struct TessParams {
    tri_count: u32,
    scale: f32,
    lod_near: f32,
    lod_far: f32,
    camera_pos: vec3<f32>,
    tess_factor: u32,
    model: mat4x4<f32>,
}

const MAX_TESS: u32 = 32u;

fn tess_cap() -> u32 {
    return clamp(params.tess_factor, 1u, MAX_TESS);
}

@group(0) @binding(0) var<uniform> params: TessParams;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var height_samp: sampler;
@group(0) @binding(3) var<storage, read> src_verts: array<u32>;
@group(0) @binding(4) var<storage, read> src_idx: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_verts: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst_idx: array<u32>;

fn verts_per_tri(t: u32) -> u32 {
    return (t + 1u) * (t + 2u) / 2u;
}

fn idx_per_tri(t: u32) -> u32 {
    return t * t * 3u;
}

fn vert_ij(t: u32, i: u32, j: u32) -> u32 {
    return i * (t + 1u) - i * (i - 1u) / 2u + j;
}

fn load_f32(base: u32, off: u32) -> f32 {
    return bitcast<f32>(src_verts[base + off]);
}

fn store_f32(base: u32, off: u32, v: f32) {
    dst_verts[base + off] = bitcast<u32>(v);
}

fn load_vec3(base: u32, off: u32) -> vec3<f32> {
    return vec3<f32>(load_f32(base, off), load_f32(base, off + 1u), load_f32(base, off + 2u));
}

fn store_vec3(base: u32, off: u32, v: vec3<f32>) {
    store_f32(base, off, v.x);
    store_f32(base, off + 1u, v.y);
    store_f32(base, off + 2u, v.z);
}

fn load_vec2(base: u32, off: u32) -> vec2<f32> {
    return vec2<f32>(load_f32(base, off), load_f32(base, off + 1u));
}

fn store_vec2(base: u32, off: u32, v: vec2<f32>) {
    store_f32(base, off, v.x);
    store_f32(base, off + 1u, v.y);
}

fn load_vec4(base: u32, off: u32) -> vec4<f32> {
    return vec4<f32>(
        load_f32(base, off),
        load_f32(base, off + 1u),
        load_f32(base, off + 2u),
        load_f32(base, off + 3u),
    );
}

fn store_vec4(base: u32, off: u32, v: vec4<f32>) {
    store_f32(base, off, v.x);
    store_f32(base, off + 1u, v.y);
    store_f32(base, off + 2u, v.z);
    store_f32(base, off + 3u, v.w);
}

fn sample_h(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(height_tex, height_samp, uv, 0.0).r;
}

struct Ctrl {
    pos: vec3<f32>,
    nrm: vec3<f32>,
    uv: vec2<f32>,
    tan: vec4<f32>,
    j0: u32,
    j1: u32,
    weights: vec4<f32>,
    color: vec4<f32>,
}

fn load_ctrl(vid: u32) -> Ctrl {
    let b = vid * VERT_U32;
    return Ctrl(
        load_vec3(b, OFF_POS),
        load_vec3(b, OFF_NRM),
        load_vec2(b, OFF_UV),
        load_vec4(b, OFF_TAN),
        src_verts[b + OFF_JOINTS],
        src_verts[b + OFF_JOINTS + 1u],
        load_vec4(b, OFF_WEIGHTS),
        load_vec4(b, OFF_COLOR),
    );
}

fn mix3v2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, w: vec3<f32>) -> vec2<f32> {
    return a * w.x + b * w.y + c * w.z;
}

fn mix3v3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, w: vec3<f32>) -> vec3<f32> {
    return a * w.x + b * w.y + c * w.z;
}

fn mix3v4(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, w: vec3<f32>) -> vec4<f32> {
    return a * w.x + b * w.y + c * w.z;
}

fn world_pos(local: vec3<f32>) -> vec3<f32> {
    return (params.model * vec4<f32>(local, 1.0)).xyz;
}

fn tess_from_dist(world: vec3<f32>) -> u32 {
    let dist = length(world - params.camera_pos);
    let span = max(params.lod_far - params.lod_near, 1e-3);
    let t = clamp((dist - params.lod_near) / span, 0.0, 1.0);
    let lod = u32(round(4.0 * (1.0 - t)));
    var level: u32;
    switch lod {
        case 0u: { level = 1u; }
        case 1u: { level = 2u; }
        case 2u: { level = 4u; }
        case 3u: { level = 8u; }
        default: { level = MAX_TESS; }
    }
    let cap = tess_cap();
    return min(level, cap);
}

fn snap_edge(s: u32, t: u32, e: u32) -> f32 {
    let ee = max(e, 1u);
    let tt = max(t, 1u);
    let q = (s * ee + tt / 2u) / tt;
    return f32(q) / f32(ee);
}

fn mix_ctrl(a: Ctrl, b: Ctrl, u: f32) -> Ctrl {
    return Ctrl(
        mix(a.pos, b.pos, u),
        mix(a.nrm, b.nrm, u),
        mix(a.uv, b.uv, u),
        mix(a.tan, b.tan, u),
        select(b.j0, a.j0, u < 0.5),
        select(b.j1, a.j1, u < 0.5),
        mix(a.weights, b.weights, u),
        mix(a.color, b.color, u),
    );
}

fn bary_ctrl(a: Ctrl, b: Ctrl, c: Ctrl, w: vec3<f32>) -> Ctrl {
    return Ctrl(
        mix3v3(a.pos, b.pos, c.pos, w),
        mix3v3(a.nrm, b.nrm, c.nrm, w),
        mix3v2(a.uv, b.uv, c.uv, w),
        mix3v4(a.tan, b.tan, c.tan, w),
        select(select(c.j0, b.j0, w.y >= w.x && w.y >= w.z), a.j0, w.x >= w.y && w.x >= w.z),
        select(select(c.j1, b.j1, w.y >= w.x && w.y >= w.z), a.j1, w.x >= w.y && w.x >= w.z),
        mix3v4(a.weights, b.weights, c.weights, w),
        mix3v4(a.color, b.color, c.color, w),
    );
}

fn write_vert(dst: u32, src: Ctrl) {
    let nrm_u = displaced_normal(src.pos, src.nrm, src.uv, src.tan);
    let pos_u = displace(src.pos, src.nrm, src.uv, src.tan);
    store_vec3(dst, OFF_POS, pos_u);
    store_vec3(dst, OFF_NRM, nrm_u);
    store_vec2(dst, OFF_UV, src.uv);
    store_vec4(dst, OFF_TAN, src.tan);
    dst_verts[dst + OFF_JOINTS] = src.j0;
    dst_verts[dst + OFF_JOINTS + 1u] = src.j1;
    store_vec4(dst, OFF_WEIGHTS, src.weights);
    store_vec4(dst, OFF_COLOR, src.color);
}

fn displace(pos: vec3<f32>, nrm: vec3<f32>, uv: vec2<f32>, tan: vec4<f32>) -> vec3<f32> {
    let n = normalize(nrm);
    let h = sample_h(uv);
    return pos + n * (h * params.scale);
}

fn displaced_normal(pos: vec3<f32>, nrm: vec3<f32>, uv: vec2<f32>, tan: vec4<f32>) -> vec3<f32> {
    let n = normalize(nrm);
    if abs(params.scale) < 1e-8 {
        return n;
    }
    var t = tan.xyz;
    let tl = length(t);
    if tl < 1e-5 {
        return n;
    }
    t = t / tl;
    // Same TBN as mesh.wgsl / glTF. Do not light with T×B: mirrored UVs (tan.w
    // = -1) make that product −N, which reads as a view-locked dark blob.
    let b = normalize(cross(n, t)) * tan.w;
    let sz = vec2<f32>(textureDimensions(height_tex, 0));
    let e = vec2<f32>(1.0) / max(sz, vec2<f32>(1.0));
    let p0 = displace(pos, n, uv, tan);
    let p1w = displace(pos + t * 0.02, n, uv + vec2<f32>(e.x, 0.0), tan);
    let p2w = displace(pos + b * 0.02, n, uv + vec2<f32>(0.0, e.y), tan);
    var nn = cross(p1w - p0, p2w - p0);
    let ln = length(nn);
    if ln < 1e-8 {
        return n;
    }
    nn = nn / ln;
    if dot(nn, n) < 0.0 {
        nn = -nn;
    }
    return nn;
}

@compute @workgroup_size(64)
fn tess_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri = gid.x;
    if tri >= params.tri_count {
        return;
    }
    let i0 = src_idx[tri * 3u];
    let i1 = src_idx[tri * 3u + 1u];
    let i2 = src_idx[tri * 3u + 2u];
    let a = load_ctrl(i0);
    let b = load_ctrl(i1);
    let c = load_ctrl(i2);

    let te_ab = tess_from_dist(world_pos((a.pos + b.pos) * 0.5));
    let te_bc = tess_from_dist(world_pos((b.pos + c.pos) * 0.5));
    let te_ca = tess_from_dist(world_pos((c.pos + a.pos) * 0.5));
    let t_in = tess_from_dist(world_pos((a.pos + b.pos + c.pos) / 3.0));
    let t = max(max(max(te_ab, te_bc), te_ca), max(t_in, 1u));
    let cap = tess_cap();

    let vbase = tri * verts_per_tri(cap);
    let ibase = tri * idx_per_tri(cap);

    for (var i = 0u; i <= t; i++) {
        for (var j = 0u; j <= (t - i); j++) {
            let k = t - i - j;
            var src = a;
            if i == 0u && j == 0u {
                src = a;
            } else if j == 0u && k == 0u {
                src = b;
            } else if i == 0u && k == 0u {
                src = c;
            } else if j == 0u {
                src = mix_ctrl(a, b, snap_edge(i, t, te_ab));
            } else if k == 0u {
                src = mix_ctrl(b, c, snap_edge(j, t, te_bc));
            } else if i == 0u {
                src = mix_ctrl(a, c, snap_edge(j, t, te_ca));
            } else {
                let w = vec3<f32>(f32(k), f32(i), f32(j)) / f32(t);
                src = bary_ctrl(a, b, c, w);
            }
            write_vert((vbase + vert_ij(t, i, j)) * VERT_U32, src);
        }
    }

    var wtri = 0u;
    for (var i = 0u; i < t; i++) {
        for (var j = 0u; j < (t - i); j++) {
            let v00 = vbase + vert_ij(t, i, j);
            let v10 = vbase + vert_ij(t, i + 1u, j);
            let v01 = vbase + vert_ij(t, i, j + 1u);
            let o = ibase + wtri * 3u;
            dst_idx[o] = v00;
            dst_idx[o + 1u] = v10;
            dst_idx[o + 2u] = v01;
            wtri++;
            if i + j + 1u < t {
                let v11 = vbase + vert_ij(t, i + 1u, j + 1u);
                let o2 = ibase + wtri * 3u;
                dst_idx[o2] = v10;
                dst_idx[o2 + 1u] = v11;
                dst_idx[o2 + 2u] = v01;
                wtri++;
            }
        }
    }
    let pad_n = idx_per_tri(cap);
    let degener = vbase;
    for (var p = wtri * 3u; p < pad_n; p++) {
        dst_idx[ibase + p] = degener;
    }
}
