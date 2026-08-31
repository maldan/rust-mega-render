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
    // Target on-screen edge length in pixels (see crate::TessSettings).
    target_px: f32,
    _reserved: f32,
    camera_pos: vec3<f32>,
    tess_factor: u32,
    model: mat4x4<f32>,
    // World-space frustum planes (left, right, bottom, top, near, far),
    // each normalized so dot(xyz, p) + w is a true signed distance.
    planes: array<vec4<f32>, 6>,
    // Combined view_proj * model, for projecting triangle edges to screen
    // space to drive the adaptive tessellation level.
    mvp: mat4x4<f32>,
    // (viewport_width_px, viewport_height_px, unused, unused).
    viewport: vec4<f32>,
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

fn world_dir(local: vec3<f32>) -> vec3<f32> {
    return (params.model * vec4<f32>(local, 0.0)).xyz;
}

// Signed distance from `p` to a normalized plane (dot(xyz, p) + w).
fn plane_dist(p: vec4<f32>, pos: vec3<f32>) -> f32 {
    return dot(p.xyz, pos) + p.w;
}

// True if the bounding sphere (center, radius) lies fully outside any
// single frustum plane, i.e. is guaranteed not to touch the screen.
fn frustum_reject(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        if plane_dist(params.planes[i], center) < -radius {
            return true;
        }
    }
    return false;
}

// Small negative bias so we don't cull triangles right at the silhouette
// edge, where interpolated vertex normals can be slightly off from the
// true geometric facing direction.
const BACKFACE_BIAS: f32 = 0.10;

// True if the triangle's outward-facing normal points away from the
// camera by more than BACKFACE_BIAS, i.e. it would be backface-culled by
// the rasterizer regardless of how much detail we generate for it.
fn is_backface(a_nrm: vec3<f32>, b_nrm: vec3<f32>, c_nrm: vec3<f32>, center: vec3<f32>) -> bool {
    let n = world_dir(a_nrm + b_nrm + c_nrm);
    let nlen = length(n);
    if nlen < 1e-6 {
        return false;
    }
    let to_cam = params.camera_pos - center;
    let vlen = length(to_cam);
    if vlen < 1e-6 {
        return false;
    }
    let facing = dot(n / nlen, to_cam / vlen);
    return facing < -BACKFACE_BIAS;
}

// Converts a clip-space position to screen-space pixel coordinates.
fn clip_to_px(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / max(clip.w, 1e-5);
    return (ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5)) * params.viewport.xy;
}

// On-screen length, in pixels, of the edge between two local-space points.
// Points behind the camera don't have a meaningful screen-space length; in
// that case we return a huge value so the caller falls back to the
// tessellation cap instead of under-tessellating near the eye.
fn edge_px(a_local: vec3<f32>, b_local: vec3<f32>) -> f32 {
    let ca = params.mvp * vec4<f32>(a_local, 1.0);
    let cb = params.mvp * vec4<f32>(b_local, 1.0);
    if ca.w <= 1e-4 || cb.w <= 1e-4 {
        return 1e9;
    }
    return distance(clip_to_px(ca), clip_to_px(cb));
}

// Subdivisions needed so a `px`-pixel-long edge splits into segments no
// longer than params.target_px, capped by the per-draw tess_factor.
fn tess_from_px(px: f32) -> u32 {
    let target_edge_px = max(params.target_px, 1.0);
    let level = u32(ceil(px / target_edge_px));
    return clamp(level, 1u, tess_cap());
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

    let aw = world_pos(a.pos);
    let bw = world_pos(b.pos);
    let cw = world_pos(c.pos);
    let center = (aw + bw + cw) / 3.0;
    let cap = tess_cap();
    let vbase = tri * verts_per_tri(cap);
    let ibase = tri * idx_per_tri(cap);

    // Skip triangles that can't possibly reach the screen: fully outside
    // the camera frustum, or facing away from the camera. Both checks are
    // cheap (a handful of dot products) compared to the O(t^2) subdivision
    // loop below, which is what actually costs GPU time.
    let radius = max(max(distance(center, aw), distance(center, bw)), distance(center, cw))
        + abs(params.scale) + 1e-4;
    if frustum_reject(center, radius) || is_backface(a.nrm, b.nrm, c.nrm, center) {
        let pad_n = idx_per_tri(cap);
        let degener = vbase;
        for (var p = 0u; p < pad_n; p++) {
            dst_idx[ibase + p] = degener;
        }
        return;
    }

    // Adaptive level per edge: keep subdividing until each edge is roughly
    // params.target_px pixels long on screen. This scales with the actual
    // on-screen footprint (distance, FOV and viewport all fall out of the
    // projection), unlike a fixed world-space distance band.
    let te_ab = tess_from_px(edge_px(a.pos, b.pos));
    let te_bc = tess_from_px(edge_px(b.pos, c.pos));
    let te_ca = tess_from_px(edge_px(c.pos, a.pos));
    let t = max(max(te_ab, te_bc), te_ca);

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
