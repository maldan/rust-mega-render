// GPU graph nodes. Matches src/graph/eval.rs (flood fill stays CPU).

struct NodeParams {
    res: u32,
    op: u32,
    kind: u32,
    octaves: u32,
    flags: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    scale: f32,
    seed: f32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
    f4: f32,
    f5: f32,
    color: vec4<f32>,
    color_b: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: NodeParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var in_a: texture_2d<f32>;
@group(0) @binding(3) var in_b: texture_2d<f32>;
@group(0) @binding(4) var in_c: texture_2d<f32>;
@group(0) @binding(5) var ramp_tex: texture_2d<f32>;

const TAU: f32 = 6.283185307179586;
const U32_MAX_F: f32 = 4294967295.0;

fn rem_euclid_i(a: i32, m: i32) -> i32 {
    let r = a % m;
    return select(r + m, r, r >= 0);
}

fn rem_euclid_f(x: f32, m: f32) -> f32 {
    return x - m * floor(x / m);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn hash2(ix: i32, iy: i32, seed: f32) -> f32 {
    var n = bitcast<u32>(ix) * 374761393u
        ^ bitcast<u32>(iy) * 668265263u
        ^ bitcast<u32>(seed) * 2246822519u;
    n = (n ^ (n >> 13u)) * 1274126177u;
    n = n ^ (n >> 16u);
    return f32(n) / U32_MAX_F;
}

fn tile_flag() -> bool {
    return (p.flags & 1u) != 0u;
}

fn has_a() -> bool { return (p.flags & 2u) != 0u; }
fn has_b() -> bool { return (p.flags & 4u) != 0u; }
fn has_c() -> bool { return (p.flags & 8u) != 0u; }

fn luma(c: vec4<f32>) -> f32 {
    return dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn gray4(v: f32) -> vec4<f32> {
    return vec4<f32>(v, v, v, 1.0);
}

fn load_wrap(tex: texture_2d<f32>, x: i32, y: i32) -> vec4<f32> {
    let dim = textureDimensions(tex);
    let w = i32(dim.x);
    let h = i32(dim.y);
    return textureLoad(tex, vec2<i32>(rem_euclid_i(x, w), rem_euclid_i(y, h)), 0);
}

fn sample_uv(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dim = textureDimensions(tex);
    let res = f32(dim.x);
    let x = rem_euclid_f(uv.x, 1.0) * res - 0.5;
    let y = rem_euclid_f(uv.y, 1.0) * res - 0.5;
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let s00 = load_wrap(tex, x0, y0);
    let s10 = load_wrap(tex, x0 + 1, y0);
    let s01 = load_wrap(tex, x0, y0 + 1);
    let s11 = load_wrap(tex, x0 + 1, y0 + 1);
    return mix(mix(s00, s10, fx), mix(s01, s11, fx), fy);
}

fn in_a_px(uv: vec2<f32>) -> vec4<f32> {
    if !has_a() {
        return p.color;
    }
    return sample_uv(in_a, uv);
}

fn in_b_px(uv: vec2<f32>) -> vec4<f32> {
    if !has_b() {
        return p.color_b;
    }
    return sample_uv(in_b, uv);
}

fn in_c_gray(uv: vec2<f32>) -> f32 {
    if !has_c() {
        return p.f0;
    }
    return luma(sample_uv(in_c, uv));
}

fn grad2(ix: i32, iy: i32, seed: f32) -> vec2<f32> {
    let a = hash2(ix, iy, seed) * TAU;
    return vec2<f32>(cos(a), sin(a));
}

fn value_noise(x: f32, y: f32, seed: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = fade(x - f32(x0));
    let fy = fade(y - f32(y0));
    let v00 = hash2(x0, y0, seed);
    let v10 = hash2(x0 + 1, y0, seed);
    let v01 = hash2(x0, y0 + 1, seed);
    let v11 = hash2(x0 + 1, y0 + 1, seed);
    return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

fn tileable_noise(x: f32, y: f32, period: f32, seed: f32) -> f32 {
    return tileable_noise_xy(x, y, period, period, seed);
}

fn tileable_noise_xy(x: f32, y: f32, px: f32, py: f32, seed: f32) -> f32 {
    let pdx = max(px, 1.0);
    let pdy = max(py, 1.0);
    let wx = rem_euclid_f(x, pdx);
    let wy = rem_euclid_f(y, pdy);
    let x0 = i32(floor(wx));
    let y0 = i32(floor(wy));
    let x1 = rem_euclid_i(x0 + 1, i32(ceil(pdx)));
    let y1 = rem_euclid_i(y0 + 1, i32(ceil(pdy)));
    let fx = fade(wx - f32(x0));
    let fy = fade(wy - f32(y0));
    let v00 = hash2(x0, y0, seed);
    let v10 = hash2(x1, y0, seed);
    let v01 = hash2(x0, y1, seed);
    let v11 = hash2(x1, y1, seed);
    return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

fn perlin_noise(x: f32, y: f32, seed: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let u = fade(fx);
    let v = fade(fy);
    let g00 = grad2(x0, y0, seed);
    let g10 = grad2(x0 + 1, y0, seed);
    let g01 = grad2(x0, y0 + 1, seed);
    let g11 = grad2(x0 + 1, y0 + 1, seed);
    let n00 = g00.x * fx + g00.y * fy;
    let n10 = g10.x * (fx - 1.0) + g10.y * fy;
    let n01 = g01.x * fx + g01.y * (fy - 1.0);
    let n11 = g11.x * (fx - 1.0) + g11.y * (fy - 1.0);
    let n = mix(mix(n00, n10, u), mix(n01, n11, u), v);
    return clamp(n * 0.5 + 0.5, 0.0, 1.0);
}

fn tileable_perlin(x: f32, y: f32, period: f32, seed: f32) -> f32 {
    let pmod = i32(ceil(max(period, 1.0)));
    let wx = rem_euclid_f(x, max(period, 1.0));
    let wy = rem_euclid_f(y, max(period, 1.0));
    let x0 = i32(floor(wx));
    let y0 = i32(floor(wy));
    let x1 = rem_euclid_i(x0 + 1, pmod);
    let y1 = rem_euclid_i(y0 + 1, pmod);
    let fx = wx - f32(x0);
    let fy = wy - f32(y0);
    let u = fade(fx);
    let v = fade(fy);
    let g00 = grad2(x0, y0, seed);
    let g10 = grad2(x1, y0, seed);
    let g01 = grad2(x0, y1, seed);
    let g11 = grad2(x1, y1, seed);
    let n00 = g00.x * fx + g00.y * fy;
    let n10 = g10.x * (fx - 1.0) + g10.y * fy;
    let n01 = g01.x * fx + g01.y * (fy - 1.0);
    let n11 = g11.x * (fx - 1.0) + g11.y * (fy - 1.0);
    let n = mix(mix(n00, n10, u), mix(n01, n11, u), v);
    return clamp(n * 0.5 + 0.5, 0.0, 1.0);
}

fn gauss_cell(ix: i32, iy: i32, seed: f32) -> f32 {
    let u1 = clamp(hash2(ix, iy, seed), 1e-6, 0.999999);
    let u2 = hash2(ix, iy, seed + 19.0);
    let g = sqrt(-2.0 * log(u1)) * cos(u2 * TAU);
    return clamp(g * 0.22 + 0.5, 0.0, 1.0);
}

fn gauss_noise(x: f32, y: f32, seed: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = fade(x - f32(x0));
    let fy = fade(y - f32(y0));
    return mix(
        mix(gauss_cell(x0, y0, seed), gauss_cell(x0 + 1, y0, seed), fx),
        mix(gauss_cell(x0, y0 + 1, seed), gauss_cell(x0 + 1, y0 + 1, seed), fx),
        fy,
    );
}

fn tileable_gauss(x: f32, y: f32, period: f32, seed: f32) -> f32 {
    let pmod = i32(ceil(max(period, 1.0)));
    let wx = rem_euclid_f(x, max(period, 1.0));
    let wy = rem_euclid_f(y, max(period, 1.0));
    let x0 = i32(floor(wx));
    let y0 = i32(floor(wy));
    let x1 = rem_euclid_i(x0 + 1, pmod);
    let y1 = rem_euclid_i(y0 + 1, pmod);
    let fx = fade(wx - f32(x0));
    let fy = fade(wy - f32(y0));
    return mix(
        mix(gauss_cell(x0, y0, seed), gauss_cell(x1, y0, seed), fx),
        mix(gauss_cell(x0, y1, seed), gauss_cell(x1, y1, seed), fx),
        fy,
    );
}

fn voronoi_cell_point(ix: i32, iy: i32, seed: f32) -> vec2<f32> {
    return vec2<f32>(f32(ix) + hash2(ix, iy, seed), f32(iy) + hash2(ix, iy, seed + 31.0));
}

fn wrap_delta(d: f32, per: f32) -> f32 {
    if d > per * 0.5 {
        return d - per;
    }
    if d < -per * 0.5 {
        return d + per;
    }
    return d;
}

fn voronoi_dists(x: f32, y: f32, seed: f32, tileable: bool, period: f32) -> vec2<f32> {
    let ix = i32(floor(x));
    let iy = i32(floor(y));
    let pmod = i32(ceil(max(period, 1.0)));
    var f1 = 1e30;
    var f2 = 1e30;
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            var cx = ix + ox;
            var cy = iy + oy;
            if tileable {
                cx = rem_euclid_i(cx, pmod);
                cy = rem_euclid_i(cy, pmod);
            }
            let pt = voronoi_cell_point(cx, cy, seed);
            var dx = pt.x - x;
            var dy = pt.y - y;
            if tileable {
                let per = f32(pmod);
                dx = wrap_delta(dx, per);
                dy = wrap_delta(dy, per);
            }
            let d = sqrt(dx * dx + dy * dy);
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    return vec2<f32>(f1, f2);
}

fn fbm_value_perlin(u: f32, v: f32, perlin: bool) -> f32 {
    var amp = 1.0;
    var freq = max(p.scale, 0.01);
    var sum = 0.0;
    var norm = 0.0;
    let oct = p.octaves;
    let tile = tile_flag();
    for (var o = 0u; o < oct; o++) {
        let s = p.seed + f32(o) * 17.0;
        var nx: f32;
        if perlin {
            if tile {
                nx = tileable_perlin(u * freq, v * freq, freq, s);
            } else {
                nx = perlin_noise(u * freq, v * freq, s);
            }
        } else {
            if tile {
                nx = tileable_noise(u * freq, v * freq, freq, s);
            } else {
                nx = value_noise(u * freq, v * freq, s);
            }
        }
        sum += nx * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return clamp(sum / max(norm, 1e-8), 0.0, 1.0);
}

fn cloud_noise(u: f32, v: f32) -> f32 {
    var amp = 1.0;
    var freq = max(p.scale, 0.01);
    var sum = 0.0;
    var norm = 0.0;
    let oct = p.octaves;
    let tile = tile_flag();
    for (var o = 0u; o < oct; o++) {
        let s = p.seed + f32(o) * 17.0;
        var nx: f32;
        if tile {
            nx = tileable_perlin(u * freq, v * freq, freq, s);
        } else {
            nx = perlin_noise(u * freq, v * freq, s);
        }
        sum += nx * amp;
        norm += amp;
        amp *= 0.55;
        freq *= 2.15;
    }
    let n = clamp(sum / max(norm, 1e-8), 0.0, 1.0);
    return clamp(n * n * (3.0 - 2.0 * n), 0.0, 1.0);
}

fn aniso_noise(u: f32, v: f32) -> f32 {
    let c = cos(p.f0);
    let s = sin(p.f0);
    let du = u - 0.5;
    let dv = v - 0.5;
    let along = du * c + dv * s + 0.5;
    let across = -du * s + dv * c + 0.5;
    let stretch = max(p.f1, 1.0);
    var amp = 1.0;
    var freq = max(p.scale, 0.01);
    var sum = 0.0;
    var norm = 0.0;
    let oct = p.octaves;
    let tile = tile_flag();
    for (var o = 0u; o < oct; o++) {
        let sd = p.seed + f32(o) * 17.0;
        let nx = along * freq;
        let ny = across * freq * stretch;
        var n: f32;
        if tile {
            n = tileable_noise_xy(nx, ny, freq, freq * stretch, sd);
        } else {
            n = value_noise(nx, ny, sd);
        }
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return clamp(sum / max(norm, 1e-8), 0.0, 1.0);
}

fn eval_noise(u: f32, v: f32) -> f32 {
    let scale = max(p.scale, 0.01);
    let seed = p.seed;
    let tile = tile_flag();
    switch p.kind {
        case 0u, 1u: {
            return fbm_value_perlin(u, v, p.kind == 1u);
        }
        case 2u: {
            return clamp(voronoi_dists(u * scale, v * scale, seed, tile, scale).x * 1.25, 0.0, 1.0);
        }
        case 3u: {
            let d = voronoi_dists(u * scale, v * scale, seed, tile, scale);
            return clamp((d.y - d.x) * 2.0, 0.0, 1.0);
        }
        case 4u: {
            if tile {
                return tileable_gauss(u * scale, v * scale, scale, seed);
            }
            return gauss_noise(u * scale, v * scale, seed);
        }
        case 6u: {
            return aniso_noise(u, v);
        }
        default: {
            return cloud_noise(u, v);
        }
    }
}

fn eval_gradient(u: f32, v: f32) -> f32 {
    if p.kind == 1u {
        let d = vec2<f32>(u - 0.5, v - 0.5);
        return clamp(1.0 - length(d) * 2.0, 0.0, 1.0);
    }
    return v;
}

fn eval_lines(u: f32, v: f32) -> f32 {
    let count = max(p.scale, 1.0);
    let half_w = max(p.f0 * 0.5, 0.0);
    let period = 1.0 / count;
    let s = sin(p.f2);
    let c = cos(p.f2);
    let t = (u - 0.5) * c + (v - 0.5) * s + 0.5;
    let local = rem_euclid_f(t, period);
    let dist = abs(local - period * 0.5);
    if dist <= half_w {
        return p.f3;
    }
    return p.f4;
}

fn eval_checker(u: f32, v: f32) -> f32 {
    let scale = max(round(p.scale), 1.0);
    let cx = i32(floor(u * scale));
    let cy = i32(floor(v * scale));
    if ((cx + cy) & 1) == 0 {
        return p.f0;
    }
    return p.f1;
}

fn sd_rounded_box(px: f32, py: f32, hx: f32, hy: f32, radius: f32) -> f32 {
    let r = clamp(radius, 0.0, min(hx, hy));
    let qx = abs(px) - (hx - r);
    let qy = abs(py) - (hy - r);
    let ox = max(qx, 0.0);
    let oy = max(qy, 0.0);
    return sqrt(ox * ox + oy * oy) + min(max(qx, qy), 0.0) - r;
}

fn cell_weight(i: i32, seed: f32, size_rand: f32) -> f32 {
    let r = hash2(i, 0, seed);
    return max(1.0 + size_rand * (r * 2.0 - 1.0) * 0.75, 0.15);
}

fn find_split_w(count: i32, t: f32, seed: f32, size_rand: f32) -> vec3<f32> {
    var sum = 0.0;
    for (var i = 0; i < 64; i++) {
        if i >= count { break; }
        sum += cell_weight(i, seed, size_rand);
    }
    let tt = clamp(t, 0.0, 0.999999);
    var acc = 0.0;
    for (var i = 0; i < 64; i++) {
        if i >= count { break; }
        let w = cell_weight(i, seed, size_rand) / sum;
        let nxt = acc + w;
        if tt < nxt || i + 1 == count {
            return vec3<f32>(f32(i), acc, nxt);
        }
        acc = nxt;
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn eval_tile(u: f32, v: f32) -> f32 {
    let nx = i32(p.octaves);
    let ny = i32(p.kind);
    let gap = p.f0;
    let size_rand_x = p.f1;
    let offset = p.f2;
    let roundness = p.f3;
    let size_rand_y = p.f4;
    let seed = p.seed;
    let row = find_split_w(ny, v, seed + 3.0, size_rand_y);
    let ri = i32(row.x);
    let v0 = row.y;
    let v1 = row.z;
    let cell_h = max(v1 - v0, 1e-5);
    var row_shift: f32;
    if (ri & 1) == 1 {
        row_shift = offset * 0.5;
    } else {
        row_shift = offset * 0.15 * (hash2(ri, 0, seed + 91.0) - 0.5);
    }
    let u_shift = rem_euclid_f(u - row_shift, 1.0);
    let col = find_split_w(nx, u_shift, seed + 17.0 + f32(ri) * 31.0, size_rand_x);
    let u0 = col.y;
    let u1 = col.z;
    let cell_w = max(u1 - u0, 1e-5);
    let lu = u_shift - u0;
    let lv = v - v0;
    let g = gap * min(1.0 / max(f32(nx), 1.0), 1.0 / max(f32(ny), 1.0));
    let gap_u = g;
    let gap_v = g;
    let inner_w = max(cell_w - 2.0 * gap_u, 0.0);
    let inner_h = max(cell_h - 2.0 * gap_v, 0.0);
    let hx = inner_w * 0.5;
    let hy = inner_h * 0.5;
    let radius = roundness * min(hx, hy);
    if hx <= 1e-6 || hy <= 1e-6 {
        return 0.0;
    }
    let sd = sd_rounded_box(lu - (gap_u + hx), lv - (gap_v + hy), hx, hy, radius);
    if sd <= 0.0 {
        return 1.0;
    }
    return 0.0;
}

fn eval_bricks(u: f32, v: f32) -> f32 {
    let nx = max(p.scale, 1.0);
    let ny = max(p.f5, 1.0);
    let gap = p.f0;
    let offset = p.f1;
    let roundness = p.f2;
    let bevel = p.f3;
    let cell_w = 1.0 / nx;
    let cell_h = 1.0 / ny;
    let row = floor(v * ny);
    var row_shift = 0.0;
    if (i32(row) & 1) == 1 {
        row_shift = offset * cell_w;
    }
    let u_shift = rem_euclid_f(u - row_shift, 1.0);
    let col = floor(u_shift * nx);
    let lu = u_shift - col * cell_w;
    let lv = v - row * cell_h;
    let gap_u = gap * cell_w;
    let gap_v = gap * cell_h;
    let inner_w = max(cell_w - 2.0 * gap_u, 0.0);
    let inner_h = max(cell_h - 2.0 * gap_v, 0.0);
    let hx = inner_w * 0.5;
    let hy = inner_h * 0.5;
    let radius = roundness * min(hx, hy);
    var sd: f32;
    if hx <= 1e-6 || hy <= 1e-6 {
        sd = 1.0;
    } else {
        sd = sd_rounded_box(lu - (gap_u + hx), lv - (gap_v + hy), hx, hy, radius);
    }
    if bevel <= 1e-6 {
        if sd <= 0.0 {
            return 1.0;
        }
        return 0.0;
    }
    let falloff = bevel * max(min(hx, hy), 1e-6);
    return clamp(-sd / falloff, 0.0, 1.0);
}

fn blend_ch(a: f32, b: f32, mode: u32) -> f32 {
    switch mode {
        case 1u: { return a * b; }
        case 2u: { return min(a + b, 1.0); }
        case 3u: {
            if a < 0.5 {
                return 2.0 * a * b;
            }
            return 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
        }
        case 4u: { return 1.0 - (1.0 - a) * (1.0 - b); }
        case 5u: { return clamp(a / max(b, 1e-4), 0.0, 1.0); }
        case 6u: { return max(a - b, 0.0); }
        case 7u: { return abs(a - b); }
        case 8u: { return min(a, b); }
        case 9u: { return max(a, b); }
        default: { return b; }
    }
}

fn eval_blend(uv: vec2<f32>) -> vec4<f32> {
    let ca = in_a_px(uv);
    let cb = in_b_px(uv);
    var m: f32;
    if has_c() {
        m = luma(sample_uv(in_c, uv));
    } else {
        m = p.f0;
    }
    var outv: vec4<f32>;
    for (var k = 0; k < 4; k++) {
        let t = blend_ch(ca[k], cb[k], p.kind);
        outv[k] = ca[k] + (t - ca[k]) * m;
    }
    return outv;
}

fn eval_levels(g: f32) -> f32 {
    let in_lo = p.f0;
    let in_hi = max(p.f1, in_lo + 1e-5);
    let gamma = max(p.f2, 0.01);
    var v = clamp((g - in_lo) / (in_hi - in_lo), 0.0, 1.0);
    v = pow(v, 1.0 / gamma);
    v = p.f3 + v * (p.f4 - p.f3);
    return clamp(v, 0.0, 1.0);
}

fn eval_height_normal(gid: vec2<i32>) -> vec4<f32> {
    if !has_a() {
        return vec4<f32>(0.5, 0.5, 1.0, 1.0);
    }
    let l = luma(load_wrap(in_a, gid.x - 1, gid.y));
    let r = luma(load_wrap(in_a, gid.x + 1, gid.y));
    let d = luma(load_wrap(in_a, gid.x, gid.y - 1));
    let u = luma(load_wrap(in_a, gid.x, gid.y + 1));
    let strength = p.f0;
    let dx = (r - l) * strength;
    let dy = (u - d) * strength;
    let n = vec3<f32>(-dx, -dy, 1.0);
    let len = max(length(n), 1e-8);
    let nn = n / len;
    return vec4<f32>(nn * 0.5 + 0.5, 1.0);
}

fn eval_curvature(gid: vec2<i32>) -> vec4<f32> {
    if !has_a() {
        return gray4(0.5);
    }
    let r = max(i32(p.f1), 1);
    let c = luma(load_wrap(in_a, gid.x, gid.y));
    let l = luma(load_wrap(in_a, gid.x - r, gid.y));
    let right = luma(load_wrap(in_a, gid.x + r, gid.y));
    let d = luma(load_wrap(in_a, gid.x, gid.y - r));
    let u = luma(load_wrap(in_a, gid.x, gid.y + r));
    let lap = l + right + u + d - 4.0 * c;
    return gray4(clamp(0.5 - lap * p.f0 * 0.25, 0.0, 1.0));
}

fn eval_color_ramp(t: f32) -> vec4<f32> {
    let x = i32(clamp(t, 0.0, 1.0) * 255.0);
    return textureLoad(ramp_tex, vec2<i32>(x, 0), 0);
}

fn eval_distort(uv: vec2<f32>) -> vec4<f32> {
    let strength = p.f0;
    let scale = max(p.scale, 0.01);
    let seed = p.seed;
    let dx = value_noise(uv.x * scale, uv.y * scale, seed) * 2.0 - 1.0;
    let dy = value_noise(uv.x * scale, uv.y * scale, seed + 19.0) * 2.0 - 1.0;
    return in_a_px(uv + vec2<f32>(dx, dy) * strength);
}

fn eval_warp(uv: vec2<f32>) -> vec4<f32> {
    let strength = p.f0;
    let eps = 1.0 / f32(p.res);
    let dx = luma(sample_uv(in_b, uv + vec2<f32>(eps, 0.0))) - luma(sample_uv(in_b, uv - vec2<f32>(eps, 0.0)));
    let dy = luma(sample_uv(in_b, uv + vec2<f32>(0.0, eps))) - luma(sample_uv(in_b, uv - vec2<f32>(0.0, eps)));
    let s = strength * 0.5 / eps;
    return in_a_px(uv + vec2<f32>(dx, dy) * s);
}

fn eval_dir_warp(uv: vec2<f32>) -> vec4<f32> {
    let amt = luma(in_b_px(uv)) * p.f0;
    let dir = vec2<f32>(cos(p.f1), sin(p.f1));
    return in_a_px(uv + dir * amt);
}

fn eval_slope_blur(uv: vec2<f32>) -> vec4<f32> {
    let eps = 1.0 / f32(p.res);
    let scale = p.f0 * 0.5 / eps;
    let gx = (luma(sample_uv(in_b, uv + vec2<f32>(eps, 0.0))) - luma(sample_uv(in_b, uv - vec2<f32>(eps, 0.0)))) * scale;
    let gy = (luma(sample_uv(in_b, uv + vec2<f32>(0.0, eps))) - luma(sample_uv(in_b, uv - vec2<f32>(0.0, eps)))) * scale;
    let samples = i32(p.octaves);
    let denom = max(f32(samples - 1), 1.0);
    var acc = vec4<f32>(0.0);
    if p.kind == 1u {
        acc = vec4<f32>(1e30);
    } else if p.kind == 2u {
        acc = vec4<f32>(-1e30);
    }
    for (var i = 0; i < 32; i++) {
        if i >= samples { break; }
        let t = f32(i) / denom;
        let c = in_a_px(uv - vec2<f32>(gx, gy) * t);
        if p.kind == 1u {
            acc = min(acc, c);
        } else if p.kind == 2u {
            acc = max(acc, c);
        } else {
            acc += c;
        }
    }
    if p.kind == 0u {
        acc /= f32(samples);
    }
    return acc;
}

fn eval_blur_h(gid: vec2<i32>) -> vec4<f32> {
    let l = load_wrap(in_a, gid.x - 1, gid.y);
    let c = load_wrap(in_a, gid.x, gid.y);
    let r = load_wrap(in_a, gid.x + 1, gid.y);
    return (l + c * 2.0 + r) * 0.25;
}

fn eval_blur_v(gid: vec2<i32>) -> vec4<f32> {
    let u = load_wrap(in_a, gid.x, gid.y - 1);
    let c = load_wrap(in_a, gid.x, gid.y);
    let d = load_wrap(in_a, gid.x, gid.y + 1);
    return (u + c * 2.0 + d) * 0.25;
}

fn eval_blur_mix(uv: vec2<f32>) -> vec4<f32> {
    let src = sample_uv(in_a, uv);
    let blur = sample_uv(in_b, uv);
    let t = clamp(luma(sample_uv(in_c, uv)), 0.0, 1.0);
    return mix(src, blur, t);
}

fn load_clamp(tex: texture_2d<f32>, x: i32, y: i32) -> vec4<f32> {
    let dim = textureDimensions(tex);
    let w = i32(dim.x);
    let h = i32(dim.y);
    let cx = clamp(x, 0, w - 1);
    let cy = clamp(y, 0, h - 1);
    return textureLoad(tex, vec2<i32>(cx, cy), 0);
}

fn sample_uv_clamp(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let dim = textureDimensions(tex);
    let res = f32(dim.x);
    let x = uv.x * res - 0.5;
    let y = uv.y * res - 0.5;
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let s00 = load_clamp(tex, x0, y0);
    let s10 = load_clamp(tex, x0 + 1, y0);
    let s01 = load_clamp(tex, x0, y0 + 1);
    let s11 = load_clamp(tex, x0 + 1, y0 + 1);
    return mix(mix(s00, s10, fx), mix(s01, s11, fx), fy);
}

fn inside_ngon(p: vec2<f32>, sides: f32, radius: f32) -> bool {
    let ns = max(sides, 3.0);
    let ang = atan2(p.x, -p.y);
    let sector = TAU / ns;
    let a = rem_euclid_f(ang + sector * 0.5, sector) - sector * 0.5;
    return length(p) * cos(a) <= radius * cos(3.14159265 / ns);
}

fn eval_shape(u: f32, v: f32) -> f32 {
    let q = vec2<f32>(u - 0.5, v - 0.5);
    let sx = max(p.f0, 1e-6);
    let sy = max(p.f1, 1e-6);
    var hit = false;
    switch p.kind {
        case 0u: {
            hit = abs(q.x) <= sx * 0.5 && abs(q.y) <= sy * 0.5;
        }
        case 1u: {
            let rx = sx * 0.5;
            let ry = sy * 0.5;
            hit = (q.x / rx) * (q.x / rx) + (q.y / ry) * (q.y / ry) <= 1.0;
        }
        case 2u: {
            hit = inside_ngon(q, 3.0, sx * 0.5);
        }
        default: {
            hit = inside_ngon(q, f32(max(p.octaves, 3u)), sx * 0.5);
        }
    }
    return select(0.0, 1.0, hit);
}

fn eval_transform(uv: vec2<f32>) -> vec4<f32> {
    var q = uv - vec2<f32>(0.5) - vec2<f32>(p.f0, p.f1);
    q.x /= max(p.f2, 1e-6);
    q.y /= max(p.f3, 1e-6);
    let c = cos(p.f4);
    let s = sin(p.f4);
    let src = vec2<f32>(q.x * c + q.y * s, -q.x * s + q.y * c) + vec2<f32>(0.5);
    if tile_flag() {
        return sample_uv(in_a, src);
    }
    return sample_uv_clamp(in_a, src);
}

fn eval_tile_sampler(uv: vec2<f32>) -> f32 {
    if !has_a() {
        return 0.0;
    }
    let nx = max(f32(p.octaves), 1.0);
    let ny = max(f32(p.kind), 1.0);
    let inx = i32(nx);
    let iny = i32(ny);
    let bx = i32(floor(uv.x * nx));
    let by = i32(floor(uv.y * ny));
    var acc = 0.0;
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let jx = bx + ox;
            let jy = by + oy;
            let hx = rem_euclid_i(jx, inx);
            let hy = rem_euclid_i(jy, iny);
            let offx = (hash2(hx, hy, p.seed) - 0.5) * 2.0 * p.f0 / nx;
            let offy = (hash2(hx, hy, p.seed + 7.0) - 0.5) * 2.0 * p.f0 / ny;
            let ang = (hash2(hx, hy, p.seed + 13.0) - 0.5) * 2.0 * p.f1 * 3.14159265;
            let sc = max(1.0 + (hash2(hx, hy, p.seed + 19.0) - 0.5) * 2.0 * p.f2, 0.05);
            let cx = (f32(jx) + 0.5) / nx + offx;
            let cy = (f32(jy) + 0.5) / ny + offy;
            var q = vec2<f32>((uv.x - cx) * nx, (uv.y - cy) * ny) / sc;
            let c = cos(ang);
            let s = sin(ang);
            let src = vec2<f32>(q.x * c + q.y * s, -q.x * s + q.y * c) + vec2<f32>(0.5);
            acc = max(acc, luma(sample_uv_clamp(in_a, src)));
        }
    }
    return acc;
}

@compute @workgroup_size(8, 8)
fn node_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.res || gid.y >= p.res {
        return;
    }
    let pi = vec2<i32>(i32(gid.x), i32(gid.y));
    let u = (f32(gid.x) + 0.5) / f32(p.res);
    let v = (f32(gid.y) + 0.5) / f32(p.res);
    let uv = vec2<f32>(u, v);
    var px: vec4<f32>;
    switch p.op {
        case 1u: { px = p.color; }
        case 2u: { px = gray4(eval_gradient(u, v)); }
        case 3u: { px = gray4(eval_lines(u, v)); }
        case 4u: { px = gray4(eval_checker(u, v)); }
        case 5u: { px = gray4(eval_tile(u, v)); }
        case 6u: { px = gray4(eval_bricks(u, v)); }
        case 7u: {
            let c = in_a_px(uv);
            if p.kind == 1u {
                px = vec4<f32>(1.0 - c.rgb, c.a);
            } else {
                px = gray4(1.0 - luma(c));
            }
        }
        case 8u: { px = gray4(eval_levels(luma(in_a_px(uv)))); }
        case 9u, 10u: { let g = luma(in_a_px(uv)); px = vec4<f32>(g, g, g, 1.0); }
        case 11u: { px = eval_color_ramp(luma(in_a_px(uv))); }
        case 12u: { px = eval_blend(uv); }
        case 13u: { px = eval_height_normal(pi); }
        case 14u: { px = eval_distort(uv); }
        case 15u: { px = eval_warp(uv); }
        case 16u: { px = eval_dir_warp(uv); }
        case 17u: { px = eval_blur_h(pi); }
        case 18u: { px = eval_blur_v(pi); }
        case 19u: { px = eval_blur_mix(uv); }
        case 20u: { px = eval_slope_blur(uv); }
        case 21u: { px = in_a_px(uv); }
        case 22u: { px = eval_curvature(pi); }
        case 23u: { px = gray4(eval_shape(u, v)); }
        case 24u: { px = gray4(luma(eval_transform(uv))); }
        case 25u: { px = gray4(eval_tile_sampler(uv)); }
        default: { px = gray4(eval_noise(u, v)); }
    }
    textureStore(out_tex, pi, px);
}
