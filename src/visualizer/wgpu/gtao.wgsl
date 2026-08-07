//! Horizon-based GTAO for LH view (+Z forward, DirectX depth).
//!
//! Uses interleaved gradient noise (not a tiled noise map) so slice rotation
//! does not form a visible pixel grid.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    let xy = p[vi];
    var o: VsOut;
    o.pos = vec4(xy, 0.0, 1.0);
    o.uv = vec2(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    return o;
}

struct GtaoUniforms {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    resolution: vec2<f32>,
    radius: f32,
    thickness: f32,
    /// x = directions, y = steps
    params: vec4<f32>,
    noise_scale: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: GtaoUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_samp: sampler;
@group(0) @binding(5) var noise_tex: texture_2d<f32>;
@group(0) @binding(6) var noise_samp: sampler;

const PI: f32 = 3.14159265;

fn view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = u.inv_proj * clip;
    return v.xyz / v.w;
}

fn view_normal_from_depth(uv: vec2<f32>, pos: vec3<f32>) -> vec3<f32> {
    let texel = 1.0 / u.resolution;
    let dx = textureSample(depth_tex, depth_samp, uv + vec2(texel.x, 0.0));
    let dy = textureSample(depth_tex, depth_samp, uv + vec2(0.0, texel.y));
    let px = view_pos(uv + vec2(texel.x, 0.0), dx);
    let py = view_pos(uv + vec2(0.0, texel.y), dy);
    var n = cross(py - pos, px - pos);
    if dot(n, -pos) < 0.0 {
        n = -n;
    }
    let len2 = dot(n, n);
    if len2 < 1e-10 {
        return normalize(-pos);
    }
    return n * inverseSqrt(len2);
}

/// Interleaved gradient noise — no tiling pattern (unlike a 4×4 noise atlas).
fn ign(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn march_horizon(
    uv0: vec2<f32>,
    pos: vec3<f32>,
    normal: vec3<f32>,
    dir: vec2<f32>,
    uv_radius: f32,
    radius: f32,
    steps: u32,
    jitter: f32,
) -> f32 {
    let r2 = radius * radius;
    var max_h = 0.05;
    var ao = 0.0;

    for (var s = 1u; s <= steps; s++) {
        let sn = (f32(s) - 0.5 + jitter) / f32(steps);
        let t = sn * sn;
        let sample_uv = uv0 + dir * (uv_radius * t);
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        let sd = textureSample(depth_tex, depth_samp, sample_uv);
        if sd >= 0.9999 {
            continue;
        }

        let sample_pos = view_pos(sample_uv, sd);
        let delta = sample_pos - pos;
        let dist2 = dot(delta, delta);
        if dist2 < 1e-8 || dist2 > r2 {
            continue;
        }

        let vn = dot(delta, normal);
        if vn <= 0.0 {
            continue;
        }

        let dist = sqrt(dist2);
        let att = max(1.0 - dist2 / r2, 0.0);
        let h = (att * att) * (vn / (dist + 1e-4));

        let add = max(h - max_h, 0.0);
        ao += add;
        max_h = max(max_h, h);
    }

    return ao;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    if depth >= 0.9999 {
        return vec4(1.0);
    }

    let pos = view_pos(i.uv, depth);

    var normal = textureSampleLevel(normal_tex, normal_samp, i.uv, 0.0).xyz;
    if dot(normal, normal) > 1e-6 {
        normal = normalize((u.view * vec4(normal, 0.0)).xyz);
    } else {
        normal = view_normal_from_depth(i.uv, pos);
    }
    if dot(normal, -pos) < 0.0 {
        normal = -normal;
    }

    let directions = u32(clamp(u.params.x, 2.0, 8.0));
    let steps = u32(clamp(u.params.y, 2.0, 16.0));

    let pixel = i.uv * u.resolution;
    // Two uncorrelated noise channels — no repeating atlas tiles.
    let n0 = ign(pixel);
    let n1 = hash21(pixel + vec2(19.19, 7.7));

    let radius = max(u.radius, 0.05) * max(abs(pos.z), 0.35) * 0.16;

    let clip0 = u.proj * vec4(pos, 1.0);
    let clip1 = u.proj * vec4(pos + vec3(radius, 0.0, 0.0), 1.0);
    let ndc0 = clip0.xy / max(clip0.w, 1e-5);
    let ndc1 = clip1.xy / max(clip1.w, 1e-5);
    let uv_a = vec2(ndc0.x * 0.5 + 0.5, 0.5 - ndc0.y * 0.5);
    let uv_b = vec2(ndc1.x * 0.5 + 0.5, 0.5 - ndc1.y * 0.5);
    // Keep radius continuous in UV (avoid snapping to texel grid).
    let uv_radius = max(length(uv_b - uv_a), 1.5 / max(u.resolution.x, 1.0));

    var obscurance = 0.0;
    for (var slice = 0u; slice < directions; slice++) {
        let phi = (f32(slice) + n0) * PI / f32(directions);
        let omega = vec2(cos(phi), sin(phi));

        let a = march_horizon(i.uv, pos, normal, omega, uv_radius, radius, steps, n1);
        let b = march_horizon(i.uv, pos, normal, -omega, uv_radius, radius, steps, 1.0 - n1);
        obscurance += a + b;
    }

    obscurance /= max(f32(directions), 1.0);
    let strength = mix(1.35, 0.55, clamp((u.thickness - 0.2) / 2.8, 0.0, 1.0));
    var ao = 1.0 / (1.0 + obscurance * strength);
    ao = mix(ao, 1.0, 0.06);
    return vec4(vec3(clamp(ao, 0.0, 1.0)), 1.0);
}
