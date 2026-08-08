//! Spatial screen-space global illumination.
//! LH view (+Z forward), DirectX clip depth.
//!
//! Cosine-weighted hemisphere with R2 low-discrepancy directions,
//! short screen-space march, gather lit HDR as irradiance,
//! then `albedo * (1 - metallic) * boost`.
//! Alpha stores clip depth for temporal reprojection.

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

struct SsgiUniforms {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    /// Half-res RT size (pass resolution).
    resolution: vec2<f32>,
    radius: f32,
    thickness: f32,
    /// x = samples, y = bias, z = max march steps, w = frame index (wrapped)
    params: vec4<f32>,
    /// Full framebuffer size — used for noise so half-res doesn't stripe.
    full_resolution: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: SsgiUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_samp: sampler;
@group(0) @binding(5) var color_tex: texture_2d<f32>;
@group(0) @binding(6) var color_samp: sampler;
@group(0) @binding(7) var albedo_tex: texture_2d<f32>;
@group(0) @binding(8) var albedo_samp: sampler;
@group(0) @binding(9) var orm_tex: texture_2d<f32>;
@group(0) @binding(10) var orm_samp: sampler;

const PI: f32 = 3.14159265;
/// R2 irrational bases (Roberts 2018) — low discrepancy in 2D.
const R2_A: f32 = 0.7548776662466927;
const R2_B: f32 = 0.5698402909980532;

fn view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = u.inv_proj * clip;
    return v.xyz / v.w;
}

fn project_uv(view_p: vec3<f32>) -> vec3<f32> {
    let clip = u.proj * vec4(view_p, 1.0);
    let ndc = clip.xyz / max(clip.w, 1e-6);
    let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3(uv, ndc.z);
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2(hash21(p), hash21(p + vec2(13.1, 7.7)));
}

/// Stable per-pixel isotropic noise (quantized to texel).
fn noise2(pixel: vec2<f32>) -> vec2<f32> {
    return hash22(floor(pixel));
}

/// R2 sequence sample `i` (0-based), Cranley–Patterson rotated by `offset`.
fn r2_cp(i: f32, offset: vec2<f32>) -> vec2<f32> {
    return fract(vec2(0.5) + vec2(i * R2_A, i * R2_B) + offset);
}

fn basis(n: vec3<f32>) -> mat3x3<f32> {
    var t: vec3<f32>;
    if abs(n.z) < 0.999 {
        t = normalize(cross(n, vec3(0.0, 0.0, 1.0)));
    } else {
        t = normalize(cross(n, vec3(0.0, 1.0, 0.0)));
    }
    let b = cross(n, t);
    return mat3x3(t, b, n);
}

fn edge_fade(uv: vec2<f32>) -> f32 {
    let e = smoothstep(vec2(0.0), vec2(0.05), uv) * smoothstep(vec2(0.0), vec2(0.05), 1.0 - uv);
    return e.x * e.y;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

/// Soft-knee HDR clamp — less harsh than hard scale, still kills sparkles.
fn suppress_firefly(c: vec3<f32>, max_luma: f32) -> vec3<f32> {
    let y = luma(c);
    if y <= max_luma {
        return c;
    }
    // Soft knee: asymptote toward max_luma instead of hard cut.
    let soft = max_luma * (1.0 + log2(1.0 + (y - max_luma) / max(max_luma, 1e-3)));
    let capped = min(soft, max_luma * 1.75);
    return c * (capped / y);
}

/// 5-tap local luminance — stabler firefly reference than a single texel.
fn local_luma_ref(uv: vec2<f32>) -> f32 {
    let texel = 1.0 / max(u.full_resolution, vec2(1.0));
    var y = luma(textureSampleLevel(color_tex, color_samp, uv, 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv + vec2(texel.x, 0.0), 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv - vec2(texel.x, 0.0), 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv + vec2(0.0, texel.y), 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv - vec2(0.0, texel.y), 0.0).rgb);
    return max(y * 0.2, 0.05);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    if depth >= 0.9999 {
        return vec4(0.0, 0.0, 0.0, depth);
    }

    let albedo = textureSampleLevel(albedo_tex, albedo_samp, i.uv, 0.0).rgb;
    let metallic = textureSampleLevel(orm_tex, orm_samp, i.uv, 0.0).b;
    let kd = 1.0 - metallic;
    if kd < 0.02 {
        return vec4(0.0, 0.0, 0.0, depth);
    }

    let pos = view_pos(i.uv, depth);
    var n_world = textureSample(normal_tex, normal_samp, i.uv).xyz;
    if dot(n_world, n_world) < 1e-6 {
        return vec4(0.0, 0.0, 0.0, depth);
    }
    n_world = normalize(n_world);
    var n_view = normalize((u.view * vec4(n_world, 0.0)).xyz);
    if dot(n_view, -pos) < 0.0 {
        n_view = -n_view;
    }

    let samples = u32(clamp(u.params.x, 4.0, 32.0));
    let bias = max(u.params.y, 0.0);
    let max_steps = u32(clamp(u.params.z, 4.0, 32.0));
    let radius = max(u.radius, 0.05);
    let thickness = max(u.thickness, 0.001);

    // Temporal only rotates the hemisphere — do NOT scroll noise in screen space.
    let slot = u32(u.params.w) % 8u;
    let angle_rot = f32(slot) * (PI * 0.25);

    let pixel = i.uv * u.full_resolution;
    let cp = noise2(pixel); // Cranley–Patterson offset
    let tbn = basis(n_view);

    let local_luma = local_luma_ref(i.uv);
    let hit_max_luma = max(local_luma * 1.75, 0.7);

    var irradiance = vec3(0.0);

    for (var s = 0u; s < samples; s++) {
        // Low-discrepancy direction + independent length/jitter channel.
        let xi = r2_cp(f32(s), cp);
        let xj = r2_cp(f32(s) + 19.0, cp.yx);

        let phi = 2.0 * PI * xi.x + angle_rot;
        // Cosine-weighted hemisphere: θ from √ξ (PDF = cosθ / π).
        let cos_t = sqrt(clamp(xi.y, 0.001, 1.0));
        let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
        let local = vec3(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
        let ray_dir = normalize(tbn * local);

        // Stratify lengths: more short rays (local bounce) + some long ones.
        let ray_len = radius * mix(0.2, 1.0, xj.x);
        let step_len = ray_len / f32(max_steps);
        let jitter = xj.y * step_len;

        var hit = false;
        var hit_uv = i.uv;
        var hit_dist = ray_len;
        var hit_pos = pos;

        for (var step_i = 1u; step_i <= max_steps; step_i++) {
            let t = bias + jitter + step_len * f32(step_i);
            if t > ray_len {
                break;
            }
            let sample_pos = pos + ray_dir * t;
            let projected = project_uv(sample_pos);
            if projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0 {
                break;
            }

            let scene_depth = textureSample(depth_tex, depth_samp, projected.xy);
            if scene_depth >= 0.9999 {
                continue;
            }

            let scene_pos = view_pos(projected.xy, scene_depth);
            let delta = sample_pos.z - scene_pos.z;
            let thick = max(thickness, step_len * 1.5);
            if delta > 0.0 && delta < thick {
                hit = true;
                hit_uv = projected.xy;
                hit_dist = t;
                hit_pos = scene_pos;
                break;
            }
            if delta > thick {
                break;
            }
        }

        if !hit {
            // Miss: no screen-space contributor (ambient / IBL covers the rest via ambient_dim).
            continue;
        }
        if length(hit_uv - i.uv) < 1.5 / max(u.full_resolution.x, 1.0) {
            continue;
        }

        // Reject back-facing / grazing emitters (wrong bounce from thin geometry).
        var hit_n = textureSampleLevel(normal_tex, normal_samp, hit_uv, 0.0).xyz;
        if dot(hit_n, hit_n) > 1e-6 {
            hit_n = normalize((u.view * vec4(normalize(hit_n), 0.0)).xyz);
            let to_recv = normalize(pos - hit_pos);
            let facing = clamp(dot(hit_n, to_recv), 0.0, 1.0);
            if facing < 0.05 {
                continue;
            }
        }

        var radiance = textureSampleLevel(color_tex, color_samp, hit_uv, 0.0).rgb;
        radiance = suppress_firefly(radiance, hit_max_luma);

        let dist_att = exp(-hit_dist / max(radius, 0.05));
        let fade = edge_fade(hit_uv);
        // Mild receiver cosine (already in sampling PDF; keep a soft factor for grazing rays).
        let recv_cos = clamp(dot(n_view, ray_dir), 0.0, 1.0);
        let w = dist_att * fade * mix(0.35, 1.0, recv_cos);

        irradiance += radiance * w;
    }

    // Normalize by sample count (unbiased for cosine MC). Misses correctly darken pockets.
    irradiance /= max(f32(samples), 1.0);
    let boost = 3.5;
    var diffuse_gi = irradiance * albedo * kd * boost;
    diffuse_gi = suppress_firefly(diffuse_gi, hit_max_luma * 1.35);
    return vec4(max(diffuse_gi, vec3(0.0)), depth);
}
