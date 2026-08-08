//! Spatial screen-space global illumination.
//! LH view (+Z forward), DirectX clip depth.
//!
//! Cosine-weighted hemisphere with R2 low-discrepancy directions,
//! Hi-Z hierarchical march (empty-space skip + binary refine),
//! gather lit HDR as **irradiance** (albedo × kd applied in composite).
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
    /// Max Hi-Z mip level.
    hiz_max_mip: f32,
    /// Irradiance energy scale (1 = neutral).
    energy: f32,
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
@group(0) @binding(11) var hiz_tex: texture_2d<f32>;
@group(0) @binding(12) var hiz_samp: sampler;

const PI: f32 = 3.14159265;
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

fn noise2(pixel: vec2<f32>) -> vec2<f32> {
    return hash22(floor(pixel));
}

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

fn suppress_firefly(c: vec3<f32>, max_luma: f32) -> vec3<f32> {
    let y = luma(c);
    if y <= max_luma {
        return c;
    }
    let soft = max_luma * (1.0 + log2(1.0 + (y - max_luma) / max(max_luma, 1e-3)));
    let capped = min(soft, max_luma * 1.75);
    return c * (capped / y);
}

fn local_luma_ref(uv: vec2<f32>) -> f32 {
    let texel = 1.0 / max(u.full_resolution, vec2(1.0));
    var y = luma(textureSampleLevel(color_tex, color_samp, uv, 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv + vec2(texel.x, 0.0), 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv - vec2(texel.x, 0.0), 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv + vec2(0.0, texel.y), 0.0).rgb);
    y += luma(textureSampleLevel(color_tex, color_samp, uv - vec2(0.0, texel.y), 0.0).rgb);
    return max(y * 0.2, 0.05);
}

fn hiz_depth(uv: vec2<f32>, level: f32) -> f32 {
    return textureSampleLevel(hiz_tex, hiz_samp, uv, level).r;
}

/// Perspective-correct Hi-Z march along one GI ray.
/// Returns xyz = hit_uv + hit_dist, w = 1 if hit else 0.
fn march_gi(
    origin: vec3<f32>,
    dir: vec3<f32>,
    ray_len: f32,
    bias: f32,
    thickness: f32,
    max_steps: u32,
    jitter: f32,
) -> vec4<f32> {
    let max_mip = i32(clamp(u.hiz_max_mip, 0.0, 8.0));
    let base_step = ray_len / f32(max(max_steps, 1u));
    var t = bias + jitter * base_step;
    var prev_t = bias;
    var step_i = 0u;

    loop {
        if step_i >= max_steps || t > ray_len {
            break;
        }

        let sample_pos = origin + dir * t;
        let projected = project_uv(sample_pos);
        if projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0 {
            break;
        }
        if projected.z < 0.0 || projected.z > 1.0 {
            break;
        }

        // Coarser mip farther along the ray — skip empty space.
        let travel = t / max(ray_len, 1e-3);
        let mip = min(i32(floor(travel * f32(max_mip) * 0.85)), max_mip);
        let scene_z = hiz_depth(projected.xy, f32(mip));

        if scene_z >= 0.9999 {
            prev_t = t;
            t += base_step * mix(1.0, 2.5, f32(mip) / max(f32(max_mip), 1.0));
            step_i = step_i + 1u;
            continue;
        }

        let scene_pos = view_pos(projected.xy, scene_z);
        let delta = sample_pos.z - scene_pos.z;
        let thick = max(thickness, base_step * 1.5);

        if delta > 0.0 && delta < thick {
            if t > bias * 1.5 {
                // Binary refine at mip 0 between prev miss and this hit.
                var lo = prev_t;
                var hi = t;
                var best_uv = projected.xy;
                var best_t = t;
                for (var r = 0u; r < 4u; r++) {
                    let mid = mix(lo, hi, 0.5);
                    let mid_pos = origin + dir * mid;
                    let mid_p = project_uv(mid_pos);
                    if mid_p.x < 0.0 || mid_p.x > 1.0 || mid_p.y < 0.0 || mid_p.y > 1.0 {
                        break;
                    }
                    let mid_d = hiz_depth(mid_p.xy, 0.0);
                    let mid_scene = view_pos(mid_p.xy, mid_d);
                    if mid_pos.z - mid_scene.z > 0.0 {
                        hi = mid;
                        best_uv = mid_p.xy;
                        best_t = mid;
                    } else {
                        lo = mid;
                    }
                }
                return vec4(best_uv, best_t, 1.0);
            }
            break;
        }

        var stride = base_step;
        if delta <= 0.0 {
            let gap = scene_pos.z - sample_pos.z;
            stride = clamp(gap * 0.65, base_step * 0.35, base_step * (1.5 + f32(mip) * 0.35));
        } else {
            // Behind without thickness window — thin miss.
            stride = base_step * 0.5;
        }

        if step_i == 0u {
            stride *= mix(0.6, 1.0, jitter);
        }

        prev_t = t;
        t += stride;
        step_i = step_i + 1u;
    }

    return vec4(0.0);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    if depth >= 0.9999 {
        return vec4(0.0, 0.0, 0.0, depth);
    }

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

    let slot = u32(u.params.w) % 8u;
    let angle_rot = f32(slot) * (PI * 0.25);

    let pixel = i.uv * u.full_resolution;
    let cp = noise2(pixel);
    let tbn = basis(n_view);

    let local_luma = local_luma_ref(i.uv);
    let hit_max_luma = max(local_luma * 1.75, 0.7);

    var irradiance = vec3(0.0);

    for (var s = 0u; s < samples; s++) {
        let xi = r2_cp(f32(s), cp);
        let xj = r2_cp(f32(s) + 19.0, cp.yx);

        let phi = 2.0 * PI * xi.x + angle_rot;
        let cos_t = sqrt(clamp(xi.y, 0.001, 1.0));
        let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
        let local = vec3(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
        let ray_dir = normalize(tbn * local);

        let ray_len = radius * mix(0.2, 1.0, xj.x);
        let march = march_gi(pos, ray_dir, ray_len, bias, thickness, max_steps, xj.y);
        if march.w < 0.5 {
            continue;
        }

        let hit_uv = march.xy;
        let hit_dist = march.z;
        if length(hit_uv - i.uv) < 1.5 / max(u.full_resolution.x, 1.0) {
            continue;
        }

        let hit_depth = hiz_depth(hit_uv, 0.0);
        let hit_pos = view_pos(hit_uv, hit_depth);

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
        let recv_cos = clamp(dot(n_view, ray_dir), 0.0, 1.0);
        let w = dist_att * fade * mix(0.35, 1.0, recv_cos);

        irradiance += radiance * w;
    }

    irradiance /= max(f32(samples), 1.0);
    var gi = irradiance * max(u.energy, 0.0);
    gi = suppress_firefly(gi, hit_max_luma * 1.35);
    return vec4(max(gi, vec3(0.0)), depth);
}
