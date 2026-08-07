//! Screen-space reflections + equirect env fallback (confidence blend).
//! LH view (+Z forward), DirectX clip depth (0=near).
//!
//! Hi-Z hierarchical march (min-depth mips) + env fallback.
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

struct SsrUniforms {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    camera_pos: vec4<f32>,
    resolution: vec2<f32>,
    max_distance: f32,
    thickness: f32,
    /// x = max_steps, y = bias, z = roughness_cutoff, w = frame index
    params: vec4<f32>,
    /// x = env intensity, y = blur levels, z = env on, w = env yaw (rad)
    env: vec4<f32>,
    /// x = hi-z max mip, yzw unused
    hiz: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: SsrUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_samp: sampler;
@group(0) @binding(5) var color_tex: texture_2d<f32>;
@group(0) @binding(6) var color_samp: sampler;
@group(0) @binding(7) var orm_tex: texture_2d<f32>;
@group(0) @binding(8) var orm_samp: sampler;
@group(0) @binding(9) var albedo_tex: texture_2d<f32>;
@group(0) @binding(10) var albedo_samp: sampler;
@group(0) @binding(11) var env_sharp: texture_2d<f32>;
@group(0) @binding(12) var env_blur: texture_2d_array<f32>;
@group(0) @binding(13) var env_samp: sampler;
@group(0) @binding(14) var hiz_tex: texture_2d<f32>;
@group(0) @binding(15) var hiz_samp: sampler;

const PI: f32 = 3.14159265;

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

fn ign(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn rotate_y(d: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3(d.x * c + d.z * s, d.y, -d.x * s + d.z * c);
}

fn dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = rotate_y(normalize(dir), u.env.w);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, theta / PI);
}

fn sample_env_level(uv: vec2<f32>, idx: i32) -> vec3<f32> {
    if idx <= 0 {
        return textureSampleLevel(env_sharp, env_samp, uv, 0.0).rgb;
    }
    let layers = max(i32(u.env.y), 1);
    let layer = u32(clamp(idx - 1, 0, layers - 1));
    return textureSampleLevel(env_blur, env_samp, uv, layer, 0.0).rgb;
}

fn sample_env(dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    if u.env.z < 0.5 {
        return vec3(0.0);
    }
    let uv = dir_to_uv(dir);
    let levels = max(u.env.y, 1.0);
    let t = clamp(roughness, 0.0, 1.0) * levels;
    let i0 = i32(floor(t));
    let i1 = min(i0 + 1, i32(levels));
    let f = fract(t);
    return mix(sample_env_level(uv, i0), sample_env_level(uv, i1), f) * u.env.x;
}

fn edge_fade(uv: vec2<f32>) -> f32 {
    let e = smoothstep(vec2(0.0), vec2(0.08), uv) * smoothstep(vec2(0.0), vec2(0.08), 1.0 - uv);
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
    return c * (max_luma / y);
}

fn hiz_depth(uv: vec2<f32>, level: f32) -> f32 {
    return textureSampleLevel(hiz_tex, hiz_samp, uv, level).r;
}

/// Perspective-correct view-space march.
/// Hi-Z only chooses stride (empty-space skip) — never interpolates UV linearly
/// (that was stretching spheres into capsules).
fn march_ssr(origin: vec3<f32>, dir: vec3<f32>, roughness: f32) -> vec4<f32> {
    let max_steps = u32(clamp(u.params.x, 8.0, 64.0));
    let bias = max(u.params.y, 0.0);
    let max_dist = max(u.max_distance, 0.1);
    let thickness = max(u.thickness, 0.001) * mix(1.0, 1.5, roughness);
    let max_mip = i32(clamp(u.hiz.x, 0.0, 8.0));
    let slot = u32(u.params.w) % 8u;
    let jitter = fract(ign(origin.xy * u.resolution + origin.zz) + f32(slot) * 0.125);

    var hit_uv = vec2(0.0);
    var conf = 0.0;
    var t = bias;
    var prev_t = bias;
    var step_i = 0u;

    // Base step size; Hi-Z may stretch it when clearly in empty space.
    let base_step = max_dist / f32(max_steps);

    loop {
        if step_i >= max_steps || t > max_dist {
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

        // Adaptive mip: coarser farther from the receiver.
        let travel = t / max_dist;
        let mip = min(i32(floor(travel * f32(max_mip) * 0.85)), max_mip);
        let scene_z = hiz_depth(projected.xy, f32(mip));

        if scene_z >= 0.9999 {
            // Sky / empty — stride farther.
            prev_t = t;
            t += base_step * mix(1.0, 2.5, f32(mip) / max(f32(max_mip), 1.0));
            step_i = step_i + 1u;
            continue;
        }

        // Compare in view space (perspective-correct).
        let scene_pos = view_pos(projected.xy, scene_z);
        let delta = sample_pos.z - scene_pos.z;

        if delta > 0.0 && delta < thickness {
            if t > bias * 2.0 {
                // Binary refine between previous miss and this hit (always mip 0).
                var lo = prev_t;
                var hi = t;
                var best_uv = projected.xy;
                for (var r = 0u; r < 5u; r++) {
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
                    } else {
                        lo = mid;
                    }
                }
                hit_uv = best_uv;
                let dist_fade = 1.0 - smoothstep(max_dist * 0.55, max_dist, hi);
                conf = edge_fade(best_uv) * dist_fade;
            }
            break;
        }

        // Still in front of geometry: larger step if Hi-Z cell is clearly farther.
        var stride = base_step;
        if delta <= 0.0 {
            let gap = scene_pos.z - sample_pos.z;
            // Don't overshoot the surface by more than ~half a step toward it.
            stride = clamp(gap * 0.65, base_step * 0.35, base_step * (1.5 + f32(mip) * 0.35));
        } else {
            // Went behind without thickness window — thin miss; nudge forward.
            stride = base_step * 0.5;
        }

        // Jitter first steps so temporal blends cleanly.
        if step_i == 0u {
            stride *= mix(0.6, 1.0, jitter);
        }

        prev_t = t;
        t += stride;
        step_i = step_i + 1u;
    }

    return vec4(hit_uv, 0.0, conf);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    if depth >= 0.9999 {
        return vec4(0.0, 0.0, 0.0, depth);
    }

    let orm = textureSampleLevel(orm_tex, orm_samp, i.uv, 0.0);
    let roughness = max(orm.g, 0.04);
    let metallic = orm.b;
    let albedo = textureSampleLevel(albedo_tex, albedo_samp, i.uv, 0.0).rgb;

    let gloss = 1.0 - roughness;
    let energy = gloss * gloss;
    if energy < 0.002 {
        return vec4(0.0, 0.0, 0.0, depth);
    }

    let pos = view_pos(i.uv, depth);
    var n_world = textureSample(normal_tex, normal_samp, i.uv).xyz;
    if dot(n_world, n_world) < 1e-6 {
        return vec4(0.0, 0.0, 0.0, depth);
    }
    n_world = normalize(n_world);
    var n_view = normalize((u.view * vec4(n_world, 0.0)).xyz);
    let to_cam = normalize(-pos);
    if dot(n_view, to_cam) < 0.0 {
        n_view = -n_view;
        n_world = -n_world;
    }

    let f0 = mix(vec3(0.04), albedo, metallic);
    let n_dot_v = max(dot(n_view, to_cam), 0.001);
    let f = fresnel_schlick(n_dot_v, f0);

    let r_view = reflect(-to_cam, n_view);
    if dot(r_view, n_view) <= 0.0 {
        return vec4(0.0, 0.0, 0.0, depth);
    }
    let r_world = normalize((u.inv_view * vec4(r_view, 0.0)).xyz);

    let env_col = sample_env(r_world, roughness);
    var ssr_col = env_col;
    var conf = 0.0;

    let cutoff = clamp(u.params.z, 0.05, 1.0);
    if roughness <= cutoff {
        // Jitter direction slightly for temporal (glossy) / firefly break-up.
        let slot = u32(u.params.w) % 8u;
        let j = ign(i.uv * u.resolution + f32(slot));
        var dir = normalize(r_view);
        if roughness > 0.08 {
            let tang = normalize(cross(dir, n_view + vec3(0.0001, 0.0, 0.0)));
            let bit = cross(dir, tang);
            let cone = roughness * roughness * 0.15;
            dir = normalize(dir + (tang * (j * 2.0 - 1.0) + bit * (fract(j * 17.0) * 2.0 - 1.0)) * cone);
        }

        let march = march_ssr(pos, dir, roughness);
        conf = march.w;
        if conf > 0.001 {
            let blur_px = roughness * roughness * 6.0;
            var hit = textureSampleLevel(color_tex, color_samp, march.xy, 0.0).rgb;
            if blur_px > 0.35 {
                let texel = 1.0 / u.resolution;
                let o = texel * blur_px;
                hit = hit * 0.4;
                hit += textureSampleLevel(color_tex, color_samp, march.xy + vec2(o.x, 0.0), 0.0).rgb * 0.15;
                hit += textureSampleLevel(color_tex, color_samp, march.xy - vec2(o.x, 0.0), 0.0).rgb * 0.15;
                hit += textureSampleLevel(color_tex, color_samp, march.xy + vec2(0.0, o.y), 0.0).rgb * 0.15;
                hit += textureSampleLevel(color_tex, color_samp, march.xy - vec2(0.0, o.y), 0.0).rgb * 0.15;
            }
            hit = suppress_firefly(hit, 8.0);
            let rough_w = 1.0 - smoothstep(cutoff * 0.65, cutoff, roughness);
            conf *= rough_w * edge_fade(i.uv);
            ssr_col = mix(env_col, hit, clamp(conf, 0.0, 1.0));
        }
    }

    let specular = ssr_col * f * energy;
    return vec4(specular, depth);
}
