//! Spatial screen-space global illumination.
//! LH view (+Z forward), DirectX clip depth.
//!
//! Marches short hemisphere rays, gathers lit HDR radiance as irradiance,
//! then applies receiver `albedo * (1 - metallic) / π`.
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

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2(hash21(p), hash21(p + vec2(13.1, 7.7)));
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

/// Kill HDR fireflies from specular / sun / lamps in the lit buffer.
fn suppress_firefly(c: vec3<f32>, max_luma: f32) -> vec3<f32> {
    let y = luma(c);
    if y <= max_luma {
        return c;
    }
    return c * (max_luma / y);
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

    // Slow pattern cycle (8 slots) — fast golden-phase made bright hits "fly" upward.
    let slot = u32(u.params.w) % 8u;
    let phase = f32(slot) / 8.0;

    let pixel = i.uv * u.full_resolution;
    let n0 = ign(pixel + vec2(phase * 37.1, phase * 73.7));
    let n1 = hash21(pixel + vec2(19.19 + phase * 11.0, 7.7));
    let tbn = basis(n_view);

    // Local luminance reference — clamp hits relative to what's already on screen.
    let local_luma = max(luma(textureSampleLevel(color_tex, color_samp, i.uv, 0.0).rgb), 0.05);
    let hit_max_luma = max(local_luma * 2.5, 1.0);

    var irradiance = vec3(0.0);

    for (var s = 0u; s < samples; s++) {
        let xi = hash22(pixel + vec2(f32(s) * 17.3 + phase * 64.0, n0 * 64.0));
        let phi = 2.0 * PI * fract(xi.x + n1);
        let cos_t = sqrt(xi.y);
        let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
        let local = vec3(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
        let ray_dir = normalize(tbn * local);

        let ray_len = radius * (0.35 + 0.65 * fract(xi.y + n0));
        let step_len = ray_len / f32(max_steps);
        let jitter = ign(pixel + vec2(f32(s) + phase * 9.0, 3.1)) * step_len;

        var hit = false;
        var hit_uv = i.uv;
        var hit_dist = ray_len;

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
                break;
            }
            if delta > thick {
                break;
            }
        }

        if !hit {
            continue;
        }
        if length(hit_uv - i.uv) < 1.5 / max(u.full_resolution.x, 1.0) {
            continue;
        }

        var radiance = textureSampleLevel(color_tex, color_samp, hit_uv, 0.0).rgb;
        radiance = suppress_firefly(radiance, hit_max_luma);
        let dist_att = exp(-hit_dist / max(radius, 0.05));
        let fade = edge_fade(hit_uv);
        irradiance += radiance * dist_att * fade;
    }

    irradiance /= max(f32(samples), 1.0);
    let boost = 3.5;
    var diffuse_gi = irradiance * albedo * kd * boost;
    // Final clamp so a single surviving hit can't sparkle after blur.
    diffuse_gi = suppress_firefly(diffuse_gi, hit_max_luma * 1.5);
    return vec4(max(diffuse_gi, vec3(0.0)), depth);
}
