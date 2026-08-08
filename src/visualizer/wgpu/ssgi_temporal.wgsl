//! Temporal resolve for SSGI — velocity reprojection + soft depth/normal reject
//! + neighborhood variance clamp (AABB).
//! LH view (+Z forward), DirectX clip depth.
//!
//! History RGB = accumulated GI, A = clip depth of that sample.
//! Velocity is screen-space motion in pixels (curr_uv - prev_uv) * resolution.

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

struct TemporalUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    /// x = history weight, y = depth reject (relative), z = has_history,
    /// w = normal reject (1 - min_dot), e.g. 0.15 → reject if dot < 0.85
    params: vec4<f32>,
    /// Full framebuffer size (velocity is in pixels).
    resolution: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: TemporalUniforms;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
/// Nearest — linear history filtering smears into streaks/lines.
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var depth_samp: sampler;
@group(0) @binding(6) var velocity_tex: texture_2d<f32>;
@group(0) @binding(7) var velocity_samp: sampler;
@group(0) @binding(8) var normal_tex: texture_2d<f32>;
@group(0) @binding(9) var normal_samp: sampler;

fn world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = u.inv_view_proj * clip;
    return w.xyz / max(w.w, 1e-6);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

fn decode_n(raw: vec4<f32>) -> vec3<f32> {
    return normalize(raw.xyz);
}

/// Clamp history into a soft neighborhood box (mean ± γ·σ) of the current 3×3.
fn variance_clamp(hist: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(current_tex));
    let texel = 1.0 / max(dims, vec2(1.0));

    var m1 = vec3(0.0);
    var m2 = vec3(0.0);
    var n_min = vec3(1e10);
    var n_max = vec3(-1e10);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let suv = uv + vec2(f32(x), f32(y)) * texel;
            let s = textureSampleLevel(current_tex, samp, suv, 0.0).rgb;
            m1 += s;
            m2 += s * s;
            n_min = min(n_min, s);
            n_max = max(n_max, s);
        }
    }

    let inv_n = 1.0 / 9.0;
    let mu = m1 * inv_n;
    let var_ = max(m2 * inv_n - mu * mu, vec3(0.0));
    let sigma = sqrt(var_);

    let gamma = 1.25;
    var box_min = max(mu - sigma * gamma, n_min);
    var box_max = min(mu + sigma * gamma, n_max);
    let pad = (box_max - box_min) * 0.05 + vec3(0.002);
    box_min -= pad;
    box_max += pad;

    return clamp(hist, box_min, box_max);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    let current = textureSampleLevel(current_tex, samp, i.uv, 0.0);

    if depth >= 0.9999 {
        return vec4(0.0, 0.0, 0.0, depth);
    }

    let cur_gi = current.rgb;
    let out_depth = depth;

    if u.params.z < 0.5 {
        return vec4(cur_gi, out_depth);
    }

    let res = max(u.resolution, vec2(1.0));
    let vel_px = textureSampleLevel(velocity_tex, velocity_samp, i.uv, 0.0).xy;
    let vel_uv = vel_px / res;

    // Camera geometric reprojection (depth validation + fallback UV).
    let wp = world_pos(i.uv, depth);
    let prev_clip = u.prev_view_proj * vec4(wp, 1.0);
    let prev_ndc = prev_clip.xyz / max(prev_clip.w, 1e-6);
    let cam_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    let expected_depth = prev_ndc.z;

    // Prefer velocity (handles moving objects); fall back to camera if tiny motion.
    let vel_mag = length(vel_px);
    var prev_uv = i.uv - vel_uv;
    if vel_mag < 0.05 {
        prev_uv = cam_uv;
    }

    if prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
        return vec4(cur_gi, out_depth);
    }

    let hist = textureSampleLevel(history_tex, samp, prev_uv, 0.0);
    let hist_depth = hist.a;
    if hist_depth >= 0.9999 {
        return vec4(cur_gi, out_depth);
    }

    // Soft relative clip-depth rejection (no hard cut).
    let depth_err = abs(hist_depth - expected_depth)
        / max(max(abs(expected_depth), abs(hist_depth)), 1e-4);
    let reject = max(u.params.y, 0.001);
    let depth_conf = 1.0 - smoothstep(reject * 0.35, reject, depth_err);
    if depth_conf < 0.05 {
        return vec4(cur_gi, out_depth);
    }

    // Soft normal reject: compare current normal vs normal at reprojected UV
    // (same-frame buffer — catches silhouette / disocclusion edges).
    let n_cur = decode_n(textureSampleLevel(normal_tex, normal_samp, i.uv, 0.0));
    let n_hist = decode_n(textureSampleLevel(normal_tex, normal_samp, prev_uv, 0.0));
    let n_dot = clamp(dot(n_cur, n_hist), 0.0, 1.0);
    let n_reject = clamp(u.params.w, 0.0, 1.0);
    let min_dot = 1.0 - n_reject;
    let normal_conf = smoothstep(min_dot - 0.1, min_dot + 0.05, n_dot);

    // Motion-adaptive history: fast movers trust current more.
    let motion_conf = exp(-vel_mag * 0.035);

    var w_hist = clamp(u.params.x, 0.0, 0.98)
        * depth_conf
        * normal_conf
        * mix(0.35, 1.0, motion_conf);

    var hist_gi = variance_clamp(hist.rgb, i.uv);

    // Firefly gate: if current is much brighter than clamped history, trust history.
    let cur_y = luma(cur_gi);
    let hist_y = luma(hist_gi);
    if cur_y > hist_y * 3.0 + 0.35 {
        w_hist = max(w_hist, 0.85 * depth_conf);
    }

    let gi = mix(cur_gi, hist_gi, w_hist);
    return vec4(gi, out_depth);
}
