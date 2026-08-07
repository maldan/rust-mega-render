//! Temporal resolve for SSR — reprojection + depth reject + AABB neighborhood clamp.

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
    /// x = history weight, y = depth reject, z = has_history, w = unused
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: TemporalUniforms;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var depth_samp: sampler;

fn world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = u.inv_view_proj * clip;
    return w.xyz / max(w.w, 1e-6);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    let current = textureSampleLevel(current_tex, samp, i.uv, 0.0);

    if depth >= 0.9999 {
        return vec4(0.0, 0.0, 0.0, depth);
    }

    let cur = current.rgb;
    let out_depth = depth;

    if u.params.z < 0.5 {
        return vec4(cur, out_depth);
    }

    let wp = world_pos(i.uv, depth);
    let prev_clip = u.prev_view_proj * vec4(wp, 1.0);
    let prev_ndc = prev_clip.xyz / max(prev_clip.w, 1e-6);
    let prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    if prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
        return vec4(cur, out_depth);
    }

    var hist = textureSampleLevel(history_tex, samp, prev_uv, 0.0);
    let hist_depth = hist.a;
    let expected_depth = prev_ndc.z;

    let depth_err = abs(hist_depth - expected_depth)
        / max(max(abs(expected_depth), abs(hist_depth)), 1e-4);
    let reject = u.params.y;
    if depth_err > reject || hist_depth >= 0.9999 {
        return vec4(cur, out_depth);
    }

    // 3×3 neighborhood AABB clamp (kills ghosting / disocclusion streaks).
    let dims = vec2<f32>(textureDimensions(current_tex, 0));
    let texel = 1.0 / dims;
    var n_min = cur;
    var n_max = cur;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let n = textureSampleLevel(
                current_tex,
                samp,
                i.uv + vec2(f32(x), f32(y)) * texel,
                0.0,
            ).rgb;
            n_min = min(n_min, n);
            n_max = max(n_max, n);
        }
    }
    // Slightly expand box so stable mirrors don't flicker.
    let pad = (n_max - n_min) * 0.15 + vec3(0.002);
    n_min -= pad;
    n_max += pad;
    hist = vec4(clamp(hist.rgb, n_min, n_max), hist.a);

    let conf = 1.0 - smoothstep(reject * 0.5, reject, depth_err);
    var w_hist = clamp(u.params.x, 0.0, 0.98) * conf;

    let cur_y = dot(cur, vec3(0.2126, 0.7152, 0.0722));
    let hist_y = dot(hist.rgb, vec3(0.2126, 0.7152, 0.0722));
    if cur_y > hist_y * 3.0 + 0.35 {
        w_hist = max(w_hist, 0.85);
    }

    let rgb = mix(cur, hist.rgb, w_hist);
    return vec4(rgb, out_depth);
}
