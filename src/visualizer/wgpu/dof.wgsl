//! Quality Temporal DOF gather — dual-field + n-gon bokeh.
//! LH view (+Z forward), DirectX clip depth.
//!
//! Sky / cubemap (clip depth ≈ 1) is treated as infinity and gets far-field CoC,
//! so the env background blurs with the rest of the scene.
//!
//! `fs`     — near/far field gather → composited HDR (A = clip depth)
//! `fs_coc` — CoC debug (magenta near / green far)

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

struct DofUniforms {
    inv_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    focus_distance: f32,
    aperture: f32,
    max_coc: f32,
    samples: f32,
    frame: f32,
    focus_range: f32,
    /// 0 = circle, 5..=8 = blades
    bokeh_blades: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: DofUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_samp: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_samp: sampler;

const GOLDEN_ANGLE: f32 = 2.39996323;
const PI: f32 = 3.14159265;
const SKY_Z: f32 = 1.0e4;

fn is_sky(depth: f32) -> bool {
    return depth >= 0.9999;
}

fn view_z(uv: vec2<f32>, depth: f32) -> f32 {
    if is_sky(depth) {
        return SKY_Z;
    }
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = u.inv_proj * clip;
    return v.z / max(v.w, 1e-6);
}

/// Signed CoC in pixels. <0 near, >0 far. Sky → far CoC at infinity.
fn signed_coc(depth_z: f32) -> f32 {
    let z = max(depth_z, 0.05);
    let dz = abs(z - u.focus_distance) - max(u.focus_range, 0.0);
    let mag = max(dz, 0.0) * u.aperture / z;
    let sign_v = select(-1.0, 1.0, z >= u.focus_distance);
    return clamp(sign_v * mag, -u.max_coc, u.max_coc);
}

fn ign(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

fn bokeh_scale(angle: f32) -> f32 {
    let n = u.bokeh_blades;
    if n < 3.5 {
        return 1.0;
    }
    let sector = 2.0 * PI / n;
    let a = angle + PI / n;
    return cos(PI / n) / max(cos(a - sector * floor(a / sector)), 0.2);
}

fn tap_far_weight(center_coc: f32, sample_coc: f32, dist_px: f32) -> f32 {
    let s = abs(sample_coc);
    // Sample contributes if its blur disc covers this pixel.
    var w = smoothstep(dist_px - 0.5, dist_px + 1.5, s);
    // Also allow the center's own kernel to pull in background (incl. sky).
    w = max(w, smoothstep(dist_px - 0.5, dist_px + 1.5, abs(center_coc)) * 0.65);
    if sample_coc < -0.5 {
        w *= 0.02;
    }
    if center_coc < -0.25 {
        return 0.0;
    }
    return w;
}

fn tap_near_weight(center_coc: f32, sample_coc: f32, dist_px: f32) -> f32 {
    let s = abs(sample_coc);
    var w = smoothstep(dist_px - 0.5, dist_px + 1.5, s);
    if sample_coc > 0.5 {
        w *= 0.25;
    }
    if center_coc > 0.5 {
        w *= 0.04;
    }
    return w;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    let center = textureSampleLevel(color_tex, color_samp, i.uv, 0.0);
    let sky = is_sky(depth);

    let z = view_z(i.uv, depth);
    let coc = signed_coc(z);
    let coc_abs = abs(coc);
    if coc_abs < 0.35 {
        return vec4(center.rgb, depth);
    }

    let texel = 1.0 / u.resolution;
    let n_samples = i32(clamp(u.samples, 4.0, 24.0));
    let pixel = i.uv * u.resolution;
    let rot = ign(pixel + vec2(u.frame, u.frame * 0.37)) * (2.0 * PI);

    var far_acc = center.rgb;
    var far_w = 1.0;
    var near_acc = vec3(0.0);
    var near_w = 0.0;

    for (var s_i = 0; s_i < n_samples; s_i++) {
        let fi = f32(s_i) + 0.5;
        let a = fi * GOLDEN_ANGLE + rot;
        let r = sqrt(fi / f32(n_samples)) * coc_abs * bokeh_scale(a);
        let offset = vec2(cos(a), sin(a)) * r * texel;
        let uv_s = clamp(i.uv + offset, vec2(0.001), vec2(0.999));

        let dist_px = length(offset / texel);
        let d_s = textureSample(depth_tex, depth_samp, uv_s);
        let c_s = textureSampleLevel(color_tex, color_samp, uv_s, 0.0).rgb;
        let z_s = view_z(uv_s, d_s);
        let coc_s = signed_coc(z_s);

        let wf = tap_far_weight(coc, coc_s, dist_px);
        if wf > 1e-4 {
            far_acc += c_s * wf;
            far_w += wf;
        }

        // Near field skip on pure sky centers — nothing in front at infinity.
        if !sky {
            let wn = tap_near_weight(coc, coc_s, dist_px);
            if wn > 1e-4 {
                near_acc += c_s * wn;
                near_w += wn;
            }
        }
    }

    let far_col = far_acc / max(far_w, 1e-3);
    var result = select(center.rgb, far_col, coc > 0.3);

    if near_w > 0.05 {
        let near_col = near_acc / near_w;
        let near_a = clamp((-min(coc, 0.0)) / max(u.max_coc, 1.0), 0.0, 1.0);
        let fill = clamp(near_w / (near_w + 1.5), 0.0, 1.0);
        let alpha = clamp(near_a * max(fill, near_a * 0.85), 0.0, 1.0);
        result = mix(result, near_col, alpha);
    }

    return vec4(result, depth);
}

@fragment
fn fs_coc(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    let z = view_z(i.uv, depth);
    let coc = signed_coc(z);
    let near_v = clamp(-coc / max(u.max_coc, 1.0), 0.0, 1.0);
    let far_v = clamp(coc / max(u.max_coc, 1.0), 0.0, 1.0);
    return vec4(near_v, far_v, near_v * 0.35, 1.0);
}
