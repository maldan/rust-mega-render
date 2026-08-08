//! Per-pixel motion blur — local velocity gather.
//!
//! Depth test is asymmetric: only suppress closer (foreground) leakage.
//! Strict relative depth reject kills camera-pan blur across the scene.

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

struct GatherUniforms {
    // xy = resolution, z = intensity * shutter_scale, w = max_blur_px
    params: vec4<f32>,
    // x = samples, y = depth_sigma, z = frame
    gather: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: GatherUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_samp: sampler;
@group(0) @binding(3) var vel_tex: texture_2d<f32>;
@group(0) @binding(4) var dilate_tex: texture_2d<f32>;
@group(0) @binding(5) var vel_samp: sampler;
@group(0) @binding(6) var depth_tex: texture_depth_2d;
@group(0) @binding(7) var depth_samp: sampler;

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let center = textureSampleLevel(color_tex, color_samp, i.uv, 0.0);
    let v_px = textureSampleLevel(vel_tex, vel_samp, i.uv, 0.0).xy;
    let d_px = textureSampleLevel(dilate_tex, vel_samp, i.uv, 0.0).xy;

    let intensity = max(u.params.z, 0.0);
    let max_blur = max(u.params.w, 1.0);
    let res = max(u.params.xy, vec2(1.0));

    let v_len = length(v_px);
    let d_len = length(d_px);

    // Prefer local motion; fall back to dilate on silhouettes.
    var blur = v_px;
    if v_len < 0.25 {
        if d_len < 1.0 {
            return center;
        }
        blur = d_px * 0.5;
    } else if d_len > v_len * 1.05 {
        let agree = dot(v_px, d_px) / (v_len * d_len);
        if agree > 0.5 {
            // Modest silhouette extension.
            blur = normalize(v_px) * mix(v_len, d_len, 0.35);
        }
    }

    blur *= intensity;
    var blur_len = length(blur);
    if blur_len < 0.35 {
        return center;
    }
    if blur_len > max_blur {
        blur *= max_blur / blur_len;
        blur_len = max_blur;
    }

    let samples = i32(clamp(u.gather.x, 4.0, 24.0));
    // Closer-surface reject width in clip-depth units (not relative!).
    let closer_sigma = max(u.gather.y, 1e-4);
    let jitter = ign(i.uv * res + vec2(u.gather.z)) * 0.99;
    let center_d = textureSample(depth_tex, depth_samp, i.uv);
    let dir_uv = blur / res;
    let center_y = max(luma(center.rgb), 1e-3);

    // Equal-ish weights along the streak (still a bit more at center).
    var acc = center.rgb * 1.25;
    var wsum = 1.25;

    for (var s = 1; s < samples; s++) {
        // jitted step in (0,1]
        let t = (f32(s) - 0.5 + jitter) / f32(samples);
        let offs = array<f32, 2>(t, -t);
        for (var k = 0; k < 2; k++) {
            let o = offs[k];
            let uv = i.uv + dir_uv * o;
            if (uv.x < 0.0) || (uv.x > 1.0) || (uv.y < 0.0) || (uv.y > 1.0) {
                continue;
            }

            let sample_d = textureSample(depth_tex, depth_samp, uv);
            // Allow any farther/equal depth. Only soften when sample is closer.
            var w = 1.0;
            let closer = center_d - sample_d; // >0 ⇒ sample closer (DX: smaller z = nearer)
            if closer > 0.0 {
                w = exp(-(closer * closer) / (closer_sigma * closer_sigma));
            }
            // Mild falloff toward streak ends.
            w *= mix(0.35, 1.0, 1.0 - abs(o));
            if w < 0.02 {
                continue;
            }

            var c = textureSampleLevel(color_tex, color_samp, uv, 0.0).rgb;
            let cy = luma(c);
            if cy > center_y * 6.0 {
                c *= (center_y * 6.0) / cy;
            }

            acc += c * w;
            wsum += w;
        }
    }

    return vec4(acc / max(wsum, 1e-4), center.a);
}
