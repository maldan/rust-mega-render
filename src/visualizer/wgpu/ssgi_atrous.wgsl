//! Edge-aware à-trous denoise for SSGI (SVGF-style 3×3 kernel).
//! Weights by depth + normal + luminance. Preserves alpha (clip depth).
//! Run multiple times with increasing `step_size` (1, 2, 4, 8).

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

struct AtrousUniforms {
    /// 1 / pass resolution
    texel: vec2<f32>,
    step_size: f32,
    depth_sigma: f32,
    normal_sigma: f32,
    luma_sigma: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: AtrousUniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_samp: sampler;
@group(0) @binding(5) var normal_tex: texture_2d<f32>;
@group(0) @binding(6) var normal_samp: sampler;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

fn decode_n(raw: vec4<f32>) -> vec3<f32> {
    // G-buffer stores world-space normals in Rgba16Float (already [-1,1]).
    return normalize(raw.xyz);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let center = textureSampleLevel(src, samp, i.uv, 0.0);
    let center_d = textureSample(depth_tex, depth_samp, i.uv);
    if center_d >= 0.9999 {
        return vec4(0.0, 0.0, 0.0, center_d);
    }

    let center_n = decode_n(textureSample(normal_tex, normal_samp, i.uv));
    let center_y = luma(center.rgb);

    // B3-spline kernel (SVGF / à-trous).
    let kernel = array<f32, 9>(
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
        1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
    );

    var result = vec3(0.0);
    var wsum = 0.0;
    var ki = 0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2(f32(x), f32(y)) * u.texel * u.step_size;
            let suv = i.uv + offset;
            let s = textureSampleLevel(src, samp, suv, 0.0);
            let sd = textureSample(depth_tex, depth_samp, suv);
            let sn = decode_n(textureSample(normal_tex, normal_samp, suv));
            let sy = luma(s.rgb);

            // Depth edge (relative — stable for near/far).
            let dz = abs(center_d - sd)
                / max(max(abs(center_d), abs(sd)), 1e-4);
            let w_z = exp(-dz * u.depth_sigma);

            // Normal edge.
            let nd = max(0.0, dot(center_n, sn));
            let w_n = pow(nd, max(u.normal_sigma, 1.0));

            // Luminance edge — keep bounce color boundaries.
            let w_l = exp(-abs(center_y - sy) / max(u.luma_sigma, 1e-3));

            let w = kernel[ki] * w_z * w_n * w_l;
            result += s.rgb * w;
            wsum += w;
            ki += 1;
        }
    }

    let gi = select(center.rgb, result / wsum, wsum > 1e-5);
    return vec4(gi, center_d);
}
