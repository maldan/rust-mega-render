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

struct FxaaUniforms {
    texel: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: FxaaUniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

// Lightweight FXAA (Jimenez 2011 style).
@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let rgb_m = textureSample(src, samp, i.uv).rgb;
    let rgb_nw = textureSample(src, samp, i.uv + vec2(-u.texel.x, -u.texel.y)).rgb;
    let rgb_ne = textureSample(src, samp, i.uv + vec2(u.texel.x, -u.texel.y)).rgb;
    let rgb_sw = textureSample(src, samp, i.uv + vec2(-u.texel.x, u.texel.y)).rgb;
    let rgb_se = textureSample(src, samp, i.uv + vec2(u.texel.x, u.texel.y)).rgb;

    let luma_m = luma(rgb_m);
    let luma_nw = luma(rgb_nw);
    let luma_ne = luma(rgb_ne);
    let luma_sw = luma(rgb_sw);
    let luma_se = luma(rgb_se);

    let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    var dir = vec2(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
        ((luma_nw + luma_sw) - (luma_ne + luma_se)),
    );
    let dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125, 1.0 / 128.0);
    let rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    dir = clamp(dir * rcp_dir_min, vec2(-8.0), vec2(8.0)) * u.texel;

    let rgb_a = 0.5 * (
        textureSample(src, samp, i.uv + dir * (1.0 / 3.0 - 0.5)).rgb
        + textureSample(src, samp, i.uv + dir * (2.0 / 3.0 - 0.5)).rgb
    );
    let rgb_b = rgb_a * 0.5 + 0.25 * (
        textureSample(src, samp, i.uv + dir * -0.5).rgb
        + textureSample(src, samp, i.uv + dir * 0.5).rgb
    );

    let luma_b = luma(rgb_b);
    if luma_b < luma_min || luma_b > luma_max {
        return vec4(rgb_a, 1.0);
    }
    return vec4(rgb_b, 1.0);
}
