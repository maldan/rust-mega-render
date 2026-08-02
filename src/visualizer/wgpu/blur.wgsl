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

struct BlurUniforms {
    // xy = texel size * direction (1,0) or (0,1)
    direction: vec2<f32>,
    depth_sigma: f32,
    use_depth: f32,
}

@group(0) @binding(0) var<uniform> u: BlurUniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_samp: sampler;

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let center = textureSample(src, samp, i.uv);
    let center_d = textureSample(depth_tex, depth_samp, i.uv);
    var result = center.rgb * 0.227027;
    var weight_sum = 0.227027;

    let offsets = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
    let weights = array<f32, 4>(0.1945946, 0.1216216, 0.054054, 0.016216);

    for (var k = 0; k < 4; k++) {
        let offset = u.direction * offsets[k];
        let uv0 = i.uv + offset;
        let uv1 = i.uv - offset;
        var w0 = weights[k];
        var w1 = weights[k];
        if u.use_depth > 0.5 {
            let d0 = textureSample(depth_tex, depth_samp, uv0);
            let d1 = textureSample(depth_tex, depth_samp, uv1);
            w0 *= exp(-abs(center_d - d0) * u.depth_sigma);
            w1 *= exp(-abs(center_d - d1) * u.depth_sigma);
        }
        result += textureSample(src, samp, uv0).rgb * w0;
        result += textureSample(src, samp, uv1).rgb * w1;
        weight_sum += w0 + w1;
    }
    return vec4(result / weight_sum, center.a);
}
