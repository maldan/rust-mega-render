//! Depth-aware upsample: half-res SSGI → full-res (kills blocky / stripe artifacts).

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

struct UpsampleUniforms {
    /// 1 / half-resolution
    half_texel: vec2<f32>,
    depth_sigma: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> u: UpsampleUniforms;
@group(0) @binding(1) var ssgi_half: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_samp: sampler;

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let d = textureSample(depth_tex, depth_samp, i.uv);
    if d >= 0.9999 {
        return vec4(0.0, 0.0, 0.0, d);
    }

    // 2×2 taps around the corresponding half-res footprint, weighted by depth.
    var result = vec3(0.0);
    var wsum = 0.0;
    let offsets = array<vec2<f32>, 4>(
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(-0.5, 0.5),
        vec2(0.5, 0.5),
    );
    for (var k = 0; k < 4; k++) {
        let suv = i.uv + offsets[k] * u.half_texel;
        let s = textureSampleLevel(ssgi_half, samp, suv, 0.0);
        let w = exp(-abs(d - s.a) * u.depth_sigma) + 1e-3;
        result += s.rgb * w;
        wsum += w;
    }
    return vec4(result / wsum, d);
}
