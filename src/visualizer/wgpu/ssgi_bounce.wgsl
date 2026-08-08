//! Cheap screen-space 2nd bounce for SSGI.
//! Gathers denoised first-bounce irradiance with depth/normal weights
//! and adds `strength * gathered` (still irradiance — albedo applied in composite).

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

struct BounceUniforms {
    /// 1 / half-res
    texel: vec2<f32>,
    strength: f32,
    depth_sigma: f32,
}

@group(0) @binding(0) var<uniform> u: BounceUniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_samp: sampler;
@group(0) @binding(5) var normal_tex: texture_2d<f32>;
@group(0) @binding(6) var normal_samp: sampler;

fn decode_n(raw: vec4<f32>) -> vec3<f32> {
    return normalize(raw.xyz);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let center = textureSampleLevel(src, samp, i.uv, 0.0);
    let center_d = textureSample(depth_tex, depth_samp, i.uv);
    if center_d >= 0.9999 || u.strength < 1e-4 {
        return center;
    }

    let center_n = decode_n(textureSample(normal_tex, normal_samp, i.uv));

    // Spiral offsets in texels — wider than à-trous for soft color bleed.
    let offs = array<vec2<f32>, 8>(
        vec2( 1.0,  0.0),
        vec2(-1.0,  0.0),
        vec2( 0.0,  1.0),
        vec2( 0.0, -1.0),
        vec2( 1.5,  1.5),
        vec2(-1.5,  1.5),
        vec2( 1.5, -1.5),
        vec2(-1.5, -1.5),
    );

    var gathered = vec3(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 8; k++) {
        let suv = i.uv + offs[k] * u.texel * 3.0;
        let s = textureSampleLevel(src, samp, suv, 0.0);
        let sd = textureSample(depth_tex, depth_samp, suv);
        let sn = decode_n(textureSample(normal_tex, normal_samp, suv));

        let dz = abs(center_d - sd) / max(max(abs(center_d), abs(sd)), 1e-4);
        let w_z = exp(-dz * u.depth_sigma);
        let w_n = pow(max(dot(center_n, sn), 0.0), 16.0);
        let w = w_z * w_n;
        gathered += s.rgb * w;
        wsum += w;
    }

    let bounce = select(vec3(0.0), gathered / wsum, wsum > 1e-4);
    let gi = center.rgb + bounce * u.strength;
    return vec4(max(gi, vec3(0.0)), center_d);
}
