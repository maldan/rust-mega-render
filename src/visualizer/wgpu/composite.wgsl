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

struct CompositeUniforms {
    ao_intensity: f32,
    bloom_intensity: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: CompositeUniforms;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var ao_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    var color = textureSample(scene_tex, samp, i.uv).rgb;
    if u.ao_intensity > 0.0 {
        let ao = textureSample(ao_tex, samp, i.uv).r;
        color *= mix(1.0, ao, u.ao_intensity);
    }
    if u.bloom_intensity > 0.0 {
        color += textureSample(bloom_tex, samp, i.uv).rgb * u.bloom_intensity;
    }
    return vec4(color, 1.0);
}
