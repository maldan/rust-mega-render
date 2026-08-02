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

struct BloomUniforms {
    texel: vec2<f32>,
    threshold: f32,
    intensity: f32, // unused in extract/down; filter strength spare
}

@group(0) @binding(0) var<uniform> u: BloomUniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_extract(i: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src, samp, i.uv).rgb;
    let bright = max(luma(c) - u.threshold, 0.0);
    let contrib = bright / (bright + 1.0);
    return vec4(c * contrib, 1.0);
}

@fragment
fn fs_downsample(i: VsOut) -> @location(0) vec4<f32> {
    // 4-tap box downsample.
    let t = u.texel;
    let a = textureSample(src, samp, i.uv + vec2(-t.x, -t.y)).rgb;
    let b = textureSample(src, samp, i.uv + vec2(t.x, -t.y)).rgb;
    let c = textureSample(src, samp, i.uv + vec2(-t.x, t.y)).rgb;
    let d = textureSample(src, samp, i.uv + vec2(t.x, t.y)).rgb;
    return vec4(0.25 * (a + b + c + d), 1.0);
}

@fragment
fn fs_upsample(i: VsOut) -> @location(0) vec4<f32> {
    // 9-tap tent upsample; additive blend set in pipeline.
    let t = u.texel;
    var r = textureSample(src, samp, i.uv).rgb * 4.0;
    r += textureSample(src, samp, i.uv + vec2(-t.x, 0.0)).rgb * 2.0;
    r += textureSample(src, samp, i.uv + vec2(t.x, 0.0)).rgb * 2.0;
    r += textureSample(src, samp, i.uv + vec2(0.0, -t.y)).rgb * 2.0;
    r += textureSample(src, samp, i.uv + vec2(0.0, t.y)).rgb * 2.0;
    r += textureSample(src, samp, i.uv + vec2(-t.x, -t.y)).rgb;
    r += textureSample(src, samp, i.uv + vec2(t.x, -t.y)).rgb;
    r += textureSample(src, samp, i.uv + vec2(-t.x, t.y)).rgb;
    r += textureSample(src, samp, i.uv + vec2(t.x, t.y)).rgb;
    return vec4(r / 16.0, 1.0);
}
