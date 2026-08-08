struct HudUniforms {
    screen: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: HudUniforms;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // Pixel coords (top-left origin) → NDC.
    let ndc = vec2<f32>(
        in.pos.x / frame.screen.x * 2.0 - 1.0,
        1.0 - in.pos.y / frame.screen.y * 2.0,
    );
    var out: VsOut;
    out.clip = vec4(ndc, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let a = textureSample(atlas, atlas_samp, in.uv).r;
    return vec4(in.color.rgb, in.color.a * a);
}
