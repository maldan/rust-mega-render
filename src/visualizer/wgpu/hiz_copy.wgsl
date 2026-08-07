//! Copy hardware depth → R32Float Hi-Z mip 0.

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

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var depth_samp: sampler;

@fragment
fn fs(i: VsOut) -> @location(0) f32 {
    return textureSample(depth_tex, depth_samp, i.uv);
}
