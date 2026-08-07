//! Build next Hi-Z mip: min of 2×2 from previous level (closest clip depth).

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

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs(i: VsOut) -> @location(0) f32 {
    // Exact 2×2 min via integer coords (handles odd sizes by clamping).
    let dims = vec2<i32>(textureDimensions(src, 0));
    let dst = vec2<i32>(i.pos.xy);
    let src0 = dst * 2;
    let x1 = min(src0.x + 1, dims.x - 1);
    let y1 = min(src0.y + 1, dims.y - 1);
    let d00 = textureLoad(src, src0, 0).r;
    let d10 = textureLoad(src, vec2(x1, src0.y), 0).r;
    let d01 = textureLoad(src, vec2(src0.x, y1), 0).r;
    let d11 = textureLoad(src, vec2(x1, y1), 0).r;
    return min(min(d00, d10), min(d01, d11));
}
