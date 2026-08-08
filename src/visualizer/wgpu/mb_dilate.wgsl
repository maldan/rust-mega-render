//! Full-res max-magnitude velocity dilate (small kernel, no tiles).

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

struct DilateUniforms {
    // xy = resolution, z = radius in pixels (1..=3), w unused
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: DilateUniforms;
@group(0) @binding(1) var velocity_tex: texture_2d<f32>;
@group(0) @binding(2) var nearest_samp: sampler;

@fragment
fn fs(i: VsOut) -> @location(0) vec2<f32> {
    let texel = 1.0 / u.params.xy;
    let radius = i32(clamp(u.params.z, 1.0, 3.0));
    var best = textureSampleLevel(velocity_tex, nearest_samp, i.uv, 0.0).xy;
    var best_l2 = dot(best, best);
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            if x == 0 && y == 0 {
                continue;
            }
            let uv = i.uv + vec2(f32(x), f32(y)) * texel;
            let v = textureSampleLevel(velocity_tex, nearest_samp, uv, 0.0).xy;
            let l2 = dot(v, v);
            if l2 > best_l2 {
                best_l2 = l2;
                best = v;
            }
        }
    }
    return best;
}
