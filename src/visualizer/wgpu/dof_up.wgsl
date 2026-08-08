//! Half-res DOF → full-res upsample with CoC-based sharp restore.
//! Sky (depth ≈ 1) uses infinity CoC so cubemap blur is preserved.

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

struct DofUpUniforms {
    inv_proj: mat4x4<f32>,
    focus_distance: f32,
    aperture: f32,
    max_coc: f32,
    focus_range: f32,
    half_texel: vec2<f32>,
    depth_sigma: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> u: DofUpUniforms;
@group(0) @binding(1) var half_tex: texture_2d<f32>;
@group(0) @binding(2) var sharp_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var depth_samp: sampler;

const SKY_Z: f32 = 1.0e4;

fn view_z(uv: vec2<f32>, depth: f32) -> f32 {
    if depth >= 0.9999 {
        return SKY_Z;
    }
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = u.inv_proj * clip;
    return v.z / max(v.w, 1e-6);
}

fn coc_abs(depth_z: f32) -> f32 {
    let z = max(depth_z, 0.05);
    let dz = abs(z - u.focus_distance) - max(u.focus_range, 0.0);
    let mag = max(dz, 0.0) * u.aperture / z;
    return clamp(mag, 0.0, u.max_coc);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    let sharp = textureSampleLevel(sharp_tex, samp, i.uv, 0.0);

    let z = view_z(i.uv, depth);
    let coc = coc_abs(z);
    let blur_w = smoothstep(0.3, 1.6, coc);

    // Bilateral upsample of half-res DOF (works for sky too — half stores blurred env).
    var result = vec3(0.0);
    var wsum = 0.0;
    let offsets = array<vec2<f32>, 4>(
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(-0.5, 0.5),
        vec2(0.5, 0.5),
    );
    let sky = depth >= 0.9999;
    for (var k = 0; k < 4; k++) {
        let suv = i.uv + offsets[k] * u.half_texel;
        let s = textureSampleLevel(half_tex, samp, suv, 0.0);
        var w: f32;
        if sky {
            // Sky depth is flat — don't depth-reject neighbor taps.
            w = 1.0;
        } else {
            w = exp(-abs(depth - s.a) * u.depth_sigma) + 1e-3;
            // Also accept sky taps from half (packed depth ~1) as background fill.
            if s.a >= 0.9999 {
                w = 0.35;
            }
        }
        result += s.rgb * w;
        wsum += w;
    }
    let blurred = result / max(wsum, 1e-3);

    if blur_w < 0.01 {
        return vec4(sharp.rgb, depth);
    }
    return vec4(mix(sharp.rgb, blurred, blur_w), depth);
}
