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

struct SsaoUniforms {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    kernel: array<vec4<f32>, 32>,
    resolution: vec2<f32>,
    radius: f32,
    bias: f32,
    noise_scale: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: SsaoUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var noise_tex: texture_2d<f32>;
@group(0) @binding(4) var noise_samp: sampler;

fn view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = u.inv_proj * clip;
    return v.xyz / v.w;
}

fn view_normal(uv: vec2<f32>, pos: vec3<f32>) -> vec3<f32> {
    let texel = 1.0 / u.resolution;
    let dx = textureSample(depth_tex, depth_samp, uv + vec2(texel.x, 0.0));
    let dy = textureSample(depth_tex, depth_samp, uv + vec2(0.0, texel.y));
    let px = view_pos(uv + vec2(texel.x, 0.0), dx);
    let py = view_pos(uv + vec2(0.0, texel.y), dy);
    // Screen +X/+Y with LH view → normal toward camera when wound this way.
    var n = cross(py - pos, px - pos);
    if dot(n, -pos) < 0.0 {
        n = -n;
    }
    let len2 = dot(n, n);
    if len2 < 1e-10 {
        return normalize(-pos);
    }
    return n * inverseSqrt(len2);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    if depth >= 0.9999 {
        return vec4(1.0);
    }

    let pos = view_pos(i.uv, depth);
    let normal = view_normal(i.uv, pos);

    let noise = textureSample(noise_tex, noise_samp, i.uv * u.noise_scale).xy * 2.0 - 1.0;
    var tangent = normalize(vec3(noise, 0.0) - normal * dot(vec3(noise, 0.0), normal));
    if dot(tangent, tangent) < 1e-6 {
        tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));
    }
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3(tangent, bitangent, normal);

    // Keep screen-space kernel size roughly stable with distance.
    let radius = u.radius * max(abs(pos.z), 0.5) * 0.15;

    var occlusion = 0.0;
    var counted = 0.0;
    for (var s = 0u; s < 32u; s++) {
        let sample_view = pos + tbn * u.kernel[s].xyz * radius;
        let clip = u.proj * vec4(sample_view, 1.0);
        let ndc = clip.xyz / max(clip.w, 1e-6);
        let sample_uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        let scene_depth = textureSample(depth_tex, depth_samp, sample_uv);
        if scene_depth >= 0.9999 {
            counted += 1.0;
            continue;
        }

        let scene_pos = view_pos(sample_uv, scene_depth);
        // LH view: +Z forward, closer = smaller z.
        // Occluded when geometry is closer than the hemisphere sample.
        let dist = abs(pos.z - scene_pos.z);
        let range = smoothstep(0.0, 1.0, radius / max(dist, 1e-3));
        if scene_pos.z < sample_view.z - u.bias {
            occlusion += range;
        }
        counted += 1.0;
    }

    var ao = 1.0;
    if counted > 0.0 {
        ao = 1.0 - (occlusion / counted);
    }
    return vec4(vec3(clamp(ao, 0.0, 1.0)), 1.0);
}
