//! Screen-space contact shadows along the primary directional light.
//! LH view (+Z forward), DirectX clip depth.

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

struct ContactUniforms {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    /// Light travel direction in world (from sun toward scene).
    light_dir_world: vec4<f32>,
    resolution: vec2<f32>,
    length: f32,
    thickness: f32,
    /// x = samples, y = bias
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: ContactUniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var depth_samp: sampler;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_samp: sampler;

fn view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = u.inv_proj * clip;
    return v.xyz / v.w;
}

fn project_uv(view_p: vec3<f32>) -> vec3<f32> {
    let clip = u.proj * vec4(view_p, 1.0);
    let ndc = clip.xyz / max(clip.w, 1e-6);
    let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3(uv, ndc.z);
}

fn ign(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_samp, i.uv);
    if depth >= 0.9999 {
        return vec4(1.0);
    }

    let pos = view_pos(i.uv, depth);
    var n_world = textureSample(normal_tex, normal_samp, i.uv).xyz;
    if dot(n_world, n_world) < 1e-6 {
        return vec4(1.0);
    }
    n_world = normalize(n_world);
    let n_view = normalize((u.view * vec4(n_world, 0.0)).xyz);

    let light_travel = normalize(u.light_dir_world.xyz);
    let light_travel_v = normalize((u.view * vec4(light_travel, 0.0)).xyz);
    // March toward the light source (against travel direction).
    let ray_dir = -light_travel_v;

    let ndotl = dot(n_view, ray_dir);
    if ndotl <= 0.05 {
        return vec4(1.0);
    }

    let samples = max(u32(u.params.x), 4u);
    let bias = u.params.y;
    let jitter = ign(i.uv * u.resolution);

    var shadow = 1.0;
    var step_i = 0u;
    loop {
        if step_i >= samples {
            break;
        }
        let t = (f32(step_i) + jitter) / f32(samples) * u.length + bias;
        let sample_pos = pos + ray_dir * t;
        let projected = project_uv(sample_pos);
        if projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0 {
            break;
        }

        let scene_depth = textureSample(depth_tex, depth_samp, projected.xy);
        if scene_depth >= 0.9999 {
            step_i = step_i + 1u;
            continue;
        }

        let scene_pos = view_pos(projected.xy, scene_depth);
        // Positive delta = occluder closer to camera than ray sample.
        let delta = sample_pos.z - scene_pos.z;
        if delta > 0.0 && delta < u.thickness {
            // Soft edge by remaining ray fraction + N·L.
            let edge = 1.0 - t / max(u.length, 1e-4);
            shadow = min(shadow, 1.0 - clamp(edge * ndotl, 0.0, 1.0));
            break;
        }

        step_i = step_i + 1u;
    }

    return vec4(shadow);
}
