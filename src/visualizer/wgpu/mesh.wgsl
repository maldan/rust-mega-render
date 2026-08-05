struct FrameUniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    ambient: vec4<f32>, // xyz ambient, w = light count
    camera_pos: vec4<f32>,
    lights: array<GpuLight, 8>,
}

struct GpuLight {
    pos_or_dir: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    albedo: vec4<f32>,
    // x = metallic, y = roughness, z = skinned
    params: vec4<f32>,
    bones: array<mat4x4<f32>, 128>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_samp: sampler_comparison;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;
@group(1) @binding(1) var albedo_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var mr_tex: texture_2d<f32>;
@group(1) @binding(4) var samp: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) world_tangent: vec4<f32>,
}

const PI: f32 = 3.14159265;

fn skin_matrix(in: VertexInput) -> mat4x4<f32> {
    if object.params.z < 0.5 {
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
    }
    return object.bones[in.joints.x] * in.weights.x
        + object.bones[in.joints.y] * in.weights.y
        + object.bones[in.joints.z] * in.weights.z
        + object.bones[in.joints.w] * in.weights.w;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = object.model * skin_matrix(in);
    let world = model * vec4<f32>(in.position, 1.0);
    out.clip_position = frame.view_proj * world;
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    let tw = (model * vec4<f32>(in.tangent.xyz, 0.0)).xyz;
    out.world_tangent = vec4<f32>(normalize(tw), in.tangent.w);
    out.uv = in.uv;
    out.world_pos = world.xyz;
    return out;
}

fn shadow_factor(world_pos: vec3<f32>, n: vec3<f32>, l: vec3<f32>) -> f32 {
    let light_clip = frame.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let bias = max(0.0004 * (1.0 - dot(n, l)), 0.0001);
    var shadow = 0.0;
    let texel = 1.0 / 2048.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            shadow += textureSampleCompare(shadow_map, shadow_samp, uv + offset, ndc.z - bias);
        }
    }
    return shadow / 9.0;
}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let nh = max(dot(n, h), 0.0);
    let nh2 = nh * nh;
    let denom = nh2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(nv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return nv / (nv * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let nv = max(dot(n, v), 0.0);
    let nl = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(nv, roughness) * geometry_schlick_ggx(nl, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn light_contrib(
    light: GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    var l: vec3<f32>;
    var atten = 1.0;
    if light.pos_or_dir.w < 0.5 {
        l = normalize(-light.pos_or_dir.xyz);
    } else {
        let to_light = light.pos_or_dir.xyz - world_pos;
        let dist = length(to_light);
        let range = max(light.params.x, 0.001);
        if dist > range {
            return vec3<f32>(0.0);
        }
        l = to_light / dist;
        let x = dist / range;
        atten = (1.0 - x * x) * (1.0 - x * x);
    }

    let h = normalize(v + l);
    let ndotl = max(dot(n, l), 0.0);
    if ndotl <= 0.0 {
        return vec3<f32>(0.0);
    }

    var s = 1.0;
    if light.params.y > 0.5 {
        s = shadow_factor(world_pos, n, l);
    }

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let d = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let specular = (d * g * f) / max(4.0 * max(dot(n, v), 0.0) * ndotl, 0.001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    let radiance = light.color.xyz * light.color.w * atten * s;
    return (diffuse + specular) * radiance * ndotl;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(albedo_tex, samp, in.uv) * object.albedo;
    let albedo = base.rgb;
    let mr = textureSample(mr_tex, samp, in.uv);
    let metallic = object.params.x * mr.b;
    let roughness = max(object.params.y * mr.g, 0.04);

    let t = normalize(in.world_tangent.xyz);
    let n0 = normalize(in.world_normal);
    let b = cross(n0, t) * in.world_tangent.w;
    let n_ts = textureSample(normal_tex, samp, in.uv).xyz * 2.0 - 1.0;
    let n = normalize(mat3x3(t, b, n0) * n_ts);

    let v = normalize(frame.camera_pos.xyz - in.world_pos);

    var lit = frame.ambient.xyz * albedo;
    let count = u32(frame.ambient.w);
    for (var i = 0u; i < count; i++) {
        lit += light_contrib(frame.lights[i], in.world_pos, n, v, albedo, metallic, roughness);
    }
    // camera_pos.w > 0.5 → linear HDR for post tonemap; else Reinhard for direct output.
    if frame.camera_pos.w < 0.5 {
        lit = lit / (lit + vec3<f32>(1.0));
    }
    return vec4<f32>(lit, base.a);
}
