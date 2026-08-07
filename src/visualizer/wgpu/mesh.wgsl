struct FrameUniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    ambient: vec4<f32>, // xyz ambient, w = light count
    camera_pos: vec4<f32>,
    // x = ibl intensity, y = prefilter max mip, z = ibl enabled, w unused
    ibl: vec4<f32>,
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
@group(0) @binding(3) var env_equirect: texture_2d<f32>;
@group(0) @binding(4) var brdf_lut: texture_2d<f32>;
@group(0) @binding(5) var env_samp: sampler;
@group(0) @binding(6) var clamp_samp: sampler;
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

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
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

// Soft-compress HDR so studio lamps don't turn dielectrics into white sparkles.
fn compress_env(c: vec3<f32>) -> vec3<f32> {
    return c / (vec3(1.0) + c);
}

fn sample_env(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    let d = normalize(dir);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let uv = vec2(phi / (2.0 * PI) + 0.5, theta / PI);
    return compress_env(textureSampleLevel(env_equirect, env_samp, uv, lod).rgb);
}

fn ibl_contrib(
    n_in: vec3<f32>,
    v_in: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let n = normalize(n_in);
    let v = normalize(v_in);

    // Dielectric F0 ≈ 0.04; metals use albedo as F0.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let n_dot_v = max(dot(n, v), 0.001);
    let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);

    // Diffuse: heavily blurred equirect along the normal (no cubemap seams).
    // Not a perfect cosine irradiance integral, but stable and energy-sane after compress.
    let irradiance = sample_env(n, frame.ibl.y);
    let diffuse = kd * albedo * irradiance;

    // Specular: reflect env. Dielectrics only reflect ~4% at face-on (F0),
    // more at grazing via the BRDF LUT. Metals reflect strongly tinted by albedo.
    var lod = roughness * roughness * frame.ibl.y;
    // Dielectrics: never sample mip0 — HDR lamps become white pin-dots otherwise.
    if metallic < 0.5 {
        lod = max(lod, 1.25);
    }
    let r = reflect(-v, n);
    let prefiltered = sample_env(r, lod);
    let brdf = textureSample(brdf_lut, clamp_samp, vec2(n_dot_v, roughness)).rg;
    // Split-sum: scale/bias. For plastic this stays small (F0*…).
    let specular = prefiltered * (f0 * brdf.x + brdf.y);

    return (diffuse + specular) * frame.ibl.x;
}

struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) orm: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOut {
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

    var lit = vec3<f32>(0.0);
    if frame.ibl.z > 0.5 {
        lit += ibl_contrib(n, v, albedo, metallic, roughness);
    } else {
        lit += frame.ambient.xyz * albedo;
    }

    let count = u32(frame.ambient.w);
    for (var i = 0u; i < count; i++) {
        lit += light_contrib(frame.lights[i], in.world_pos, n, v, albedo, metallic, roughness);
    }
    // Always linear HDR into G-buffer color (present/tonemap happens later).
    var out: GBufferOut;
    out.color = vec4<f32>(lit, base.a);
    out.normal = vec4<f32>(n, 0.0);
    // R = occlusion placeholder, G = roughness, B = metallic
    out.orm = vec4<f32>(1.0, roughness, metallic, 1.0);
    return out;
}
