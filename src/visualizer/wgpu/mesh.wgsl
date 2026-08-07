struct FrameUniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    ambient: vec4<f32>, // xyz ambient, w = light count
    camera_pos: vec4<f32>,
    // x = env intensity, y = blur layer count, z = env active, w = pcss filter samples
    ibl: vec4<f32>,
    // x = filter (0=pcf, 1=pcss), y = light_size, z = 1/shadow_map_size, w = blocker samples
    shadow: vec4<f32>,
    // x = constant-ambient scale, y = env yaw rotation (radians)
    gi: vec4<f32>,
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
@group(0) @binding(3) var env_sharp: texture_2d<f32>;
@group(0) @binding(4) var env_blur: texture_2d_array<f32>;
@group(0) @binding(5) var env_samp: sampler;
@group(0) @binding(6) var shadow_depth_samp: sampler;
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

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

/// Vogel disk sample in unit disk; `phi` rotates the pattern per-pixel to kill banding.
fn vogel(i: u32, count: u32, phi: f32) -> vec2<f32> {
    let r = sqrt((f32(i) + 0.5) / f32(count));
    let theta = f32(i) * 2.39996323 + phi;
    return vec2(cos(theta), sin(theta)) * r;
}

fn pcf_3x3(uv: vec2<f32>, z_recv: f32, texel: f32) -> f32 {
    var shadow = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            shadow += textureSampleCompare(shadow_map, shadow_samp, uv + offset, z_recv);
        }
    }
    return shadow / 9.0;
}

/// Soft disk PCF — hides shadow-map texel squares better than a 3×3 grid up close.
fn pcf_vogel(uv: vec2<f32>, z_recv: f32, radius: f32, phi: f32) -> f32 {
    const TAPS: u32 = 24u;
    var shadow = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < TAPS; i++) {
        let sample_uv = uv + vogel(i, TAPS, phi) * radius;
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }
        shadow += textureSampleCompare(shadow_map, shadow_samp, sample_uv, z_recv);
        wsum += 1.0;
    }
    return select(1.0, shadow / wsum, wsum > 0.5);
}

fn shadow_factor(world_pos: vec3<f32>, n: vec3<f32>, l: vec3<f32>) -> f32 {
    let light_clip = frame.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let bias = max(0.0005 * (1.0 - dot(n, l)), 0.00015);
    let z_recv = ndc.z - bias;
    let texel = frame.shadow.z;
    let use_pcss = frame.shadow.x > 0.5;
    // Continuous noise — NOT floored to shadow texels (that locks pattern into visible squares).
    let map_res = 1.0 / max(texel, 1e-6);
    let phi = ign(uv * map_res) * 6.2831853;

    if !use_pcss {
        return pcf_3x3(uv, z_recv, texel);
    }

    let softness = clamp(frame.shadow.y, 0.0, 1.0);
    let blocker_taps = u32(clamp(frame.shadow.w, 4.0, 16.0));
    let filter_taps = u32(clamp(frame.ibl.w, 8.0, 48.0));

    if softness < 0.001 {
        return pcf_vogel(uv, z_recv, texel * 2.5, phi);
    }

    let size_uv = softness * softness * 0.10;
    let search_radius = max(size_uv * 0.7, texel * 4.0);
    let blocker_bias = bias * 2.5 + texel * 1.5;

    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    for (var i = 0u; i < blocker_taps; i++) {
        let sample_uv = uv + vogel(i, blocker_taps, phi) * search_radius;
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }
        let z = textureSample(shadow_map, shadow_depth_samp, sample_uv);
        if z + blocker_bias < z_recv {
            blocker_sum += z;
            blocker_count += 1.0;
        }
    }

    let min_radius = texel * mix(2.5, 6.0, softness);
    if blocker_count < 1.5 {
        return pcf_vogel(uv, z_recv, min_radius, phi);
    }

    let avg_blocker = blocker_sum / blocker_count;
    let gap = max(z_recv - avg_blocker, 0.0);
    var penumbra = gap * size_uv * 8.0;
    penumbra = clamp(penumbra, min_radius, max(size_uv, min_radius));

    var shadow = 0.0;
    var wsum = 0.0;
    let phi_f = phi + 1.6180339;
    for (var i = 0u; i < filter_taps; i++) {
        let sample_uv = uv + vogel(i, filter_taps, phi_f) * penumbra;
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }
        shadow += textureSampleCompare(shadow_map, shadow_samp, sample_uv, z_recv);
        wsum += 1.0;
    }
    return select(pcf_vogel(uv, z_recv, min_radius, phi), shadow / wsum, wsum > 0.5);
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

// Sharp full-res + discrete blurred ≤1024 maps (no mip pyramid).
fn rotate_y(d: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3(d.x * c + d.z * s, d.y, -d.x * s + d.z * c);
}

fn dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = rotate_y(normalize(dir), frame.gi.y);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, theta / PI);
}

fn sample_env_level(uv: vec2<f32>, idx: i32) -> vec3<f32> {
    // idx 0 = sharp full-res; 1..N = blur array layers (still ~1024 wide).
    if idx <= 0 {
        return textureSampleLevel(env_sharp, env_samp, uv, 0.0).rgb;
    }
    let layers = max(i32(frame.ibl.y), 1);
    let layer = u32(clamp(idx - 1, 0, layers - 1));
    return textureSampleLevel(env_blur, env_samp, uv, layer, 0.0).rgb;
}

fn sample_env(dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    let uv = dir_to_uv(dir);
    let levels = max(frame.ibl.y, 1.0);
    // 0 = sharp, levels = last blur layer.
    let t = clamp(roughness, 0.0, 1.0) * levels;
    let i0 = i32(floor(t));
    let i1 = min(i0 + 1, i32(levels));
    let f = fract(t);
    return mix(sample_env_level(uv, i0), sample_env_level(uv, i1), f);
}

fn env_reflect(
    n_in: vec3<f32>,
    v_in: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let n = normalize(n_in);
    let v = normalize(v_in);
    let rgh = clamp(roughness, 0.04, 1.0);

    // Matte: almost no mirror; glossy keeps most of the env.
    let gloss = 1.0 - rgh;
    let energy = gloss * gloss;
    if energy < 0.002 {
        return vec3(0.0);
    }

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let n_dot_v = max(dot(n, v), 0.001);
    let f = fresnel_schlick(n_dot_v, f0);
    let r = reflect(-v, n);
    return sample_env(r, rgh) * f * energy * frame.ibl.x;
}

struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) orm: vec4<f32>,
    @location(3) albedo: vec4<f32>,
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

    // Scale down constant ambient when SSGI is active (frame.gi.x).
    let ambient_gi = clamp(frame.gi.x, 0.0, 1.0);

    var lit = frame.ambient.xyz * albedo * ambient_gi;
    if frame.ibl.z > 0.5 {
        lit += env_reflect(n, v, albedo, metallic, roughness);
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
    out.albedo = vec4<f32>(albedo, 1.0);
    return out;
}
