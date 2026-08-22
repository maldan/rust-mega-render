// Dedicated hair pass: Scheuermann-style dual-specular (shifted-tangent Kajiya-Kay)
// shading, drawn after the opaque G-buffer pass with a depth prepass (fs_depth)
// followed by opaque solid cores (fs_solid) and alpha-blended fringe / soft
// (fs_fringe / fs_main). Cutout cores overwrite G-buffer aux so GTAO/SSGI see
// hair, not the head behind it. Not skinned (hair meshes are generated in
// object space directly).

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    ambient: vec4<f32>,
    camera_pos: vec4<f32>,
    ibl: vec4<f32>,
    shadow: vec4<f32>,
    gi: vec4<f32>,
    lights: array<GpuLight, 8>,
    prev_view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
}

struct GpuLight {
    pos_or_dir: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    albedo: vec4<f32>, // rgb tint, a = depth-prepass alpha cutoff
    // x = metallic (unused), y = roughness, z = skin mode (unused), w = sss (unused)
    params: vec4<f32>,
    sss: vec4<f32>, // unused for hair
    prev_model: mat4x4<f32>,
    // x = primary shift, y = secondary shift, z = primary exponent, w = secondary exponent
    hair0: vec4<f32>,
    // rgb = secondary tint, w = secondary strength
    hair1: vec4<f32>,
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
    // Homogeneous UV: store uv * q, divide by q in the fragment shader so
    // trapezoid hair cards interpolate fibers along the ribbon edges instead
    // of kinking across the two-triangle diagonal. q=1 is a no-op.
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) uv_q: f32,
    // Per-strand shade (weights.y) and layer opacity (weights.z).
    @location(5) shade_alpha: vec2<f32>,
}

const PI: f32 = 3.14159265;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = object.model * vec4<f32>(in.position, 1.0);
    out.clip_position = frame.view_proj * world;
    out.world_normal = normalize((object.model * vec4<f32>(in.normal, 0.0)).xyz);
    let tw = (object.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz;
    out.world_tangent = vec4<f32>(normalize(tw), in.tangent.w);
    let q = max(in.weights.x, 1e-5);
    out.uv = in.uv * q;
    out.uv_q = q;
    out.shade_alpha = vec2<f32>(in.weights.y, in.weights.z);
    out.world_pos = world.xyz;
    return out;
}

fn hair_uv(in: VertexOutput) -> vec2<f32> {
    return in.uv / max(in.uv_q, 1e-5);
}

/// Soften the last `object.sss.w` fraction of the strand (UV.v → tip).
/// Color/blend only — depth prepass ignores this.
fn tip_fade_mul(v: f32) -> f32 {
    let fade = object.sss.w;
    if fade <= 1e-4 {
        return 1.0;
    }
    let start = clamp(1.0 - fade, 0.0, 0.999);
    if v <= start {
        return 1.0;
    }
    let u = clamp((v - start) / max(1.0 - start, 1e-4), 0.0, 1.0);
    return 1.0 - u * u * (3.0 - 2.0 * u);
}

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

fn vogel(i: u32, count: u32, phi: f32) -> vec2<f32> {
    let r = sqrt((f32(i) + 0.5) / f32(count));
    let theta = f32(i) * 2.39996323 + phi;
    return vec2(cos(theta), sin(theta)) * r;
}

fn pcf_vogel(uv: vec2<f32>, z_recv: f32, radius: f32, phi: f32) -> f32 {
    const TAPS: u32 = 16u;
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
    let bias_scale = max(frame.gi.z, 0.0);
    let bias = max(bias_scale * (1.0 - dot(n, l)), bias_scale * 0.3);
    let z_recv = ndc.z - bias;
    let texel = frame.shadow.z;
    let map_res = 1.0 / max(texel, 1e-6);
    let phi = ign(uv * map_res) * 6.2831853;
    return pcf_vogel(uv, z_recv, texel * 2.5, phi);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

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
    let t = clamp(roughness, 0.0, 1.0) * levels;
    let i0 = i32(floor(t));
    let i1 = min(i0 + 1, i32(levels));
    let f = fract(t);
    return mix(sample_env_level(uv, i0), sample_env_level(uv, i1), f);
}

fn env_reflect(n_in: vec3<f32>, v_in: vec3<f32>, albedo: vec3<f32>, roughness: f32) -> vec3<f32> {
    let n = normalize(n_in);
    let v = normalize(v_in);
    let rgh = clamp(roughness, 0.04, 1.0);
    let gloss = 1.0 - rgh;
    let energy = gloss * gloss * 0.5;
    if energy < 0.002 {
        return vec3(0.0);
    }
    let f0 = vec3<f32>(0.04);
    let n_dot_v = max(dot(n, v), 0.001);
    let f = fresnel_schlick(n_dot_v, f0);
    let r = reflect(-v, n);
    return sample_env(r, rgh) * f * energy * frame.ibl.x * albedo;
}

/// Shift a strand tangent along the shading normal (Scheuermann's shifted-tangent trick).
fn shift_tangent(t: vec3<f32>, n: vec3<f32>, shift: f32) -> vec3<f32> {
    return normalize(t + shift * n);
}

/// Single Kajiya-Kay-like specular lobe for a thin cylindrical fiber.
fn strand_specular(t: vec3<f32>, v: vec3<f32>, l: vec3<f32>, exponent: f32) -> f32 {
    let h = normalize(v + l);
    let dot_th = dot(t, h);
    let sin_th = sqrt(clamp(1.0 - dot_th * dot_th, 0.0, 1.0));
    let dir_atten = smoothstep(-1.0, 0.0, dot_th);
    return dir_atten * pow(sin_th, exponent);
}

fn hair_light_contrib(
    light: GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    t: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
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

    var s = 1.0;
    if light.params.y > 0.5 {
        s = shadow_factor(world_pos, n, l);
    }

    // Kajiya diffuse: a thin fiber scatters ∝ sin(angle to light), not N·L.
    // Also fold in a mild N·L-ish term so fully back-facing strands still dim a bit
    // (pure sin(T,L) never reaches zero, which otherwise reads as "lit from everywhere").
    let tdotl = clamp(dot(t, l), -1.0, 1.0);
    let diffuse_amt = sqrt(max(1.0 - tdotl * tdotl, 0.0));
    let facing = clamp(dot(n, l) * 0.5 + 0.5, 0.35, 1.0);
    let diffuse = albedo * (diffuse_amt * facing / PI);

    // Rougher fiber surface ⇒ broader, dimmer highlight (matte hair); smoother ⇒
    // tight, bright highlight (silky hair). Mirrors how roughness drives specular
    // everywhere else in the renderer, just reusing the dual-lobe exponents as the
    // "smooth" end of the range instead of a GGX lobe.
    let rgh = clamp(roughness, 0.04, 1.0);
    let gloss = 1.0 - rgh;
    let spec_energy = gloss * gloss;
    let exp1 = mix(4.0, object.hair0.z, spec_energy);
    let exp2 = mix(4.0, object.hair0.w, spec_energy);

    let t1 = shift_tangent(t, n, object.hair0.x);
    let t2 = shift_tangent(t, n, object.hair0.y);
    let spec1 = strand_specular(t1, v, l, exp1);
    let spec2 = strand_specular(t2, v, l, exp2) * object.hair1.w;
    let specular = (vec3<f32>(spec1) + spec2 * object.hair1.xyz) * spec_energy;

    let radiance = light.color.xyz * light.color.w * atten * s;
    return (diffuse + specular) * radiance;
}

struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) orm: vec4<f32>,
    @location(4) albedo: vec4<f32>,
}

/// Depth + G-buffer wipe: hard alpha-test before cutout hair shading.
/// Aux targets are cleared so post cannot keep the head's normals/albedo.
/// (Skipped when HairShading.soft_blend is on — see host draw loop.)
@fragment
fn fs_depth(in: VertexOutput) -> GBufferOut {
    let uv = hair_uv(in);
    let a = textureSample(albedo_tex, samp, uv).a * in.shade_alpha.y;
    let high_cut = max(object.albedo.a, 0.02);
    if a < high_cut {
        discard;
    }
    var out: GBufferOut;
    out.color = vec4<f32>(0.0);
    out.normal = vec4<f32>(0.0);
    out.velocity = vec2<f32>(0.0);
    out.orm = vec4<f32>(0.0);
    out.albedo = vec4<f32>(0.0);
    return out;
}

/// Soft-blend path: full live texture alpha (no depth prepass on host).
@fragment
fn fs_main(in: VertexOutput) -> GBufferOut {
    return shade_hair(in, 0u);
}

/// Cutout solid cores: opaque fill where coverage ≥ cutout (after depth prepass).
@fragment
fn fs_solid(in: VertexOutput) -> GBufferOut {
    return shade_hair(in, 1u);
}

/// Cutout soft fringe: only the below-cutout coverage, soft-blended on top.
@fragment
fn fs_fringe(in: VertexOutput) -> GBufferOut {
    return shade_hair(in, 2u);
}

@fragment
fn fs_wire_fill(_in: VertexOutput) -> GBufferOut {
    return wire_gbuf(vec3<f32>(0.36, 0.37, 0.39));
}

@fragment
fn fs_wire(_in: VertexOutput) -> GBufferOut {
    return wire_gbuf(vec3<f32>(0.92, 0.93, 0.95));
}

fn wire_gbuf(rgb: vec3<f32>) -> GBufferOut {
    var out: GBufferOut;
    out.color = vec4<f32>(rgb, 1.0);
    out.normal = vec4<f32>(0.0);
    out.velocity = vec2<f32>(0.0);
    out.orm = vec4<f32>(0.0);
    out.albedo = vec4<f32>(rgb, 1.0);
    return out;
}

fn shade_hair(in: VertexOutput, mode: u32) -> GBufferOut {
    let uv = hair_uv(in);
    let sampled = textureSample(albedo_tex, samp, uv);
    let fade = tip_fade_mul(uv.y);
    var alpha = sampled.a * in.shade_alpha.y * fade;
    let high_cut = max(object.albedo.a, 0.02);

    if mode == 1u {
        // Solid cores — kill softness so depth occlusion actually reads as cover.
        if alpha < high_cut {
            discard;
        }
        alpha = 1.0;
    } else if mode == 2u {
        // Soft fringe only — cores already drawn opaque in fs_solid.
        if alpha >= high_cut || alpha < 0.02 {
            discard;
        }
    } else {
        // Full soft composite.
        if alpha < 0.02 {
            discard;
        }
    }

    let albedo = sampled.rgb * object.albedo.rgb * in.shade_alpha.x;
    let mr = textureSample(mr_tex, samp, uv);
    let roughness = clamp(object.params.y * mr.g, 0.04, 1.0);

    let geo_n = normalize(in.world_normal);
    let t_geo = normalize(in.world_tangent.xyz - geo_n * dot(geo_n, in.world_tangent.xyz));
    let t = normalize(cross(geo_n, t_geo) * in.world_tangent.w);
    let n_sample = textureSample(normal_tex, samp, uv).xyz * 2.0 - vec3<f32>(1.0);
    let n = normalize(t_geo * n_sample.x + t * n_sample.y + geo_n * n_sample.z);
    let v = normalize(frame.camera_pos.xyz - in.world_pos);

    let ambient_gi = clamp(frame.gi.x, 0.0, 1.0);
    var lit = frame.ambient.xyz * albedo * ambient_gi;
    if frame.ibl.z > 0.5 {
        lit += env_reflect(n, v, albedo, roughness);
    }

    let count = u32(frame.ambient.w);
    for (var i = 0u; i < count; i++) {
        lit += hair_light_contrib(frame.lights[i], in.world_pos, n, t, v, albedo, roughness);
    }

    var out: GBufferOut;
    out.color = vec4<f32>(lit, alpha);
    if mode == 1u {
        // Opaque cores: G-buffer belongs to the hair surface.
        out.normal = vec4<f32>(n, 0.0);
        out.velocity = vec2<f32>(0.0);
        out.orm = vec4<f32>(1.0, roughness, 0.0, 1.0);
        out.albedo = vec4<f32>(albedo, 1.0);
    } else {
        // Soft / fringe: host pipelines do not write aux targets.
        out.normal = vec4<f32>(0.0);
        out.velocity = vec2<f32>(0.0);
        out.orm = vec4<f32>(0.0);
        out.albedo = vec4<f32>(0.0);
    }
    return out;
}
