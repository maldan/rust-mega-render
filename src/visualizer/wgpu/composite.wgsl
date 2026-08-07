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

struct CompositeUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    fog_color: vec4<f32>, // rgb + density
    fog_height: f32,
    fog_height_falloff: f32,
    fog_enabled: f32,
    ao_intensity: f32,
    bloom_intensity: f32,
    exposure: f32,
    tonemap_mode: f32, // 0 = none (already LDR), 1 = reinhard, 2 = aces
    contrast: f32,
    saturation: f32,
    brightness: f32,
    vignette_intensity: f32,
    vignette_smoothness: f32,
    grain_intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: CompositeUniforms;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var ao_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var depth_samp: sampler;

fn world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = u.inv_view_proj * clip;
    return w.xyz / max(w.w, 1e-6);
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    // Narkowicz 2015 fitted ACES.
    let a = x * (x * 2.51 + 0.03);
    let b = x * (x * 2.43 + 0.59) + 0.14;
    return clamp(a / b, vec3(0.0), vec3(1.0));
}

fn reinhard_tonemap(x: vec3<f32>) -> vec3<f32> {
    return x / (x + vec3(1.0));
}

fn apply_grade(color: vec3<f32>) -> vec3<f32> {
    var c = color + u.brightness;
    c = (c - 0.5) * u.contrast + 0.5;
    let luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma), c, u.saturation);
    return clamp(c, vec3(0.0), vec3(1.0));
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    var color = textureSample(scene_tex, samp, i.uv).rgb;

    if u.ao_intensity > 0.0 {
        let ao = textureSample(ao_tex, samp, i.uv).r;
        // intensity 0 = off, 1 = full AO, >1 = stronger darkening.
        color *= mix(1.0, ao, clamp(u.ao_intensity, 0.0, 2.0));
    }
    if u.bloom_intensity > 0.0 {
        color += textureSample(bloom_tex, samp, i.uv).rgb * u.bloom_intensity;
    }

    if u.fog_enabled > 0.5 {
        let depth = textureSample(depth_tex, depth_samp, i.uv);
        if depth < 0.9999 {
            let wp = world_pos(i.uv, depth);
            let dist = length(wp - u.camera_pos.xyz);
            let height_factor = exp(-u.fog_height_falloff * max(wp.y - u.fog_height, 0.0));
            let fog = 1.0 - exp(-u.fog_color.w * dist * height_factor);
            color = mix(color, u.fog_color.xyz, clamp(fog, 0.0, 1.0));
        }
    }

    color *= max(u.exposure, 0.0);

    if u.tonemap_mode > 1.5 {
        color = aces_tonemap(color);
    } else if u.tonemap_mode > 0.5 {
        color = reinhard_tonemap(color);
    }

    if u.contrast != 1.0 || u.saturation != 1.0 || u.brightness != 0.0 {
        color = apply_grade(color);
    }

    if u.vignette_intensity > 0.0 {
        let d = length(i.uv - vec2(0.5));
        let soft = max(u.vignette_smoothness, 0.05);
        let vig = 1.0 - smoothstep(0.35, 0.35 + soft, d);
        color *= mix(1.0, vig, u.vignette_intensity);
    }

    if u.grain_intensity > 0.0 {
        let n = hash12(i.uv * vec2(1920.0, 1080.0)) * 2.0 - 1.0;
        color += n * u.grain_intensity;
    }

    return vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
}
