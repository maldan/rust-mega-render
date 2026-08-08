struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    // x = intensity, y = yaw rotation (radians), zw = resolution
    params: vec4<f32>,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) dir: vec3<f32>,
    @location(1) velocity_px: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: SkyUniforms;
@group(0) @binding(1) var env_sharp: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

const PI: f32 = 3.14159265;

fn rotate_y(d: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3(d.x * c + d.z * s, d.y, -d.x * s + d.z * c);
}

fn clip_to_uv(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / max(clip.w, 1e-6);
    return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
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
    o.pos = vec4(xy, 1.0, 1.0);
    let world = u.inv_view_proj * vec4(xy, 1.0, 1.0);
    o.dir = normalize(world.xyz / max(world.w, 1e-6));
    // Camera-only velocity for infinite sky.
    let prev_clip = u.prev_view_proj * vec4(o.dir, 0.0);
    let curr_uv = clip_to_uv(vec4(xy, 1.0, 1.0));
    let prev_uv = clip_to_uv(prev_clip);
    o.velocity_px = clamp((curr_uv - prev_uv) * u.params.zw, vec2(-256.0), vec2(256.0));
    return o;
}

struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) orm: vec4<f32>,
    @location(4) albedo: vec4<f32>,
}

@fragment
fn fs(i: VsOut) -> GBufferOut {
    let d = rotate_y(normalize(i.dir), u.params.y);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let uv = vec2(phi / (2.0 * PI) + 0.5, theta / PI);
    let color = textureSampleLevel(env_sharp, env_samp, uv, 0.0).rgb;
    var out: GBufferOut;
    out.color = vec4(color * u.params.x, 1.0);
    out.normal = vec4(-normalize(i.dir), 0.0);
    out.velocity = i.velocity_px;
    out.orm = vec4(1.0, 1.0, 0.0, 1.0);
    // Sky is not a diffuse bounce surface for SSGI.
    out.albedo = vec4(0.0);
    return out;
}
