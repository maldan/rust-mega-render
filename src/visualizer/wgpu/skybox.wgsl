struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) dir: vec3<f32>,
}

struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    params: vec4<f32>, // x = intensity
}

@group(0) @binding(0) var<uniform> u: SkyUniforms;
@group(0) @binding(1) var env_equirect: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

const PI: f32 = 3.14159265;

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
    return o;
}

struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) orm: vec4<f32>,
}

@fragment
fn fs(i: VsOut) -> GBufferOut {
    let d = normalize(i.dir);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let uv = vec2(phi / (2.0 * PI) + 0.5, theta / PI);
    let color = textureSampleLevel(env_equirect, env_samp, uv, 0.0).rgb;
    var out: GBufferOut;
    out.color = vec4(color * u.params.x, 1.0);
    out.normal = vec4(-d, 0.0);
    out.orm = vec4(1.0, 1.0, 0.0, 1.0);
    return out;
}
