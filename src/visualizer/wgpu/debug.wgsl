struct DebugUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: DebugUniforms;

struct LineInput {
    @location(0) start: vec3<f32>,
    @location(1) width_from: f32,
    @location(2) end: vec3<f32>,
    @location(3) width_to: f32,
    @location(4) color_from: vec4<f32>,
    @location(5) color_to: vec4<f32>,
}

struct PointInput {
    @location(0) position: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
}

struct TriInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_line(@builtin(vertex_index) vid: u32, in: LineInput) -> VertexOutput {
    // Quad corners: (t, side) — t=0 start / t=1 end, side = ±0.5
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, -0.5), vec2(0.0, 0.5), vec2(1.0, 0.5),
        vec2(0.0, -0.5), vec2(1.0, 0.5), vec2(1.0, -0.5),
    );
    let c = corners[vid];
    let t = c.x;
    let side = c.y;

    let clip0 = frame.view_proj * vec4<f32>(in.start, 1.0);
    let clip1 = frame.view_proj * vec4<f32>(in.end, 1.0);

    // Screen-space direction for a stable perpendicular.
    var dir = (clip1.xy / clip1.w - clip0.xy / clip0.w) * frame.resolution;
    let len = length(dir);
    if len > 1e-6 {
        dir = dir / len;
    } else {
        dir = vec2(1.0, 0.0);
    }
    let perp = vec2(-dir.y, dir.x);

    let clip = mix(clip0, clip1, t);
    let width = mix(in.width_from, in.width_to, t);
    let offset = perp * side * width * (2.0 / frame.resolution) * clip.w;

    var out: VertexOutput;
    out.clip_position = vec4(clip.xy + offset, clip.z, clip.w);
    out.color = mix(in.color_from, in.color_to, t);
    return out;
}

@vertex
fn vs_point(@builtin(vertex_index) vid: u32, in: PointInput) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
        vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5),
    );
    let clip = frame.view_proj * vec4<f32>(in.position, 1.0);
    let px = corners[vid] * in.size * (2.0 / frame.resolution) * clip.w;
    var out: VertexOutput;
    out.clip_position = vec4<f32>(clip.xy + px, clip.z, clip.w);
    out.color = in.color;
    return out;
}

@vertex
fn vs_tri(in: TriInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = frame.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) orm: vec4<f32>,
    @location(4) albedo: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOut {
    var out: GBufferOut;
    out.color = in.color;
    out.normal = vec4(0.0);
    out.velocity = vec2(0.0);
    out.orm = vec4(0.0);
    out.albedo = vec4(0.0);
    return out;
}
