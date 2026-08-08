struct ShadowFrame {
    light_view_proj: mat4x4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    albedo: vec4<f32>,
    params: vec4<f32>,
    sss: vec4<f32>,
    prev_model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> frame: ShadowFrame;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;
@group(2) @binding(0) var bone_tex: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
}

fn load_bone(index: u32) -> mat4x4<f32> {
    let i = i32(index) * 4;
    return mat4x4<f32>(
        textureLoad(bone_tex, vec2(i, 0), 0),
        textureLoad(bone_tex, vec2(i + 1, 0), 0),
        textureLoad(bone_tex, vec2(i + 2, 0), 0),
        textureLoad(bone_tex, vec2(i + 3, 0), 0),
    );
}

fn skin_matrix(in: VertexInput) -> mat4x4<f32> {
    if object.params.z < 0.5 {
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
    }
    return load_bone(in.joints.x) * in.weights.x
        + load_bone(in.joints.y) * in.weights.y
        + load_bone(in.joints.z) * in.weights.z
        + load_bone(in.joints.w) * in.weights.w;
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let model = object.model * skin_matrix(in);
    let world = model * vec4<f32>(in.position, 1.0);
    return frame.light_view_proj * world;
}
