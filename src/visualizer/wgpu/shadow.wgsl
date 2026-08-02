struct ShadowFrame {
    light_view_proj: mat4x4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    albedo: vec4<f32>,
    params: vec4<f32>,
    bones: array<mat4x4<f32>, 128>,
}

@group(0) @binding(0) var<uniform> frame: ShadowFrame;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
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
    return object.bones[in.joints.x] * in.weights.x
        + object.bones[in.joints.y] * in.weights.y
        + object.bones[in.joints.z] * in.weights.z
        + object.bones[in.joints.w] * in.weights.w;
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let model = object.model * skin_matrix(in);
    let world = model * vec4<f32>(in.position, 1.0);
    return frame.light_view_proj * world;
}
