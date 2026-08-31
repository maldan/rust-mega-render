struct ShadowFrame {
    light_view_proj: mat4x4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    albedo: vec4<f32>,
    // x = metallic, y = roughness, z = skin mode (0=off, 1=LBS, 2=DQS), w = sss
    params: vec4<f32>,
    sss: vec4<f32>,
    prev_model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> frame: ShadowFrame;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;
@group(1) @binding(1) var albedo_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var mr_tex: texture_2d<f32>;
@group(1) @binding(4) var samp: sampler;
@group(1) @binding(10) var height_tex: texture_2d<f32>;
@group(2) @binding(0) var bone_tex: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
}

fn identity4() -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}

fn load_bone_mat(index: u32) -> mat4x4<f32> {
    let i = i32(index) * 4;
    return mat4x4<f32>(
        textureLoad(bone_tex, vec2(i, 0), 0),
        textureLoad(bone_tex, vec2(i + 1, 0), 0),
        textureLoad(bone_tex, vec2(i + 2, 0), 0),
        textureLoad(bone_tex, vec2(i + 3, 0), 0),
    );
}

fn load_bone_dq(index: u32) -> mat2x4<f32> {
    let i = i32(index) * 2;
    return mat2x4<f32>(
        textureLoad(bone_tex, vec2(i, 0), 0),
        textureLoad(bone_tex, vec2(i + 1, 0), 0),
    );
}

fn dq_to_mat(real: vec4<f32>, dual: vec4<f32>) -> mat4x4<f32> {
    let x = real.x;
    let y = real.y;
    let z = real.z;
    let w = real.w;
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    let tx = 2.0 * (-dual.w * x + dual.x * w - dual.y * z + dual.z * y);
    let ty = 2.0 * (-dual.w * y + dual.y * w + dual.x * z - dual.z * x);
    let tz = 2.0 * (-dual.w * z + dual.z * w - dual.x * y + dual.y * x);
    return mat4x4<f32>(
        vec4<f32>(1.0 - (yy + zz), xy + wz, xz - wy, 0.0),
        vec4<f32>(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0),
        vec4<f32>(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0),
        vec4<f32>(tx, ty, tz, 1.0),
    );
}

fn blend_lbs(in: VertexInput) -> mat4x4<f32> {
    return load_bone_mat(in.joints.x) * in.weights.x
        + load_bone_mat(in.joints.y) * in.weights.y
        + load_bone_mat(in.joints.z) * in.weights.z
        + load_bone_mat(in.joints.w) * in.weights.w;
}

fn blend_dqs(in: VertexInput) -> mat4x4<f32> {
    var real = vec4<f32>(0.0);
    var dual = vec4<f32>(0.0);
    var w_sum = 0.0;
    var has_ref = false;
    var ref_real = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    let indices = array<u32, 4>(in.joints.x, in.joints.y, in.joints.z, in.joints.w);
    let weights = array<f32, 4>(in.weights.x, in.weights.y, in.weights.z, in.weights.w);

    for (var k = 0; k < 4; k++) {
        let w = weights[k];
        if w <= 0.0 {
            continue;
        }
        let dq = load_bone_dq(indices[k]);
        var r = dq[0];
        var d = dq[1];
        if !has_ref {
            ref_real = r;
            has_ref = true;
        } else if dot(ref_real, r) < 0.0 {
            r = -r;
            d = -d;
        }
        real += r * w;
        dual += d * w;
        w_sum += w;
    }

    if w_sum < 1e-6 {
        return identity4();
    }
    if abs(w_sum - 1.0) > 1e-3 {
        real /= w_sum;
        dual /= w_sum;
    }
    let len = length(real);
    if len < 1e-8 {
        return identity4();
    }
    real /= len;
    dual /= len;
    return dq_to_mat(real, dual);
}

fn skin_matrix(in: VertexInput) -> mat4x4<f32> {
    let mode = object.params.z;
    if mode < 0.5 {
        return identity4();
    }
    if mode < 1.5 {
        return blend_lbs(in);
    }
    return blend_dqs(in);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let model = object.model * skin_matrix(in);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: VertexOutput;
    out.clip_position = frame.light_view_proj * world;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) {
    let cutoff = object.albedo.a;
    if cutoff > 0.001 {
        let a = textureSample(albedo_tex, samp, in.uv).a;
        if a < cutoff {
            discard;
        }
    }
}
