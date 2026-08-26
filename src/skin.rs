use super::mesh::Mesh;
use super::node::Node;
use super::store::{Handle, Store};
use glam::{Mat3, Mat4, Quat, Vec3};

pub use crate::io::skin::{SkinBytesError, SkinFile};

/// How skinned vertices blend joint transforms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SkinningMode {
    /// Linear blend of 4×4 matrices (glTF default). Volume loss on bends.
    #[default]
    LinearBlend,
    /// Dual-quaternion blend. Better volume on joints; scale is discarded.
    DualQuat,
}

impl SkinningMode {
    /// Value written to `ObjectUniforms.params.z` for skinned draws.
    pub fn shader_flag(self) -> f32 {
        match self {
            Self::LinearBlend => 1.0,
            Self::DualQuat => 2.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LinearBlend => "LBS",
            Self::DualQuat => "DQS",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::LinearBlend => Self::DualQuat,
            Self::DualQuat => Self::LinearBlend,
        }
    }
}

pub struct Skin {
    pub joints: Vec<Handle<Node>>,
    pub inverse_bind: Vec<Mat4>,
}

impl Skin {
    /// Look up each joint node and write bind matrices + names + parents + rest pose.
    /// Handles are not stored; `nodes` is only the lookup table.
    pub fn to_bytes(&self, nodes: &Store<Node>) -> Vec<u8> {
        SkinFile::from_skin(self, nodes).to_bytes()
    }

    /// Spawn joint nodes into `nodes` and return a skin whose `joints` point at them.
    pub fn from_bytes(bytes: &[u8], nodes: &mut Store<Node>) -> Result<Self, SkinBytesError> {
        Ok(SkinFile::from_bytes(bytes)?.into_skin(nodes))
    }
}

/// Unit dual quaternion: real = rotation, dual encodes translation.
#[derive(Clone, Copy, Debug)]
pub struct DualQuat {
    pub real: Quat,
    pub dual: Quat,
}

impl DualQuat {
    pub const IDENTITY: Self = Self {
        real: Quat::IDENTITY,
        dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
    };

    /// Rigid part of `m` → dual quat (scale stripped by normalizing basis).
    pub fn from_mat4(m: Mat4) -> Self {
        let t = m.w_axis.truncate();
        let x = m.x_axis.truncate();
        let y = m.y_axis.truncate();
        let z = m.z_axis.truncate();
        let lx = x.length();
        let ly = y.length();
        let lz = z.length();
        let rot = if lx > 1e-8 && ly > 1e-8 && lz > 1e-8 {
            Mat3::from_cols(x / lx, y / ly, z / lz)
        } else {
            Mat3::IDENTITY
        };
        let real = Quat::from_mat3(&rot).normalize();
        let tq = Quat::from_xyzw(t.x, t.y, t.z, 0.0);
        let dual = (tq * real) * 0.5;
        Self { real, dual }
    }

    pub fn to_mat4(self) -> Mat4 {
        let r = self.real.normalize();
        let d = self.dual;
        // t = 2 * dual * conjugate(real)
        let t = translation_from_dq(r, d);
        Mat4::from_rotation_translation(r, t)
    }

    pub fn negate(self) -> Self {
        Self {
            real: Quat::from_xyzw(-self.real.x, -self.real.y, -self.real.z, -self.real.w),
            dual: Quat::from_xyzw(-self.dual.x, -self.dual.y, -self.dual.z, -self.dual.w),
        }
    }

    pub fn mul_scalar(self, s: f32) -> Self {
        Self {
            real: Quat::from_xyzw(
                self.real.x * s,
                self.real.y * s,
                self.real.z * s,
                self.real.w * s,
            ),
            dual: Quat::from_xyzw(
                self.dual.x * s,
                self.dual.y * s,
                self.dual.z * s,
                self.dual.w * s,
            ),
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            real: Quat::from_xyzw(
                self.real.x + other.real.x,
                self.real.y + other.real.y,
                self.real.z + other.real.z,
                self.real.w + other.real.w,
            ),
            dual: Quat::from_xyzw(
                self.dual.x + other.dual.x,
                self.dual.y + other.dual.y,
                self.dual.z + other.dual.z,
                self.dual.w + other.dual.w,
            ),
        }
    }

    pub fn normalize(self) -> Self {
        let len = self.real.length();
        if len < 1e-8 {
            return Self::IDENTITY;
        }
        self.mul_scalar(1.0 / len)
    }

    /// Pack as two RGBA32F texels: real then dual.
    pub fn to_texels(self) -> [[f32; 4]; 2] {
        [
            [self.real.x, self.real.y, self.real.z, self.real.w],
            [self.dual.x, self.dual.y, self.dual.z, self.dual.w],
        ]
    }
}

fn translation_from_dq(real: Quat, dual: Quat) -> Vec3 {
    // t = 2 * dual * conjugate(real)  (vector part)
    let rx = real.x;
    let ry = real.y;
    let rz = real.z;
    let rw = real.w;
    let dx = dual.x;
    let dy = dual.y;
    let dz = dual.z;
    let dw = dual.w;
    Vec3::new(
        2.0 * (-dw * rx + dx * rw - dy * rz + dz * ry),
        2.0 * (-dw * ry + dy * rw + dx * rz - dz * rx),
        2.0 * (-dw * rz + dz * rw - dx * ry + dy * rx),
    )
}

/// Blend joint matrices for one vertex under `mode`.
pub fn blend_skin_matrix(
    mats: &[Mat4],
    joints: [u16; 4],
    weights: [f32; 4],
    mode: SkinningMode,
) -> Mat4 {
    match mode {
        SkinningMode::LinearBlend => blend_lbs(mats, joints, weights),
        SkinningMode::DualQuat => blend_dqs(mats, joints, weights),
    }
}

pub fn blend_skin_point(
    mats: &[Mat4],
    joints: [u16; 4],
    weights: [f32; 4],
    p: Vec3,
    mode: SkinningMode,
) -> Vec3 {
    blend_skin_matrix(mats, joints, weights, mode).transform_point3(p)
}

pub fn blend_skin_vector(
    mats: &[Mat4],
    joints: [u16; 4],
    weights: [f32; 4],
    v: Vec3,
    mode: SkinningMode,
) -> Vec3 {
    blend_skin_matrix(mats, joints, weights, mode).transform_vector3(v)
}

/// Skin a mesh vertex position (bind-space) with optional joints/weights.
pub fn skin_mesh_point(mesh: &Mesh, mats: &[Mat4], i: usize, p: Vec3, mode: SkinningMode) -> Vec3 {
    let (Some(joints), Some(weights)) = (mesh.joints.first(), mesh.weights.first()) else {
        return p;
    };
    if i >= joints.len() || i >= weights.len() {
        return p;
    }
    blend_skin_point(mats, joints[i], weights[i], p, mode)
}

pub fn skin_mesh_matrix(mesh: &Mesh, mats: &[Mat4], i: usize, mode: SkinningMode) -> Mat4 {
    let (Some(joints), Some(weights)) = (mesh.joints.first(), mesh.weights.first()) else {
        return Mat4::IDENTITY;
    };
    if i >= joints.len() || i >= weights.len() {
        return Mat4::IDENTITY;
    }
    blend_skin_matrix(mats, joints[i], weights[i], mode)
}

fn blend_lbs(mats: &[Mat4], joints: [u16; 4], weights: [f32; 4]) -> Mat4 {
    let mut out = Mat4::ZERO;
    let mut w_sum = 0.0f32;
    for k in 0..4 {
        let idx = joints[k] as usize;
        let w = weights[k];
        if w <= 0.0 || idx >= mats.len() {
            continue;
        }
        out += mats[idx] * w;
        w_sum += w;
    }
    if w_sum < 1e-6 {
        Mat4::IDENTITY
    } else if (w_sum - 1.0).abs() > 1e-3 {
        out * (1.0 / w_sum)
    } else {
        out
    }
}

fn blend_dqs(mats: &[Mat4], joints: [u16; 4], weights: [f32; 4]) -> Mat4 {
    let mut blended = DualQuat {
        real: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
    };
    let mut w_sum = 0.0f32;
    let mut has_ref = false;
    let mut ref_real = Quat::IDENTITY;

    for k in 0..4 {
        let idx = joints[k] as usize;
        let w = weights[k];
        if w <= 0.0 || idx >= mats.len() {
            continue;
        }
        let mut dq = DualQuat::from_mat4(mats[idx]);
        if !has_ref {
            ref_real = dq.real;
            has_ref = true;
        } else if ref_real.dot(dq.real) < 0.0 {
            dq = dq.negate();
        }
        blended = blended.add(dq.mul_scalar(w));
        w_sum += w;
    }

    if w_sum < 1e-6 {
        return Mat4::IDENTITY;
    }
    if (w_sum - 1.0).abs() > 1e-3 {
        blended = blended.mul_scalar(1.0 / w_sum);
    }
    blended.normalize().to_mat4()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn dq_roundtrip_rigid() {
        let q = Quat::from_rotation_y(0.7);
        let t = Vec3::new(1.0, 2.0, -0.5);
        let m = Mat4::from_rotation_translation(q, t);
        let back = DualQuat::from_mat4(m).to_mat4();
        let p = Vec3::new(0.3, -0.1, 0.8);
        let a = m.transform_point3(p);
        let b = back.transform_point3(p);
        assert!((a - b).length() < 1e-4, "{a:?} vs {b:?}");
    }

    #[test]
    fn single_bone_dqs_matches_lbs() {
        let q = Quat::from_rotation_x(0.4);
        let m = Mat4::from_rotation_translation(q, Vec3::new(0.0, 1.0, 0.0));
        let mats = [m];
        let joints = [0u16, 0, 0, 0];
        let weights = [1.0f32, 0.0, 0.0, 0.0];
        let lbs = blend_lbs(&mats, joints, weights);
        let dqs = blend_dqs(&mats, joints, weights);
        let p = Vec3::new(1.0, 0.0, 0.0);
        assert!((lbs.transform_point3(p) - dqs.transform_point3(p)).length() < 1e-4);
    }
}
