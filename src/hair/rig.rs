use super::mesh::{expand_guides, skin_layout};
use super::params::HairGuide;
use crate::node::{Node, Transform};
use crate::scene::Scene;
use crate::skin::Skin;
use crate::store::Handle;
use glam::{Mat3, Mat4, Quat, Vec3};

#[derive(Clone)]
pub struct HairChain {
    pub rest: Vec<Vec3>,
    pub bone_base: u16,
    pub guide: usize,
    pub mirrored: bool,
}

impl HairChain {
    pub fn n_bones(&self) -> u16 {
        self.rest.len().saturating_sub(1) as u16
    }
}

#[derive(Clone)]
pub struct HairRig {
    pub chains: Vec<HairChain>,
    pub joint_locals: Vec<Transform>,
    pub inverse_bind: Vec<Mat4>,
}

/// Rest-pose bone chains + inverse bind. Same order as mesh joint indices.
pub fn generate_hair_rig(guides: &[HairGuide], lift: f32) -> HairRig {
    let expanded = expand_guides(guides);
    let layout = skin_layout(&expanded);
    let mut chains = Vec::new();
    let mut joint_locals = Vec::new();
    let mut inverse_bind = Vec::new();
    for (e, slot) in expanded.iter().zip(layout.iter()) {
        let Some((bone_base, _)) = *slot else {
            continue;
        };
        let rest: Vec<Vec3> = e
            .guide
            .points
            .iter()
            .map(|p| p.pos + p.normal * lift)
            .collect();
        if rest.len() < 2 {
            continue;
        }
        let frames = chain_frames(&rest);
        for frame in &frames {
            joint_locals.push(transform_from_mat4(*frame));
            inverse_bind.push(frame.inverse());
        }
        chains.push(HairChain {
            rest,
            bone_base,
            guide: e.src,
            mirrored: e.mirrored,
        });
    }
    HairRig {
        chains,
        joint_locals,
        inverse_bind,
    }
}

pub fn spawn_hair_joints(scene: &mut Scene, prefix: &str, rig: &HairRig) -> (Vec<Handle<Node>>, Option<Handle<Skin>>) {
    if rig.joint_locals.is_empty() {
        return (Vec::new(), None);
    }
    let mut joints = Vec::with_capacity(rig.joint_locals.len());
    for (i, local) in rig.joint_locals.iter().enumerate() {
        joints.push(scene.nodes.insert(Node {
            id: Node::new_id(),
            name: format!("{prefix}_{i}"),
            parent: None,
            local: *local,
            mesh: None,
            material: None,
            skin: None,
            visible: false,
        }));
    }
    let skin = scene.skins.insert(Skin {
        joints: joints.clone(),
        inverse_bind: rig.inverse_bind.clone(),
    });
    (joints, Some(skin))
}

fn chain_frames(pts: &[Vec3]) -> Vec<Mat4> {
    let n = pts.len().saturating_sub(1);
    if n == 0 {
        return Vec::new();
    }
    let d = pts[1] - pts[0];
    let hint = {
        let h = d.cross(Vec3::Y);
        if h.length_squared() > 1e-10 {
            h
        } else {
            Vec3::X
        }
    };
    (0..n)
        .map(|i| bone_frame(pts[i], pts[i + 1], hint))
        .collect()
}

fn bone_frame(a: Vec3, b: Vec3, hint: Vec3) -> Mat4 {
    let y = (b - a).normalize_or_zero();
    if y.length_squared() < 1e-10 {
        return Mat4::from_translation(a);
    }
    let mut x = y.cross(hint).normalize_or_zero();
    if x.length_squared() < 1e-10 {
        x = y.any_orthonormal_vector();
    }
    let z = x.cross(y).normalize_or_zero();
    Mat4::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), a.extend(1.0))
}

fn transform_from_mat4(m: Mat4) -> Transform {
    let translation = m.w_axis.truncate();
    let x = m.x_axis.truncate().normalize_or_zero();
    let y = m.y_axis.truncate().normalize_or_zero();
    let z = m.z_axis.truncate().normalize_or_zero();
    let rotation = if x.length_squared() > 1e-10
        && y.length_squared() > 1e-10
        && z.length_squared() > 1e-10
    {
        Quat::from_mat3(&Mat3::from_cols(x, y, z)).normalize()
    } else {
        Quat::IDENTITY
    };
    Transform {
        translation,
        rotation,
        scale: Vec3::ONE,
    }
}
