use super::material::Material;
use super::mesh::Mesh;
use super::skin::Skin;
use super::store::Handle;
use glam::{Mat4, Quat, Vec3};

#[derive(Clone, Copy)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            translation: t,
            ..Default::default()
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

pub struct Node {
    pub name: String,
    pub parent: Option<Handle<Node>>,
    pub local: Transform,
    pub mesh: Option<Handle<Mesh>>,
    pub material: Option<Handle<Material>>,
    pub skin: Option<Handle<Skin>>,
    pub visible: bool,
}
