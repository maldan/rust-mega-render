use super::material::Material;
use super::mesh::Mesh;
use super::skin::Skin;
use super::store::Handle;
use glam::{Mat4, Quat, Vec3};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NODE_ID_SEQ: AtomicU64 = AtomicU64::new(1);

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
    /// Stable identity, distinct from the (renamable, non-unique) display `name`.
    /// Not interned/deduped: multiple nodes may legitimately share an id when
    /// spawned from the same source template (e.g. two instances of one glTF).
    pub id: String,
    pub name: String,
    pub parent: Option<Handle<Node>>,
    pub local: Transform,
    pub mesh: Option<Handle<Mesh>>,
    pub material: Option<Handle<Material>>,
    pub skin: Option<Handle<Skin>>,
    pub visible: bool,
}

impl Node {
    /// Unique id for runtime-created nodes (not loaded from a file).
    pub fn new_id() -> String {
        let n = NODE_ID_SEQ.fetch_add(1, Ordering::Relaxed);
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("node/{t:x}-{n:x}")
    }
}
