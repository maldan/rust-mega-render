//! Hair card bake, guide meshing, rest-pose skin rig, and `spawn_hair`.

mod curve;
mod mesh;
mod params;
mod rig;
mod spawn;
mod texture;

pub const MAX_HAIR_BONES: u16 = 255;

pub use curve::{HairColorStop, HairCurve, HairCurvePoint, HairCurvePreset};
pub use mesh::{
    apply_hair_mesh, apply_hair_mesh_rigid, clear_hair_mesh, generate_hair_mesh, hair_mesh_has_geo,
    HairCardMeshes, HairMeshBuffers, HairMeshes,
};
pub use params::{
    fill_pairs_of, HairGuide, HairGuidePoint, HairParams, HairShape, HairStyle, LayerRandom,
    RandRange,
};
pub use rig::{generate_hair_rig, spawn_hair_joints, HairChain, HairRig};
pub use spawn::{spawn_hair, HairDesc, HairInstance, HairLayerDesc, HairLayerInstance, HairStack};
pub use texture::{bake_hair_maps, HairLayerBake, HairMaps};

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn guide(is_static: bool, x: f32) -> HairGuide {
        HairGuide {
            points: vec![
                HairGuidePoint {
                    pos: Vec3::new(x, 0.0, 0.0),
                    normal: Vec3::Y,
                },
                HairGuidePoint {
                    pos: Vec3::new(x, 0.1, 0.0),
                    normal: Vec3::Y,
                },
                HairGuidePoint {
                    pos: Vec3::new(x, 0.2, 0.0),
                    normal: Vec3::Y,
                },
            ],
            mirror_x: false,
            lift: 0.06,
            width: 0.02,
            fill_with: None,
            is_static,
        }
    }

    #[test]
    fn static_guide_skips_bones() {
        let guides = vec![guide(true, 0.0), guide(false, 0.1)];
        let rig = generate_hair_rig(&guides, 0.06);
        assert_eq!(rig.chains.len(), 1);
        assert_eq!(rig.chains[0].guide, 1);
        assert_eq!(rig.chains[0].bone_base, 0);
        assert_eq!(rig.joint_locals.len(), 2);
        let meshes = generate_hair_mesh(&guides, &[], &HairParams::default(), 0);
        assert!(meshes.rigid.has_geo());
        assert!(meshes.skinned.has_geo());
        let max_j = meshes
            .skinned
            .front
            .4
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(0);
        assert!(max_j < 2);
        assert!(meshes.rigid.front.5.iter().flatten().all(|&w| w.abs() < 1e-6));
    }

    #[test]
    fn static_only_is_rigid_mesh() {
        let guides = vec![guide(true, 0.0)];
        let meshes = generate_hair_mesh(&guides, &[], &HairParams::default(), 0);
        assert!(meshes.rigid.has_geo());
        assert!(!meshes.skinned.has_geo());
        let rig = generate_hair_rig(&guides, 0.06);
        assert!(rig.chains.is_empty());
    }
}
