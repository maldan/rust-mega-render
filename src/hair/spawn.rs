use super::mesh::{apply_hair_mesh, generate_hair_mesh};
use super::params::{fill_pairs_of, HairGuide, HairParams, HairShape};
use super::rig::{generate_hair_rig, spawn_hair_joints, HairChain};
use super::texture::{bake_hair_maps, HairLayerBake};
use crate::material::{Material, ShadingModel};
use crate::mesh::Mesh;
use crate::node::{Node, Transform};
use crate::scene::Scene;
use crate::skin::Skin;
use crate::store::Handle;
use crate::texture::Texture;

pub struct HairLayerDesc {
    pub name: String,
    pub guides: Vec<HairGuide>,
    pub params: HairParams,
    pub fiber_gap: f32,
    pub visible: bool,
}

pub struct HairDesc {
    pub layers: Vec<HairLayerDesc>,
}

pub struct HairStack {
    pub back_node: Handle<Node>,
    pub back_mesh: Handle<Mesh>,
    pub node: Handle<Node>,
    pub mesh: Handle<Mesh>,
}

pub struct HairLayerInstance {
    pub material: Handle<Material>,
    pub albedo: Handle<Texture>,
    pub roughness: Handle<Texture>,
    pub normal: Handle<Texture>,
    pub stacks: Vec<HairStack>,
    pub skin: Option<Handle<Skin>>,
    pub joints: Vec<Handle<Node>>,
    pub chains: Vec<HairChain>,
}

pub struct HairInstance {
    pub layers: Vec<HairLayerInstance>,
}

/// Bake maps, build meshes, spawn rest-pose joints + skin. No simulation.
pub fn spawn_hair(scene: &mut Scene, desc: &HairDesc) -> HairInstance {
    let fiber = desc
        .layers
        .first()
        .map(|l| l.params.clone())
        .unwrap_or_default();
    let bake_layers: Vec<HairLayerBake> = desc
        .layers
        .iter()
        .map(|l| HairLayerBake {
            gap: l.fiber_gap,
            color_stops: l.params.color_stops.clone(),
        })
        .collect();
    let maps = bake_hair_maps(&fiber, &bake_layers);

    let n = desc.layers.len();
    let counts: Vec<usize> = desc
        .layers
        .iter()
        .map(|l| l.params.auto_stack_count())
        .collect();

    let mut mats = Vec::with_capacity(n);
    let mut texs = Vec::with_capacity(n);
    for (i, layer) in desc.layers.iter().enumerate() {
        let albedo = maps.albedos.get(i).cloned().unwrap_or_default();
        let (mat, grad, rough, normal) = spawn_maps(
            scene,
            maps.width,
            maps.height,
            albedo,
            maps.roughness.clone(),
            maps.normal.clone(),
            &layer.params,
        );
        mats.push(mat);
        texs.push((grad, rough, normal));
    }

    let mut backs = Vec::with_capacity(n);
    for li in 0..n {
        let mut row = Vec::new();
        for ai in 0..counts[li] {
            row.push(spawn_mesh_node(
                scene,
                &format!("hair_g{li}_a{ai}_back"),
                mats[li],
            ));
        }
        backs.push(row);
    }

    let mut layers = Vec::with_capacity(n);
    for (li, layer) in desc.layers.iter().enumerate() {
        let mut stacks = Vec::new();
        for ai in 0..counts[li] {
            let (back_node, back_mesh) = backs[li][ai];
            let (node, mesh) = spawn_mesh_node(
                scene,
                &format!("hair_g{li}_a{ai}_front"),
                mats[li],
            );
            stacks.push(HairStack {
                back_node,
                back_mesh,
                node,
                mesh,
            });
        }

        let fills = fill_pairs_of(&layer.guides);
        let has = !layer.guides.is_empty();
        for (ai, slot) in stacks.iter().enumerate() {
            let show = layer.visible && has;
            if let Some(n) = scene.nodes.get_mut(slot.node) {
                n.visible = show;
            }
            if !has {
                continue;
            }
            let (front, back) = generate_hair_mesh(&layer.guides, &fills, &layer.params, ai as u32);
            if let Some(mesh) = scene.meshes.get_mut(slot.mesh) {
                apply_hair_mesh(mesh, front);
            }
            match back {
                Some(bufs) => {
                    if let Some(n) = scene.nodes.get_mut(slot.back_node) {
                        n.visible = show;
                    }
                    if let Some(m) = scene.meshes.get_mut(slot.back_mesh) {
                        apply_hair_mesh(m, bufs);
                    }
                }
                None => {
                    if let Some(n) = scene.nodes.get_mut(slot.back_node) {
                        n.visible = false;
                    }
                }
            }
        }

        let rig = generate_hair_rig(&layer.guides, layer.guides.first().map(|g| g.lift).unwrap_or(0.0));
        let (joints, skin) = spawn_hair_joints(
            scene,
            &format!(
                "hair_j{li}{}",
                if layer.name.is_empty() {
                    String::new()
                } else {
                    format!("_{}", layer.name)
                }
            ),
            &rig,
        );
        if let Some(h) = skin {
            for s in &stacks {
                for node in [s.back_node, s.node] {
                    if let Some(n) = scene.nodes.get_mut(node) {
                        n.skin = Some(h);
                    }
                }
            }
        }

        let (grad, rough, normal) = texs[li];
        layers.push(HairLayerInstance {
            material: mats[li],
            albedo: grad,
            roughness: rough,
            normal,
            stacks,
            skin,
            joints,
            chains: rig.chains,
        });
    }

    HairInstance { layers }
}

fn spawn_maps(
    scene: &mut Scene,
    w: u32,
    h: u32,
    albedo: Vec<u8>,
    roughness: Vec<u8>,
    normal: Vec<u8>,
    params: &HairParams,
) -> (Handle<Material>, Handle<Texture>, Handle<Texture>, Handle<Texture>) {
    let n = (w * h * 4) as usize;
    let albedo = if albedo.len() == n {
        albedo
    } else {
        vec![40, 16, 8, 255].repeat((w * h) as usize)
    };
    let grad = scene.textures.insert(Texture {
        width: w,
        height: h,
        rgba: albedo,
        version: 1,
        srgb: true,
        dirty: None,
        gpu_resident: false,
    });
    let rough = scene.textures.insert(Texture {
        width: w,
        height: h,
        rgba: roughness,
        version: 1,
        srgb: false,
        dirty: None,
        gpu_resident: false,
    });
    let nrm = scene.textures.insert(Texture {
        width: w,
        height: h,
        rgba: normal,
        version: 1,
        srgb: false,
        dirty: None,
        gpu_resident: false,
    });
    let mut mat = Material::new([1.0, 1.0, 1.0, 1.0], 0.0, params.roughness);
    mat.albedo_map = Some(grad);
    mat.metallic_roughness_map = Some(rough);
    mat.normal_map = Some(nrm);
    mat.alpha_cutoff = match params.shape {
        HairShape::Ribbon => params.cutout.clamp(0.02, 1.0),
        HairShape::Tube => 0.0,
    };
    let mut shading = params.hair_shading;
    shading.tip_fade = params.tip_fade.clamp(0.0, 1.0);
    shading.soft_blend = params.soft_blend;
    shading.cutout_fringe = params.cutout_fringe.clamp(0.0, 1.0);
    mat.shading_model = ShadingModel::Hair(shading);
    let mat = scene.materials.insert(mat);
    (mat, grad, rough, nrm)
}

fn spawn_mesh_node(
    scene: &mut Scene,
    name: &str,
    mat: Handle<Material>,
) -> (Handle<Node>, Handle<Mesh>) {
    let mesh = scene.meshes.insert(Mesh::new(
        vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        vec![0, 1, 2],
    ));
    let node = scene.nodes.insert(Node {
        name: name.into(),
        parent: None,
        local: Transform::default(),
        mesh: Some(mesh),
        material: Some(mat),
        skin: None,
        visible: true,
    });
    (node, mesh)
}
