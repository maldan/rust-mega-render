use super::mesh::{
    apply_hair_mesh, apply_hair_mesh_rigid, clear_hair_mesh, generate_hair_mesh, HairCardMeshes,
};
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
    pub static_back_node: Handle<Node>,
    pub static_back_mesh: Handle<Mesh>,
    pub static_node: Handle<Node>,
    pub static_mesh: Handle<Mesh>,
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

    // Draw order: skinned backs, rigid backs, skinned fronts, rigid fronts.
    let mut skinned_backs = Vec::with_capacity(n);
    for li in 0..n {
        let mut row = Vec::new();
        for ai in 0..counts[li] {
            row.push(spawn_mesh_node(
                scene,
                &format!("hair_g{li}_a{ai}_back"),
                mats[li],
            ));
        }
        skinned_backs.push(row);
    }
    let mut rigid_backs = Vec::with_capacity(n);
    for li in 0..n {
        let mut row = Vec::new();
        for ai in 0..counts[li] {
            row.push(spawn_mesh_node(
                scene,
                &format!("hair_g{li}_a{ai}_static_back"),
                mats[li],
            ));
        }
        rigid_backs.push(row);
    }
    let mut skinned_fronts = Vec::with_capacity(n);
    for li in 0..n {
        let mut row = Vec::new();
        for ai in 0..counts[li] {
            row.push(spawn_mesh_node(
                scene,
                &format!("hair_g{li}_a{ai}_front"),
                mats[li],
            ));
        }
        skinned_fronts.push(row);
    }
    let mut rigid_fronts = Vec::with_capacity(n);
    for li in 0..n {
        let mut row = Vec::new();
        for ai in 0..counts[li] {
            row.push(spawn_mesh_node(
                scene,
                &format!("hair_g{li}_a{ai}_static_front"),
                mats[li],
            ));
        }
        rigid_fronts.push(row);
    }

    let mut layers = Vec::with_capacity(n);
    for (li, layer) in desc.layers.iter().enumerate() {
        let mut stacks = Vec::new();
        for ai in 0..counts[li] {
            let (back_node, back_mesh) = skinned_backs[li][ai];
            let (static_back_node, static_back_mesh) = rigid_backs[li][ai];
            let (node, mesh) = skinned_fronts[li][ai];
            let (static_node, static_mesh) = rigid_fronts[li][ai];
            stacks.push(HairStack {
                back_node,
                back_mesh,
                node,
                mesh,
                static_back_node,
                static_back_mesh,
                static_node,
                static_mesh,
            });
        }

        let fills = fill_pairs_of(&layer.guides);
        let show = layer.visible && !layer.guides.is_empty();
        for (ai, slot) in stacks.iter().enumerate() {
            if !show {
                hide_pair(scene, slot.back_node, slot.node);
                hide_pair(scene, slot.static_back_node, slot.static_node);
                continue;
            }
            let meshes = generate_hair_mesh(&layer.guides, &fills, &layer.params, ai as u32);
            write_cards(
                scene,
                slot.back_node,
                slot.back_mesh,
                slot.node,
                slot.mesh,
                meshes.skinned,
                true,
            );
            write_cards(
                scene,
                slot.static_back_node,
                slot.static_back_mesh,
                slot.static_node,
                slot.static_mesh,
                meshes.rigid,
                false,
            );
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

fn hide_pair(scene: &mut Scene, back: Handle<Node>, front: Handle<Node>) {
    if let Some(n) = scene.nodes.get_mut(back) {
        n.visible = false;
    }
    if let Some(n) = scene.nodes.get_mut(front) {
        n.visible = false;
    }
}

fn write_cards(
    scene: &mut Scene,
    back_node: Handle<Node>,
    back_mesh: Handle<Mesh>,
    node: Handle<Node>,
    mesh: Handle<Mesh>,
    cards: HairCardMeshes,
    skinned: bool,
) {
    let show = cards.has_geo();
    if let Some(n) = scene.nodes.get_mut(node) {
        n.visible = show;
    }
    if show {
        if let Some(m) = scene.meshes.get_mut(mesh) {
            if skinned {
                apply_hair_mesh(m, cards.front);
            } else {
                apply_hair_mesh_rigid(m, cards.front);
            }
        }
    } else if let Some(m) = scene.meshes.get_mut(mesh) {
        clear_hair_mesh(m);
    }
    match cards.back {
        Some(bufs) if show => {
            if let Some(n) = scene.nodes.get_mut(back_node) {
                n.visible = true;
            }
            if let Some(m) = scene.meshes.get_mut(back_mesh) {
                if skinned {
                    apply_hair_mesh(m, bufs);
                } else {
                    apply_hair_mesh_rigid(m, bufs);
                }
            }
        }
        _ => {
            if let Some(n) = scene.nodes.get_mut(back_node) {
                n.visible = false;
            }
            if let Some(m) = scene.meshes.get_mut(back_mesh) {
                clear_hair_mesh(m);
            }
        }
    }
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
        id: Texture::new_id(),
        width: w,
        height: h,
        rgba: albedo,
        version: 1,
        srgb: true,
        dirty: None,
        gpu_resident: false,
    });
    let rough = scene.textures.insert(Texture {
        id: Texture::new_id(),
        width: w,
        height: h,
        rgba: roughness,
        version: 1,
        srgb: false,
        dirty: None,
        gpu_resident: false,
    });
    let nrm = scene.textures.insert(Texture {
        id: Texture::new_id(),
        width: w,
        height: h,
        rgba: normal,
        version: 1,
        srgb: false,
        dirty: None,
        gpu_resident: false,
    });
    let mut mat = Material::new([1.0, 1.0, 1.0, 1.0], 0.0, params.roughness);
    mat.maps = crate::MaterialMaps::Single {
        albedo: Some(grad),
        normal: Some(nrm),
        metallic_roughness: Some(rough),
    };
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
        id: Node::new_id(),
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
