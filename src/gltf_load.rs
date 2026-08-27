use super::scene::PendingLoad;
use super::store::Store;
use super::{
    AnimChannel, AnimPath, AnimValues, AnimationClip, Animator, Handle, Material, Mesh, Node,
    Scene, Skin, Texture, Transform,
};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::thread;

const FLIP_Z: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
]);

impl Scene {
    /// Spawn background glTF load; result is merged on `poll_loads`.
    pub fn load_gltf_async(
        &mut self,
        path: impl AsRef<Path>,
        parent: Option<Handle<Node>>,
        on_ready: impl FnOnce(&mut Scene, Handle<Node>) + Send + 'static,
    ) {
        let path = path.as_ref().to_path_buf();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut tmp = Scene::new();
            let _ = tx.send(load_gltf(&mut tmp, &path, None).map(|root| (tmp, root)));
        });
        self.pending_loads.push(PendingLoad::Gltf {
            rx,
            parent,
            on_ready: Box::new(on_ready),
        });
    }
}

/// Load glTF/GLB into the scene under `parent`. Returns the import root node.
pub fn load_gltf(
    scene: &mut Scene,
    path: impl AsRef<Path>,
    parent: Option<Handle<Node>>,
) -> Result<Handle<Node>, String> {
    let (doc, buffers, images) = gltf::import(path.as_ref()).map_err(|e| e.to_string())?;

    let mut linear = vec![false; images.len()];
    for mat in doc.materials() {
        if let Some(n) = mat.normal_texture() {
            linear[n.texture().source().index()] = true;
        }
        if let Some(mr) = mat.pbr_metallic_roughness().metallic_roughness_texture() {
            linear[mr.texture().source().index()] = true;
        }
        if let Some(occ) = mat.occlusion_texture() {
            linear[occ.texture().source().index()] = true;
        }
    }

    let mut tex_handles = Vec::with_capacity(images.len());
    for (i, img) in images.iter().enumerate() {
        let mut tex = image_to_texture(img, gltf_image_id(path.as_ref(), &doc, i))?;
        tex.srgb = !linear.get(i).copied().unwrap_or(false);
        tex_handles.push(scene.textures.insert(tex));
    }

    let mut mat_handles = Vec::with_capacity(doc.materials().len());
    for mat in doc.materials() {
        mat_handles.push(scene.materials.insert(convert_material(&mat, &tex_handles)));
    }
    let default_mat = scene.materials.insert(Material::default());

    let root = scene.nodes.insert(Node {
        id: Node::new_id(),
        name: path
            .as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "gltf".into()),
        parent,
        local: Transform::default(),
        mesh: None,
        material: None,
        skin: None,
        visible: true,
    });

    let mut node_map = vec![None; doc.nodes().len()];
    let roots: Vec<_> = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .map(|s| s.nodes().collect())
        .unwrap_or_else(|| doc.nodes().collect());

    for node in &roots {
        spawn_hierarchy(scene, node, Some(root), &mut node_map);
    }

    let skin_handles: Vec<_> = doc
        .skins()
        .map(|skin| {
            let joints: Vec<_> = skin
                .joints()
                .filter_map(|n| node_map.get(n.index()).copied().flatten())
                .collect();
            let reader = skin.reader(|b| Some(buffers.get(b.index())?.0.as_slice()));
            let inverse_bind = if let Some(iter) = reader.read_inverse_bind_matrices() {
                iter.map(|m| FLIP_Z * Mat4::from_cols_array_2d(&m) * FLIP_Z)
                    .collect()
            } else {
                vec![Mat4::IDENTITY; joints.len()]
            };
            scene.skins.insert(Skin {
                joints,
                inverse_bind,
            })
        })
        .collect();

    for node in &roots {
        attach_meshes(
            scene,
            node,
            &node_map,
            &buffers,
            &mat_handles,
            default_mat,
            &skin_handles,
        )?;
    }

    let mut first_clip: Option<Handle<AnimationClip>> = None;
    let mut first_moving: Option<Handle<AnimationClip>> = None;
    for anim in doc.animations() {
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for channel in anim.channels() {
            let Some(target) = node_map
                .get(channel.target().node().index())
                .copied()
                .flatten()
            else {
                continue;
            };
            let reader = channel.reader(|b| Some(buffers.get(b.index())?.0.as_slice()));
            let Some(inputs) = reader.read_inputs() else {
                continue;
            };
            let times: Vec<f32> = inputs.collect();
            if let Some(&t) = times.last() {
                duration = duration.max(t);
            }
            let step = matches!(
                channel.sampler().interpolation(),
                gltf::animation::Interpolation::Step
            );
            let cubic = matches!(
                channel.sampler().interpolation(),
                gltf::animation::Interpolation::CubicSpline
            );
            let Some(outputs) = reader.read_outputs() else {
                continue;
            };
            use gltf::animation::util::ReadOutputs;
            let (path, values) = match outputs {
                ReadOutputs::Translations(iter) => {
                    let mut vals: Vec<Vec3> = iter.map(convert_translation).collect();
                    if cubic {
                        vals = cubic_values(vals);
                    }
                    (AnimPath::Translation, AnimValues::Vec3(vals))
                }
                ReadOutputs::Scales(iter) => {
                    let mut vals: Vec<Vec3> = iter.map(|s| Vec3::from_array(s)).collect();
                    if cubic {
                        vals = cubic_values(vals);
                    }
                    (AnimPath::Scale, AnimValues::Vec3(vals))
                }
                ReadOutputs::Rotations(rots) => {
                    let mut vals: Vec<Quat> = rots.into_f32().map(convert_rotation).collect();
                    if cubic {
                        vals = cubic_values(vals);
                    }
                    (AnimPath::Rotation, AnimValues::Quat(vals))
                }
                _ => continue,
            };
            channels.push(AnimChannel {
                target,
                path,
                times,
                values,
                step,
            });
        }
        if channels.is_empty() {
            continue;
        }
        let moving = clip_has_motion(&channels);
        let name = anim.name().unwrap_or("anim").to_string();
        let clip = scene.animations.insert(AnimationClip {
            name: name.clone(),
            duration,
            channels,
        });
        if first_clip.is_none() {
            first_clip = Some(clip);
        }
        if moving && first_moving.is_none() {
            first_moving = Some(clip);
            eprintln!("gltf anim play: {name} ({duration:.2}s)");
        }
    }
    if let Some(clip) = first_moving.or(first_clip) {
        scene.animators.push(Animator::new(clip));
    }

    fit_root(scene, root, 2.0);
    Ok(root)
}

/// Merge a temp scene (from background `load_gltf`) into `dst`.
pub(crate) fn absorb_gltf(
    dst: &mut Scene,
    mut src: Scene,
    src_root: Handle<Node>,
    parent: Option<Handle<Node>>,
) -> Handle<Node> {
    let mut tex_map = HashMap::new();
    for (h, tex) in take_textures(&mut src.textures) {
        tex_map.insert(h.key(), dst.textures.insert(tex));
    }

    let mut mat_map = HashMap::new();
    for (h, mut mat) in take_all(&mut src.materials) {
        mat.maps.remap_textures(|t| tex_map.get(&t.key()).copied());
        mat_map.insert(h.key(), dst.materials.insert(mat));
    }

    let mut mesh_map = HashMap::new();
    for (h, mesh) in take_all(&mut src.meshes) {
        mesh_map.insert(h.key(), dst.meshes.insert(mesh));
    }

    let src_nodes = take_all(&mut src.nodes);
    let mut node_map = HashMap::new();
    for (h, _) in &src_nodes {
        node_map.insert(
            h.key(),
            dst.nodes.insert(Node {
                id: Node::new_id(),
                name: String::new(),
                parent: None,
                local: Transform::default(),
                mesh: None,
                material: None,
                skin: None,
                visible: true,
            }),
        );
    }

    let mut pending_skins = Vec::new();
    for (h, n) in src_nodes {
        let Some(&new_h) = node_map.get(&h.key()) else {
            continue;
        };
        let new_parent = if h.key() == src_root.key() {
            parent
        } else {
            n.parent.and_then(|p| node_map.get(&p.key()).copied())
        };
        if let Some(skin) = n.skin {
            pending_skins.push((new_h, skin));
        }
        if let Some(node) = dst.nodes.get_mut(new_h) {
            *node = Node {
                id: n.id,
                name: n.name,
                parent: new_parent,
                local: n.local,
                mesh: n.mesh.and_then(|m| mesh_map.get(&m.key()).copied()),
                material: n.material.and_then(|m| mat_map.get(&m.key()).copied()),
                skin: None,
                visible: n.visible,
            };
        }
    }

    let mut skin_map = HashMap::new();
    for (h, skin) in take_all(&mut src.skins) {
        let joints: Vec<_> = skin
            .joints
            .iter()
            .filter_map(|j| node_map.get(&j.key()).copied())
            .collect();
        skin_map.insert(
            h.key(),
            dst.skins.insert(Skin {
                joints,
                inverse_bind: skin.inverse_bind,
            }),
        );
    }
    for (new_h, old_skin) in pending_skins {
        if let Some(node) = dst.nodes.get_mut(new_h) {
            node.skin = skin_map.get(&old_skin.key()).copied();
        }
    }

    let mut anim_map = HashMap::new();
    for (h, mut clip) in take_all(&mut src.animations) {
        for ch in &mut clip.channels {
            if let Some(&t) = node_map.get(&ch.target.key()) {
                ch.target = t;
            }
        }
        anim_map.insert(h.key(), dst.animations.insert(clip));
    }
    for anim in src.animators.drain(..) {
        let Some(&clip) = anim_map.get(&anim.clip.key()) else {
            continue;
        };
        dst.animators.push(Animator {
            clip,
            time: anim.time,
            speed: anim.speed,
            playing: anim.playing,
            looping: anim.looping,
        });
    }

    *node_map.get(&src_root.key()).expect("gltf root")
}

fn take_textures(store: &mut crate::TextureStore) -> Vec<(Handle<Texture>, Texture)> {
    let handles: Vec<_> = store.iter().map(|(h, _)| h).collect();
    handles
        .into_iter()
        .filter_map(|h| store.remove(h).map(|v| (h, v)))
        .collect()
}

fn take_all<T>(store: &mut Store<T>) -> Vec<(Handle<T>, T)> {
    let handles: Vec<_> = store.iter().map(|(h, _)| h).collect();
    handles
        .into_iter()
        .filter_map(|h| store.remove(h).map(|v| (h, v)))
        .collect()
}

fn clip_has_motion(channels: &[AnimChannel]) -> bool {
    for ch in channels {
        match &ch.values {
            AnimValues::Vec3(v) => {
                if v.windows(2).any(|w| (w[0] - w[1]).length_squared() > 1e-10) {
                    return true;
                }
            }
            AnimValues::Quat(v) => {
                if v.windows(2).any(|w| (1.0 - w[0].dot(w[1]).abs()) > 1e-6) {
                    return true;
                }
            }
        }
    }
    false
}

fn cubic_values<T: Copy>(vals: Vec<T>) -> Vec<T> {
    // cubic: in-tangent, value, out-tangent per key
    vals.chunks(3).filter_map(|c| c.get(1).copied()).collect()
}

fn spawn_hierarchy(
    scene: &mut Scene,
    node: &gltf::Node,
    parent: Option<Handle<Node>>,
    node_map: &mut [Option<Handle<Node>>],
) {
    let h = scene.nodes.insert(Node {
        id: Node::new_id(),
        name: node.name().unwrap_or("node").into(),
        parent,
        local: convert_transform(node),
        mesh: None,
        material: None,
        skin: None,
        visible: true,
    });
    if let Some(slot) = node_map.get_mut(node.index()) {
        *slot = Some(h);
    }
    for child in node.children() {
        spawn_hierarchy(scene, &child, Some(h), node_map);
    }
}

fn attach_meshes(
    scene: &mut Scene,
    node: &gltf::Node,
    node_map: &[Option<Handle<Node>>],
    buffers: &[gltf::buffer::Data],
    mats: &[Handle<Material>],
    default_mat: Handle<Material>,
    skins: &[Handle<Skin>],
) -> Result<(), String> {
    let Some(h) = node_map.get(node.index()).copied().flatten() else {
        return Ok(());
    };
    let skin = node.skin().and_then(|s| skins.get(s.index()).copied());

    if let Some(mesh) = node.mesh() {
        let prims: Vec<_> = mesh.primitives().collect();
        if prims.len() == 1 {
            let (mesh_h, mat_h) = load_primitive(scene, &prims[0], buffers, mats, default_mat)?;
            if let Some(n) = scene.nodes.get_mut(h) {
                n.mesh = Some(mesh_h);
                n.material = Some(mat_h);
                n.skin = skin;
            }
        } else {
            for (i, prim) in prims.iter().enumerate() {
                let (mesh_h, mat_h) = load_primitive(scene, prim, buffers, mats, default_mat)?;
                scene.nodes.insert(Node {
                    id: Node::new_id(),
                    name: format!("{}_prim{i}", mesh.name().unwrap_or("mesh")),
                    parent: Some(h),
                    local: Transform::default(),
                    mesh: Some(mesh_h),
                    material: Some(mat_h),
                    skin,
                    visible: true,
                });
            }
        }
    }

    for child in node.children() {
        attach_meshes(scene, &child, node_map, buffers, mats, default_mat, skins)?;
    }
    Ok(())
}

/// Scale so the tallest axis ≈ `target_size`, then sit on Y=0.
fn fit_root(scene: &mut Scene, root: Handle<Node>, target_size: f32) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;
    for (h, node) in scene.nodes.iter() {
        let Some(mesh_h) = node.mesh else { continue };
        if !is_under(scene, h, root) {
            continue;
        }
        let Some(mesh) = scene.meshes.get(mesh_h) else { continue };
        let world = scene.world_matrix(h);
        for p in &mesh.positions {
            let wp = world.transform_point3(Vec3::from_array(*p));
            min = min.min(wp);
            max = max.max(wp);
            any = true;
        }
    }
    if !any {
        return;
    }
    let size = (max - min).max_element().max(1e-4);
    let scale = target_size / size;
    let center_xz = Vec3::new((min.x + max.x) * 0.5, min.y, (min.z + max.z) * 0.5);
    if let Some(n) = scene.nodes.get_mut(root) {
        n.local.scale = Vec3::splat(scale);
        n.local.translation = -center_xz * scale;
    }
}

fn is_under(scene: &Scene, mut node: Handle<Node>, root: Handle<Node>) -> bool {
    loop {
        if node.key() == root.key() {
            return true;
        }
        let Some(n) = scene.nodes.get(node) else {
            return false;
        };
        let Some(p) = n.parent else {
            return false;
        };
        node = p;
    }
}

fn load_primitive(
    scene: &mut Scene,
    prim: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    mats: &[Handle<Material>],
    default_mat: Handle<Material>,
) -> Result<(Handle<Mesh>, Handle<Material>), String> {
    let reader = prim.reader(|b| Some(buffers.get(b.index())?.0.as_slice()));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or("primitive missing POSITION")?
        .map(|p| [p[0], p[1], -p[2]])
        .collect();
    let normals: Vec<[f32; 3]> = if let Some(iter) = reader.read_normals() {
        iter.map(|n| [n[0], n[1], -n[2]]).collect()
    } else {
        vec![[0.0, 1.0, 0.0]; positions.len()]
    };
    let uvs: Vec<[f32; 2]> = if let Some(iter) = reader.read_tex_coords(0).map(|t| t.into_f32()) {
        iter.collect()
    } else {
        vec![[0.0, 0.0]; positions.len()]
    };
    let mut indices: Vec<u32> = match reader.read_indices() {
        Some(idx) => idx.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    for tri in indices.chunks_exact_mut(3) {
        tri.swap(1, 2);
    }

    let joints = reader.read_joints(0).map(|j| {
        j.into_u16()
            .map(|v| [v[0], v[1], v[2], v[3]])
            .collect::<Vec<_>>()
    });
    let weights = reader.read_weights(0).map(|w| {
        w.into_f32()
            .map(|v| [v[0], v[1], v[2], v[3]])
            .collect::<Vec<_>>()
    });

    let mat = prim
        .material()
        .index()
        .and_then(|i| mats.get(i).copied())
        .unwrap_or(default_mat);

    let mut mesh = Mesh::new(positions, normals, uvs, indices);
    if let Some(j) = joints {
        mesh.joints.push(j);
    }
    if let Some(w) = weights {
        mesh.weights.push(w);
    }
    if let Some(iter) = reader.read_colors(0) {
        mesh.colors
            .push(iter.into_rgba_f32().map(|c| [c[0], c[1], c[2], c[3]]).collect());
    }

    Ok((scene.meshes.insert(mesh), mat))
}

fn convert_material(mat: &gltf::Material, textures: &[Handle<Texture>]) -> Material {
    let pbr = mat.pbr_metallic_roughness();
    let base = pbr.base_color_factor();
    let mut m = Material::new(base, pbr.metallic_factor(), pbr.roughness_factor());
    if let Some(info) = pbr.base_color_texture() {
        let i = info.texture().source().index();
        if let Some(&tex) = textures.get(i) {
            m.maps = crate::MaterialMaps::Single {
                albedo: Some(tex),
                normal: None,
                metallic_roughness: None,
            };
        }
    }
    if let Some(info) = mat.normal_texture() {
        let i = info.texture().source().index();
        if let Some(&tex) = textures.get(i) {
            match &mut m.maps {
                crate::MaterialMaps::Single { normal, .. } => *normal = Some(tex),
                _ => {}
            }
        }
    }
    if let Some(info) = pbr.metallic_roughness_texture() {
        let i = info.texture().source().index();
        if let Some(&tex) = textures.get(i) {
            match &mut m.maps {
                crate::MaterialMaps::Single {
                    metallic_roughness, ..
                } => {
                    *metallic_roughness = Some(tex);
                }
                _ => {}
            }
        }
    }
    m
}

fn convert_transform(node: &gltf::Node) -> Transform {
    let (t, r, s) = node.transform().decomposed();
    let (scale, rotation, translation) = convert_trs(t, r, s);
    Transform {
        translation,
        rotation,
        scale,
    }
}

fn convert_trs(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> (Vec3, Quat, Vec3) {
    let m = Mat4::from_scale_rotation_translation(
        Vec3::from_array(s),
        Quat::from_array(r),
        Vec3::from_array(t),
    );
    (FLIP_Z * m * FLIP_Z).to_scale_rotation_translation()
}

fn convert_translation(t: [f32; 3]) -> Vec3 {
    Vec3::new(t[0], t[1], -t[2])
}

fn convert_rotation(r: [f32; 4]) -> Quat {
    convert_trs([0.0, 0.0, 0.0], r, [1.0, 1.0, 1.0]).1
}

fn gltf_image_id(path: &Path, doc: &gltf::Document, index: usize) -> String {
    let base = path.to_string_lossy().replace('\\', "/");
    let name = doc.images().nth(index).and_then(|im| im.name());
    match name {
        Some(n) if !n.is_empty() => format!("{base}/images/{index}:{n}"),
        _ => format!("{base}/images/{index}"),
    }
}

fn image_to_texture(img: &gltf::image::Data, id: String) -> Result<Texture, String> {
    use gltf::image::Format;
    let rgba = match img.format {
        Format::R8 => img.pixels.iter().flat_map(|&r| [r, r, r, 255]).collect(),
        Format::R8G8 => img
            .pixels
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        Format::R8G8B8 => img
            .pixels
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        Format::R8G8B8A8 => img.pixels.clone(),
        other => return Err(format!("unsupported glTF image format: {other:?}")),
    };
    Ok(Texture {
        id,
        width: img.width,
        height: img.height,
        rgba,
        version: 1,
        srgb: true,
        dirty: None,
        gpu_resident: false,
    })
}
