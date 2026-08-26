use super::animation::{sample_quat, sample_vec3, AnimPath, AnimValues, AnimationClip, Animator};
use super::camera::Camera;
use super::debug_draw::{
    DebugDraw, LineOpts, SKELETON_FILL, SKELETON_IK_POLE_FILL, SKELETON_IK_POLE_OUTLINE,
    SKELETON_IK_TARGET_FILL, SKELETON_IK_TARGET_OUTLINE, SKELETON_JOINT, SKELETON_JOINT_OUTLINE,
    SKELETON_LINE_W, SKELETON_OUTLINE, SKELETON_OUTLINE_W, SKELETON_SEL_FILL,
    SKELETON_SEL_OUTLINE,
};
use super::hud::Hud;
use super::light::{DirectionalLight, Light};
use super::material::Material;
use super::mesh::Mesh;
use super::node::Node;
use super::skin::{Skin, SkinningMode};
use super::store::{Handle, Store};
use super::texture::TextureStore;
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, TryRecvError};

/// Highlight / overlay options for stick skeleton debug draw.
#[derive(Clone, Copy)]
pub struct SkeletonDebugOpts<'a> {
    pub selected: &'a [Handle<Node>],
    pub ik_targets: &'a [Handle<Node>],
    pub ik_poles: &'a [Handle<Node>],
    /// Draw without depth test so the mesh never hides the stick figure.
    pub overlay: bool,
}

impl Default for SkeletonDebugOpts<'static> {
    fn default() -> Self {
        Self {
            selected: &[],
            ik_targets: &[],
            ik_poles: &[],
            overlay: true,
        }
    }
}

impl SkeletonDebugOpts<'static> {
    pub const DEFAULT: Self = Self {
        selected: &[],
        ik_targets: &[],
        ik_poles: &[],
        overlay: true,
    };
}

fn key_in(list: &[Handle<Node>], joint: Handle<Node>) -> bool {
    let k = joint.key();
    list.iter().any(|h| h.key() == k)
}

fn is_ik_control(joint: Handle<Node>, opts: &SkeletonDebugOpts<'_>) -> bool {
    key_in(opts.ik_targets, joint) || key_in(opts.ik_poles, joint)
}

/// `(outline, fill, joint_px, joint_outline_px)` for a joint under `opts`.
fn skeleton_joint_style(
    joint: Handle<Node>,
    opts: &SkeletonDebugOpts<'_>,
) -> ([f32; 4], [f32; 4], f32, f32) {
    let selected = key_in(opts.selected, joint);
    let is_target = key_in(opts.ik_targets, joint);
    let is_pole = key_in(opts.ik_poles, joint);
    match (is_target, is_pole, selected) {
        (true, _, false) => (
            SKELETON_IK_TARGET_OUTLINE,
            SKELETON_IK_TARGET_FILL,
            SKELETON_JOINT * 1.35,
            SKELETON_JOINT_OUTLINE * 1.35,
        ),
        (_, true, false) => (
            SKELETON_IK_POLE_OUTLINE,
            SKELETON_IK_POLE_FILL,
            SKELETON_JOINT * 1.2,
            SKELETON_JOINT_OUTLINE * 1.2,
        ),
        (true, _, true) | (_, true, true) => (
            SKELETON_SEL_OUTLINE,
            SKELETON_SEL_FILL,
            SKELETON_JOINT * 1.4,
            SKELETON_JOINT_OUTLINE * 1.4,
        ),
        (false, false, true) => (
            SKELETON_SEL_OUTLINE,
            SKELETON_SEL_FILL,
            SKELETON_JOINT,
            SKELETON_JOINT_OUTLINE,
        ),
        (false, false, false) => (
            SKELETON_OUTLINE,
            SKELETON_FILL,
            SKELETON_JOINT,
            SKELETON_JOINT_OUTLINE,
        ),
    }
}

pub(crate) enum PendingLoad {
    Gltf {
        rx: Receiver<Result<(Scene, Handle<Node>), String>>,
        parent: Option<Handle<Node>>,
        on_ready: Box<dyn FnOnce(&mut Scene, Handle<Node>) + Send>,
    },
}

pub struct Scene {
    pub meshes: Store<Mesh>,
    pub textures: TextureStore,
    pub materials: Store<Material>,
    pub skins: Store<Skin>,
    pub animations: Store<AnimationClip>,
    pub animators: Vec<Animator>,
    pub nodes: Store<Node>,
    pub camera: Camera,
    pub lights: Vec<Light>,
    /// Flat ambient (diffuse fill). Always applied; SSGI can dim it via ambient_dim.
    pub ambient: [f32; 3],
    /// HDR clear color for the scene pass (RGBA).
    pub clear_color: [f32; 4],
    pub debug: DebugDraw,
    /// Immediate screen-space HUD (built each frame, drawn on top).
    pub hud: Hud,
    /// LBS vs dual-quaternion skinning (GPU bone palette + CPU helpers).
    pub skinning_mode: SkinningMode,
    pub(crate) pending_loads: Vec<PendingLoad>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            meshes: Store::default(),
            textures: TextureStore::default(),
            materials: Store::default(),
            skins: Store::default(),
            animations: Store::default(),
            animators: Vec::new(),
            nodes: Store::default(),
            camera: Camera::orbit(0.8, 0.45, 8.0, Vec3::new(0.0, 0.5, 0.0)),
            lights: vec![Light::Directional(DirectionalLight::default())],
            ambient: [0.03, 0.03, 0.04],
            clear_color: [0.08, 0.09, 0.12, 1.0],
            debug: DebugDraw::default(),
            hud: Hud::new(),
            skinning_mode: SkinningMode::default(),
            pending_loads: Vec::new(),
        }
    }

    /// Apply finished background loads (gltf and future asset types).
    pub fn poll_loads(&mut self) {
        let mut i = 0;
        while i < self.pending_loads.len() {
            let ready = match &self.pending_loads[i] {
                PendingLoad::Gltf { rx, .. } => match rx.try_recv() {
                    Ok(r) => Some(r),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => Some(Err("load worker disconnected".into())),
                },
            };
            let Some(result) = ready else {
                i += 1;
                continue;
            };
            let PendingLoad::Gltf {
                parent, on_ready, ..
            } = self.pending_loads.swap_remove(i);
            match result {
                Ok((src, root)) => {
                    let h = super::gltf_load::absorb_gltf(self, src, root, parent);
                    on_ready(self, h);
                }
                Err(e) => eprintln!("asset load failed: {e}"),
            }
        }
    }

    pub fn world_matrix(&self, node: Handle<Node>) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        let mut cur = Some(node);
        while let Some(h) = cur {
            let Some(n) = self.nodes.get(h) else { break };
            m = n.local.matrix() * m;
            cur = n.parent;
        }
        m
    }

    /// Build world matrices for every node once (parent before child via recursion + memo).
    pub fn world_matrices(&self) -> HashMap<(u32, u32), Mat4> {
        let mut cache = HashMap::new();
        for (h, _) in self.nodes.iter() {
            let _ = self.world_matrix_cached(h, &mut cache);
        }
        cache
    }

    fn world_matrix_cached(
        &self,
        node: Handle<Node>,
        cache: &mut HashMap<(u32, u32), Mat4>,
    ) -> Mat4 {
        if let Some(&m) = cache.get(&node.key()) {
            return m;
        }
        let Some(n) = self.nodes.get(node) else {
            return Mat4::IDENTITY;
        };
        let m = match n.parent {
            Some(p) => self.world_matrix_cached(p, cache) * n.local.matrix(),
            None => n.local.matrix(),
        };
        cache.insert(node.key(), m);
        m
    }

    /// First enabled directional light with `cast_shadows`.
    pub fn shadow_directional(&self) -> Option<&DirectionalLight> {
        self.lights.iter().find_map(|l| match l {
            Light::Directional(d) if d.enabled && d.cast_shadows => Some(d),
            _ => None,
        })
    }

    pub fn update_animations(&mut self, dt: f32) {
        enum Update {
            T(Handle<Node>, Vec3),
            R(Handle<Node>, glam::Quat),
            S(Handle<Node>, Vec3),
        }
        let mut updates = Vec::new();
        for anim in &mut self.animators {
            if !anim.playing {
                continue;
            }
            let Some(clip) = self.animations.get(anim.clip) else {
                continue;
            };
            anim.time += dt * anim.speed;
            if anim.looping && clip.duration > 0.0 {
                anim.time = anim.time.rem_euclid(clip.duration);
            } else {
                anim.time = anim.time.clamp(0.0, clip.duration);
            }
            let t = anim.time;
            for ch in &clip.channels {
                match (&ch.path, &ch.values) {
                    (AnimPath::Translation, AnimValues::Vec3(v)) => {
                        updates.push(Update::T(ch.target, sample_vec3(&ch.times, v, t, ch.step)));
                    }
                    (AnimPath::Scale, AnimValues::Vec3(v)) => {
                        updates.push(Update::S(ch.target, sample_vec3(&ch.times, v, t, ch.step)));
                    }
                    (AnimPath::Rotation, AnimValues::Quat(v)) => {
                        updates.push(Update::R(ch.target, sample_quat(&ch.times, v, t, ch.step)));
                    }
                    _ => {}
                }
            }
        }
        for u in updates {
            match u {
                Update::T(h, v) => {
                    if let Some(n) = self.nodes.get_mut(h) {
                        n.local.translation = v;
                    }
                }
                Update::R(h, q) => {
                    if let Some(n) = self.nodes.get_mut(h) {
                        n.local.rotation = q;
                    }
                }
                Update::S(h, v) => {
                    if let Some(n) = self.nodes.get_mut(h) {
                        n.local.scale = v;
                    }
                }
            }
        }
    }

    pub fn joint_matrices(&self, skin: Handle<Skin>, mesh_node: Handle<Node>) -> Vec<Mat4> {
        let world = self.world_matrices();
        self.joint_matrices_with_cache(skin, mesh_node, &world)
    }

    pub fn joint_matrices_with_cache(
        &self,
        skin: Handle<Skin>,
        mesh_node: Handle<Node>,
        world: &HashMap<(u32, u32), Mat4>,
    ) -> Vec<Mat4> {
        let Some(skin) = self.skins.get(skin) else {
            return Vec::new();
        };
        let mesh_world = world
            .get(&mesh_node.key())
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        let inv_mesh = mesh_world.inverse();
        skin.joints
            .iter()
            .zip(skin.inverse_bind.iter())
            .map(|(&joint, &ibm)| {
                let jw = world
                    .get(&joint.key())
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                inv_mesh * jw * ibm
            })
            .collect()
    }

    /// Stick bones for a skin.
    /// Hierarchy parent→child when present; ARP-style flat deform bones
    /// (siblings under a non-joint rig) are linked along local +Y.
    ///
    /// Pass `selected` / `ik_targets` / `ik_poles` joint handles in `opts` for
    /// highlight colors. IK control joints get markers only (no stick).
    pub fn debug_skeleton(&mut self, skin: Handle<Skin>, opts: &SkeletonDebugOpts<'_>) {
        let Some(skin) = self.skins.get(skin) else {
            return;
        };
        let joints = skin.joints.clone();
        let joint_set: HashSet<_> = joints.iter().map(|j| j.key()).collect();

        let parent_joint = |scene: &Self, j: Handle<Node>| -> Option<Handle<Node>> {
            let mut parent = scene.nodes.get(j).and_then(|n| n.parent);
            while let Some(p) = parent {
                if joint_set.contains(&p.key()) {
                    return Some(p);
                }
                parent = scene.nodes.get(p).and_then(|n| n.parent);
            }
            None
        };
        let is_descendant = |scene: &Self, ancestor: Handle<Node>, node: Handle<Node>| -> bool {
            let mut cur = scene.nodes.get(node).and_then(|n| n.parent);
            while let Some(p) = cur {
                if p.key() == ancestor.key() {
                    return true;
                }
                cur = scene.nodes.get(p).and_then(|n| n.parent);
            }
            false
        };
        let pos_of =
            |scene: &Self, j: Handle<Node>| scene.world_matrix(j).transform_point3(Vec3::ZERO);

        let mut segments: Vec<(Vec3, Vec3, Handle<Node>)> = Vec::new();
        let mut has_joint_child = HashSet::new();

        // 1) Real hierarchy bones (colored by parent / outgoing joint).
        for &j in &joints {
            if is_ik_control(j, opts) {
                continue;
            }
            let Some(p) = parent_joint(self, j) else {
                continue;
            };
            if is_ik_control(p, opts) {
                continue;
            }
            has_joint_child.insert(p.key());
            let from = pos_of(self, p);
            let to = pos_of(self, j);
            if (to - from).length_squared() > 1e-12 {
                segments.push((from, to, p));
            }
        }

        // 2) Flat deform bones (Auto-Rig Pro etc.): no joint parent → next along +Y.
        for &j in &joints {
            if is_ik_control(j, opts) {
                continue;
            }
            if parent_joint(self, j).is_some() {
                continue;
            }
            if self
                .nodes
                .get(j)
                .is_some_and(|n| n.name.to_ascii_lowercase().starts_with("root"))
            {
                continue;
            }
            let m = self.world_matrix(j);
            let from = m.transform_point3(Vec3::ZERO);
            let mut axis = m.transform_vector3(Vec3::Y);
            if axis.length_squared() < 1e-8 {
                continue;
            }
            axis = axis.normalize();

            let mut best: Option<(f32, Vec3)> = None;
            for &k in &joints {
                if k.key() == j.key() || is_descendant(self, j, k) || is_ik_control(k, opts) {
                    continue;
                }
                let to = pos_of(self, k);
                let d = to - from;
                let dist = d.length();
                if dist < 1e-6 {
                    continue;
                }
                let align = d.dot(axis) / dist;
                if align < 0.85 {
                    continue;
                }
                let cost = dist / align.powi(8);
                if best.is_none_or(|(c, _)| cost < c) {
                    best = Some((cost, to));
                }
            }
            if let Some((_, to)) = best {
                segments.push((from, to, j));
            } else if !has_joint_child.contains(&j.key()) {
                let len = segments
                    .iter()
                    .map(|(a, b, _)| (*b - *a).length())
                    .sum::<f32>()
                    / segments.len().max(1) as f32;
                let len = if len > 1e-4 { len } else { 0.1 };
                segments.push((from, from + axis * len, j));
            }
        }

        let joint_pts: Vec<(Vec3, Handle<Node>)> =
            joints.iter().map(|&j| (pos_of(self, j), j)).collect();
        self.debug_skeleton_sticks(&segments, &joint_pts, opts);
    }

    /// Stick skeleton from explicit segments + joint markers.
    ///
    /// `segments`: `(from, to, owner)` — `owner` picks fill/outline via `opts`.
    /// `joints`: `(world_pos, joint)` for dots (including IK controls).
    pub fn debug_skeleton_sticks(
        &mut self,
        segments: &[(Vec3, Vec3, Handle<Node>)],
        joints: &[(Vec3, Handle<Node>)],
        opts: &SkeletonDebugOpts<'_>,
    ) {
        let overlay = opts.overlay;

        // Outlines first so fills of every bone sit above all outlines.
        for &(from, to, owner) in segments {
            let (outline, _, _, _) = skeleton_joint_style(owner, opts);
            let mut lo = LineOpts::color(outline).width(SKELETON_OUTLINE_W);
            if overlay {
                lo = lo.overlay();
            }
            self.debug.line(from, to, lo);
        }
        for &(from, to, owner) in segments {
            let (_, fill, _, _) = skeleton_joint_style(owner, opts);
            let mut lf = LineOpts::color(fill).width(SKELETON_LINE_W);
            if overlay {
                lf = lf.overlay();
            }
            self.debug.line(from, to, lf);
        }

        for &(pos, joint) in joints {
            let (outline, fill, joint_px, joint_out_px) = skeleton_joint_style(joint, opts);
            self.debug
                .bone_joint(pos, fill, outline, joint_px, joint_out_px, overlay);
        }
    }

    pub fn debug_skeletons(&mut self, opts: &SkeletonDebugOpts<'_>) {
        let skins: Vec<_> = self.skins.iter().map(|(h, _)| h).collect();
        for h in skins {
            self.debug_skeleton(h, opts);
        }
    }
}
