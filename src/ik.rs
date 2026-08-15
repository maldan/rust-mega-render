//! Generic skeletal IK solvers: CCD chain-pull and two-bone analytic + pole
//! vector constraint. Operates purely on [`Scene`] node transforms — no
//! knowledge of any host app's bone/document model.

use glam::{Mat4, Quat, Vec3};

use super::node::Node;
use super::scene::Scene;
use super::store::Handle;

const MAX_STEP_RAD: f32 = 0.45;

/// A configured IK chain, in generic node-handle terms.
///
/// `bones` are the rotating joints, root-first (does not include `tip`).
/// `lengths[i]` is the rest length from `bones[i]` to `bones[i+1]` (last
/// entry is `bones.last()` → `tip`).
pub struct IkChainDef {
    pub tip: Handle<Node>,
    pub bones: Vec<Handle<Node>>,
    pub lengths: Vec<f32>,
    /// World-space target the tip should reach.
    pub target_pos: Vec3,
    /// Desired world rotation for the tip (already includes any offset).
    pub target_rot: Quat,
    /// World-space pole (knee/elbow aim) position.
    pub pole_pos: Vec3,
    /// Extra twist around the root→target axis, radians.
    pub pole_angle: f32,
    /// Rest "bend" direction in `bones[0]` local space (fallback pole ref
    /// when the chain is straight).
    pub pole_ref_local: Vec3,
    /// Local rotations to reset `bones` + `tip` to before solving
    /// (root..=tip, `bones.len() + 1` entries). `None` skips the reset.
    pub bind_rotations: Option<Vec<Quat>>,
}

/// Solve `def` in place: resets to bind (if given), positions the chain to
/// reach `target_pos` (analytic for 2 bones, CCD + pole align otherwise),
/// then sets the tip's world rotation to `target_rot`.
pub fn solve_ik(scene: &mut Scene, def: &IkChainDef) {
    if def.bones.is_empty() {
        return;
    }
    if let Some(bind) = &def.bind_rotations {
        restore_bind(scene, &def.bones, def.tip, bind);
    }

    if def.bones.len() == 2 && def.lengths.len() >= 2 {
        solve_two_bone(scene, def);
    } else {
        solve_ccd(scene, def, 28);
    }

    apply_tip_rotation(scene, def.tip, def.target_rot);
}

fn restore_bind(scene: &mut Scene, bones: &[Handle<Node>], tip: Handle<Node>, bind: &[Quat]) {
    for (&h, &q) in bones.iter().chain(std::iter::once(&tip)).zip(bind.iter()) {
        if let Some(n) = scene.nodes.get_mut(h) {
            n.local.rotation = q;
        }
    }
}

fn solve_two_bone(scene: &mut Scene, def: &IkChainDef) {
    let root = def.bones[0];
    let mid = def.bones[1];
    let root_pos = scene.world_matrix(root).transform_point3(Vec3::ZERO);
    let len1 = def.lengths[0].max(1e-4);
    let len2 = def.lengths[1].max(1e-4);
    let (mid_pos, tip_pos) =
        two_bone_positions(root_pos, len1, len2, def.target_pos, def.pole_pos, def.pole_angle);
    swing_bone_to(scene, root, mid, mid_pos);
    apply_pole_roll(scene, root, mid, def.pole_ref_local, def.pole_pos, def.pole_angle);
    swing_bone_to(scene, mid, def.tip, tip_pos);
}

fn solve_ccd(scene: &mut Scene, def: &IkChainDef, iters: u32) {
    for _ in 0..iters {
        for i in (0..def.bones.len()).rev() {
            ccd_rotate_joint(scene, def.bones[i], def.tip, def.target_pos, 1.0);
        }
    }
    let mid = def.bones[def.bones.len() / 2];
    align_chain_to_pole(
        scene,
        def.bones[0],
        mid,
        def.tip,
        def.target_pos,
        def.pole_pos,
        def.pole_angle,
        def.pole_ref_local,
    );
    for _ in 0..(iters / 2) {
        for i in (0..def.bones.len()).rev() {
            ccd_rotate_joint(scene, def.bones[i], def.tip, def.target_pos, 1.0);
        }
    }
    align_chain_to_pole(
        scene,
        def.bones[0],
        mid,
        def.tip,
        def.target_pos,
        def.pole_pos,
        def.pole_angle,
        def.pole_ref_local,
    );
}

/// Tip world rotation = `desired` (already includes any host-side offset).
fn apply_tip_rotation(scene: &mut Scene, tip: Handle<Node>, desired: Quat) {
    let parent_world = scene
        .nodes
        .get(tip)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(Mat4::IDENTITY);
    let parent_rot = quat_from_matrix(parent_world);
    let Some(node) = scene.nodes.get_mut(tip) else {
        return;
    };
    node.local.rotation = (parent_rot.inverse() * desired.normalize()).normalize();
}

/// Rotate `joint` so the chain tip moves toward `target`. `weight` 0..1 scales the step.
pub fn ccd_rotate_joint(
    scene: &mut Scene,
    joint: Handle<Node>,
    tip: Handle<Node>,
    target: Vec3,
    weight: f32,
) {
    if weight < 1e-4 {
        return;
    }
    let joint_pos = scene.world_matrix(joint).transform_point3(Vec3::ZERO);
    let tip_pos = scene.world_matrix(tip).transform_point3(Vec3::ZERO);
    let to_tip = (tip_pos - joint_pos).normalize_or_zero();
    let to_tgt = (target - joint_pos).normalize_or_zero();
    if to_tip.length_squared() < 1e-10 || to_tgt.length_squared() < 1e-10 {
        return;
    }

    let rot = clamped_arc(to_tip, to_tgt, MAX_STEP_RAD * weight);
    if (rot.xyz().length_squared() + (rot.w - 1.0) * (rot.w - 1.0)) < 1e-12 {
        return;
    }

    let parent_world = scene
        .nodes
        .get(joint)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(Mat4::IDENTITY);
    let parent_rot = quat_from_matrix(parent_world);
    let Some(node) = scene.nodes.get_mut(joint) else {
        return;
    };
    let world_r = parent_rot * node.local.rotation;
    let new_world = (rot * world_r).normalize();
    node.local.rotation = (parent_rot.inverse() * new_world).normalize();
}

/// Translate `bone` in world space by `world_delta` (adjusts local translation).
pub fn translate_bone_world(scene: &mut Scene, bone: Handle<Node>, world_delta: Vec3) {
    let parent_world = scene
        .nodes
        .get(bone)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(Mat4::IDENTITY);
    let Some(n) = scene.nodes.get_mut(bone) else {
        return;
    };
    let world = parent_world.transform_point3(n.local.translation) + world_delta;
    n.local.translation = parent_world.inverse().transform_point3(world);
}

/// Swing `bone` so its child joint moves toward `desired_child` (preserves twist).
fn swing_bone_to(scene: &mut Scene, bone: Handle<Node>, child: Handle<Node>, desired_child: Vec3) {
    let origin = scene.world_matrix(bone).transform_point3(Vec3::ZERO);
    let cur_child = scene.world_matrix(child).transform_point3(Vec3::ZERO);
    let from = (cur_child - origin).normalize_or_zero();
    let to = (desired_child - origin).normalize_or_zero();
    if from.length_squared() < 1e-10 || to.length_squared() < 1e-10 {
        return;
    }
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 0.999999 {
        return;
    }
    let q = Quat::from_rotation_arc(from, to);
    apply_world_delta_rot(scene, bone, q);
}

fn apply_world_delta_rot(scene: &mut Scene, bone: Handle<Node>, world_delta: Quat) {
    let parent_world = scene
        .nodes
        .get(bone)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(Mat4::IDENTITY);
    let parent_rot = quat_from_matrix(parent_world);
    let Some(node) = scene.nodes.get_mut(bone) else {
        return;
    };
    let world_r = parent_rot * node.local.rotation;
    node.local.rotation = (parent_rot.inverse() * (world_delta * world_r).normalize()).normalize();
}

fn apply_pole_roll(
    scene: &mut Scene,
    bone: Handle<Node>,
    child: Handle<Node>,
    ref_local: Vec3,
    pole: Vec3,
    pole_angle: f32,
) {
    if ref_local.length_squared() < 1e-12 {
        return;
    }
    let origin = scene.world_matrix(bone).transform_point3(Vec3::ZERO);
    let child_pos = scene.world_matrix(child).transform_point3(Vec3::ZERO);
    let aim = (child_pos - origin).normalize_or_zero();
    if aim.length_squared() < 1e-8 {
        return;
    }
    let rot = quat_from_matrix(scene.world_matrix(bone));
    let from = reject(rot * ref_local, aim);
    let mut to = reject(pole - origin, aim);
    if pole_angle.abs() > 1e-6 {
        to = Quat::from_axis_angle(aim, pole_angle) * to;
    }
    twist_dir(scene, bone, aim, from, to);
}

fn twist_dir(scene: &mut Scene, bone: Handle<Node>, axis: Vec3, from: Vec3, to: Vec3) {
    if from.length_squared() < 1e-10 || to.length_squared() < 1e-10 {
        return;
    }
    let from = from.normalize();
    let to = to.normalize();
    let mut angle = from.dot(to).clamp(-1.0, 1.0).acos();
    if from.cross(to).dot(axis) < 0.0 {
        angle = -angle;
    }
    if angle.abs() < 1e-6 {
        return;
    }
    apply_world_delta_rot(scene, bone, Quat::from_axis_angle(axis, angle));
}

fn align_chain_to_pole(
    scene: &mut Scene,
    root: Handle<Node>,
    mid: Handle<Node>,
    tip: Handle<Node>,
    target: Vec3,
    pole: Vec3,
    pole_angle: f32,
    ref_local: Vec3,
) {
    let root_pos = scene.world_matrix(root).transform_point3(Vec3::ZERO);
    let mid_pos = scene.world_matrix(mid).transform_point3(Vec3::ZERO);
    let tip_pos = scene.world_matrix(tip).transform_point3(Vec3::ZERO);
    let mut axis = (target - root_pos).normalize_or_zero();
    if axis.length_squared() < 1e-8 {
        axis = (tip_pos - root_pos).normalize_or_zero();
    }
    if axis.length_squared() < 1e-8 {
        return;
    }

    let mut from = reject(mid_pos - root_pos, axis);
    if from.length_squared() < 1e-8 && ref_local.length_squared() > 1e-12 {
        let rot = quat_from_matrix(scene.world_matrix(root));
        from = reject(rot * ref_local, axis);
    }
    let to = pole_in_plane(root_pos, target, pole, pole_angle);
    twist_dir(scene, root, axis, from, to);
}

fn pole_in_plane(root: Vec3, target: Vec3, pole: Vec3, pole_angle: f32) -> Vec3 {
    let dir = (target - root).normalize_or_zero();
    if dir.length_squared() < 1e-8 {
        return Vec3::ZERO;
    }
    let mut bend = reject(pole - root, dir);
    if bend.length_squared() < 1e-10 {
        return Vec3::ZERO;
    }
    if pole_angle.abs() > 1e-6 {
        bend = Quat::from_axis_angle(dir, pole_angle) * bend;
    }
    bend
}

fn reject(v: Vec3, axis: Vec3) -> Vec3 {
    v - axis * v.dot(axis)
}

fn two_bone_positions(
    root: Vec3,
    len1: f32,
    len2: f32,
    target: Vec3,
    pole: Vec3,
    pole_angle: f32,
) -> (Vec3, Vec3) {
    let mut to_t = target - root;
    let mut dist = to_t.length();
    if dist < 1e-6 {
        to_t = Vec3::Y * (len1 + len2);
        dist = to_t.length();
    }
    let max_r = len1 + len2;
    let min_r = (len1 - len2).abs() + 1e-4;
    let dist_c = dist.clamp(min_r, max_r - 1e-4);
    let dir = to_t / dist;

    let mut bend = pole_in_plane(root, target, pole, pole_angle);
    if bend.length_squared() < 1e-10 {
        bend = orphan_perp(dir);
    }
    let bend = bend.normalize();

    let cos_a = ((len1 * len1 + dist_c * dist_c - len2 * len2) / (2.0 * len1 * dist_c))
        .clamp(-1.0, 1.0);
    let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();

    let mid = root + (dir * cos_a + bend * sin_a) * len1;
    let tip = root + dir * dist_c;
    (mid, tip)
}

fn orphan_perp(dir: Vec3) -> Vec3 {
    let mut n = dir.cross(Vec3::Y);
    if n.length_squared() < 1e-10 {
        n = dir.cross(Vec3::X);
    }
    n.normalize_or_zero()
}

fn clamped_arc(from: Vec3, to: Vec3, max_rad: f32) -> Quat {
    let f = from.normalize_or_zero();
    let t = to.normalize_or_zero();
    let dot = f.dot(t).clamp(-1.0, 1.0);
    let angle = dot.acos();
    if angle < 1e-5 {
        return Quat::IDENTITY;
    }
    let axis = f.cross(t);
    if axis.length_squared() < 1e-10 {
        return Quat::IDENTITY;
    }
    Quat::from_axis_angle(axis.normalize(), angle.min(max_rad))
}

pub fn quat_from_matrix(m: Mat4) -> Quat {
    let (_, r, _) = m.to_scale_rotation_translation();
    r.normalize()
}
