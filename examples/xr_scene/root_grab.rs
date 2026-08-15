//! Rigid whole-model "root handle" grab. Unlike `soft_chain`'s jiggle
//! chains (which run a physics simulation), grabbing a model's root handle
//! cube (see `xr_scene.rs`'s `HANDLE_CUBE_SIZE`) just drives the model
//! root node's local transform directly every frame, so moving *and*
//! rotating the controller moves/rotates the whole skeleton rigidly — like
//! picking the doll up by an invisible handle glued to its hip.

use glam::{Mat3, Quat, Vec3};
use mega_render::{Handle, Node, Scene};

/// How a grab keeps tracking its target point every frame — mirrors
/// `soft_chain::GrabAnchor` (see there for why `Ray` needs a remembered
/// distance instead of just tracking the hand every frame).
#[derive(Clone, Copy)]
pub enum RootAnchor {
    /// Grabbed by touch: track the hand directly.
    Touch,
    /// Grabbed by aiming the laser through the handle from `distance`
    /// meters away: track a point that far out along the *current* aim
    /// ray, so moving/rotating the controller drags the model even from
    /// across the room.
    Ray { distance: f32 },
}

/// A hand's grab-test inputs this frame.
pub struct RootGrabProbe {
    pub hand_pos: Vec3,
    pub ray: Option<(Vec3, Vec3)>,
    pub ray_max_dist: f32,
}

/// Result of [`pick_root`]: which root (by index into the caller's `roots`
/// slice) was hit, and how to keep tracking it.
pub struct RootHit {
    pub root_idx: usize,
    pub anchor: RootAnchor,
}

/// Distance from `point` to the closest point on the ray `origin + t*dir`
/// (`t` clamped to `[0, max_dist]`, `dir` assumed normalized), plus that `t`.
fn ray_point_dist(origin: Vec3, dir: Vec3, max_dist: f32, point: Vec3) -> (f32, f32) {
    let t = (point - origin).dot(dir).clamp(0.0, max_dist);
    let closest = origin + dir * t;
    ((closest - point).length(), t)
}

/// Picks the closest root handle (a sphere of `radius` meters centered on
/// each root's current world position) touched or aimed-at by `probe`,
/// within `touch_margin` extra reach beyond the sphere surface for the
/// touch test. Returns `None` if no handle is touched or intersected.
pub fn pick_root(
    scene: &Scene,
    roots: &[Handle<Node>],
    probe: &RootGrabProbe,
    radius: f32,
    touch_margin: f32,
) -> Option<RootHit> {
    let mut best: Option<(f32, RootHit)> = None;
    for (i, &root) in roots.iter().enumerate() {
        let center = scene.world_matrix(root).transform_point3(Vec3::ZERO);
        let touch_d = (center - probe.hand_pos).length() - radius - touch_margin;
        let ray_hit = probe.ray.map(|(origin, dir)| {
            let (d, t) = ray_point_dist(origin, dir, probe.ray_max_dist, center);
            (d - radius, t)
        });
        let (hit_d, anchor) = match ray_hit {
            Some((ray_d, t)) if ray_d <= touch_d => (ray_d, RootAnchor::Ray { distance: t }),
            _ => (touch_d, RootAnchor::Touch),
        };
        if hit_d <= 0.0 && best.as_ref().is_none_or(|(bd, _)| hit_d < *bd) {
            best = Some((hit_d, RootHit { root_idx: i, anchor }));
        }
    }
    best.map(|(_, h)| h)
}

/// Live grab state for one held root handle: the rigid offset (expressed
/// in the *hand's* rotated frame, captured the instant the grab started)
/// between the model root's transform and the hand/ray target — reapplied
/// every frame in [`apply_grab`] so the whole model tracks the controller
/// rigidly (both position and orientation) instead of snapping to it.
pub struct RootGrab {
    pub root: Handle<Node>,
    pub anchor: RootAnchor,
    rel_rot: Quat,
    rel_pos: Vec3,
}

/// Starts a grab: captures the model root's current world transform
/// relative to `hand_rot`/`anchor_point` (the hand position for `Touch`,
/// or the ray's hit point for `Ray`) so [`apply_grab`] reproduces the exact
/// current pose on the first frame, then keeps the offset fixed as the
/// hand moves and rotates.
pub fn begin_grab(scene: &Scene, root: Handle<Node>, anchor: RootAnchor, hand_rot: Quat, anchor_point: Vec3) -> RootGrab {
    let (_, rot, pos) = scene.world_matrix(root).to_scale_rotation_translation();
    let rel_rot = (hand_rot.inverse() * rot).normalize();
    let rel_pos = hand_rot.inverse() * (pos - anchor_point);
    RootGrab { root, anchor, rel_rot, rel_pos }
}

/// Drives `grab.root`'s local transform so its world transform tracks
/// `hand_rot` rotated by the captured offset, and `target_point` (the
/// hand position for `Touch` grabs, or a point along the live aim ray for
/// `Ray` grabs — see [`RootAnchor`]) translated by the captured offset.
pub fn apply_grab(scene: &mut Scene, grab: &RootGrab, hand_rot: Quat, target_point: Vec3) {
    let world_rot = (hand_rot * grab.rel_rot).normalize();
    let world_pos = hand_rot * grab.rel_pos + target_point;

    let parent = scene.nodes.get(grab.root).and_then(|n| n.parent);
    let (parent_rot, parent_pos) = match parent {
        Some(p) => {
            let (_, rot, pos) = scene.world_matrix(p).to_scale_rotation_translation();
            (rot, pos)
        }
        None => (Quat::IDENTITY, Vec3::ZERO),
    };

    let Some(node) = scene.nodes.get_mut(grab.root) else { return };
    node.local.rotation = (parent_rot.inverse() * world_rot).normalize();
    node.local.translation = parent_rot.inverse() * (world_pos - parent_pos);
}

/// Builds a hand's engine-space world rotation from an OpenXR pose's
/// orientation: mirrors each individual local axis into engine space via
/// `to_engine` (the same trick `xr_scene.rs`'s flight code uses to build
/// `forward`/`right` from `head_rot`), then composes with the rig's turn —
/// rather than naively reinterpreting the raw quaternion, which would get
/// the handedness wrong (see `xr_scene.rs::to_engine`'s doc comment).
pub fn hand_rot_world(to_engine: impl Fn(Vec3) -> Vec3, rig_rot: Quat, orientation: Quat) -> Quat {
    let right = rig_rot * to_engine(orientation * Vec3::X);
    let up = rig_rot * to_engine(orientation * Vec3::Y);
    let forward = rig_rot * to_engine(orientation * Vec3::NEG_Z);
    Quat::from_mat3(&Mat3::from_cols(right, up, forward)).normalize()
}
