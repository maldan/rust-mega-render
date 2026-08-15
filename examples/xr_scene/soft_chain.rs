//! Adapter between glTF skeleton bones and `mega_physics::chain`'s generic
//! Verlet particle-chain simulation — used here for secondary "jiggle"
//! motion on named bones (e.g. `Boob.L` / `Boob.R`), plus VR-controller grab
//! via [`PullTarget`] and capsule↔capsule collision via
//! [`resolve_capsule_collisions`] so the left/right chains actually push
//! each other apart instead of interpenetrating. The physics crate knows
//! nothing about bones or scene graphs (see its own module doc); this file
//! is entirely about turning bone world transforms into `ChainFrame` inputs
//! and writing simulated particle positions back into bone rotations.
//!
//! Ported from the same pattern used for breast jiggle in
//! `model-rig/src/soft_chain.rs` (Miruko rig), minus the rig-document/UI
//! bits that don't apply to a single VR demo scene.

use glam::{Quat, Vec3};
use mega_physics::chain::{
    resolve_capsule_collisions, Chain, ChainCapsule, ChainFrame, PreparedChain, PullTarget,
};
use mega_physics::Isometry;
use mega_render::{quat_from_matrix, Handle, Node, Scene};

const SUBSTEPS: u32 = 4;
const CONSTRAINT_ITERS: u32 = 6;
/// How hard a grab pulls a particle toward the controller per constraint
/// iteration (0..1).
const GRAB_BLEND: f32 = 0.55;
/// Pull falloff from root toward the grabbed particle (`< 1` = mid-chain
/// feels more of the pull).
const GRAB_CHAIN_POWER: f32 = 0.5;
/// Capsule↔capsule collision passes per substep (see
/// `resolve_breast_capsules`) — same value model-rig's Miruko rig uses.
const COLLISION_PASSES: u32 = 3;
/// 0 = gentle push (jelly), 1 = firmer push — see `ChainCapsule::softness`.
const COLLISION_SOFTNESS: f32 = 0.5;

/// Tunable jiggle parameters — see `mega_physics::chain::ChainFrame` for the
/// exact meaning of each; these are just a per-chain "preset".
#[derive(Clone, Copy)]
pub struct SoftChainConfig {
    pub gravity: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub inertia: f32,
    /// Max bend angle per segment, radians.
    pub max_angle: f32,
}

impl Default for SoftChainConfig {
    fn default() -> Self {
        // Starting point from model-rig's Miruko breast rig
        // (`scripts/miruko.lua`'s `rig.create_soft(bone("Boob", side), ...)`),
        // with a bit more damping (settles faster, less endless wobble) and
        // a bit less max bend angle (slightly firmer swing) for this scene.
        Self {
            gravity: 9.8,
            stiffness: 55.0,
            damping: 7.5,
            inertia: 0.05,
            max_angle: 85f32.to_radians(),
        }
    }
}

/// One simulated jiggle chain: a linear run of bones (root -> ... -> tip)
/// plus one virtual particle past the tip, so even a single real bone still
/// gets somewhere to swing toward.
pub struct SoftChain {
    /// Kept for debugging/introspection (e.g. logging which chain a grab
    /// landed on); not read by the simulation itself.
    #[allow(dead_code)]
    pub name: String,
    /// Fixed parent of `bones[0]` — follows body motion, never simulated.
    anchor: Handle<Node>,
    /// Bones root -> tip, in chain order.
    bones: Vec<Handle<Node>>,
    /// Rest length `bones[i-1] -> bones[i]` (index 0 unused).
    lengths: Vec<f32>,
    /// Length of the virtual tip particle past `bones.last()`.
    tip_length: f32,
    /// Bone-local rotations captured at chain-creation time; restored every
    /// frame before recomputing the animated "rest" pose the chain springs
    /// toward — otherwise last frame's sim result would feed back into
    /// itself as the new rest pose and the chain would never settle.
    bind_rotations: Vec<Quat>,
    /// "Out of body" direction in `anchor`-local space; leave zero to
    /// disable the support-plane push (see `mega_physics::chain`).
    pub support_normal_local: Vec3,
    pub config: SoftChainConfig,
    /// Radius (meters) of the capsule wrapped around the chain's tip segment
    /// — see [`tip_capsule`] and [`pick_chain_tip`]. `None` means the chain
    /// can't be grabbed (no touch/ray pick) at all; it still jiggles.
    pub collider_radius: Option<f32>,
    /// Whether this chain's tip capsule participates in
    /// [`resolve_breast_capsules`]'s capsule↔capsule push-apart against
    /// *other* chains' tip capsules. Independent of `collider_radius` — a
    /// chain can be grabbable without ever colliding with other chains (or
    /// vice versa); `false` here has no effect if `collider_radius` is
    /// `None` (no capsule to collide with in the first place).
    pub collide_with_others: bool,
    sim: Chain,
}

/// Auto-collects a linear bone chain starting at `start` (follows
/// single-child descendants down to a leaf/branch) and builds a
/// [`SoftChain`] from the current bind pose. Returns `None` if `start`
/// doesn't exist in `scene`.
pub fn create_soft_chain(
    scene: &Scene,
    name: impl Into<String>,
    start: Handle<Node>,
    config: SoftChainConfig,
    collider_radius: Option<f32>,
    collide_with_others: bool,
) -> Option<SoftChain> {
    let start_node = scene.nodes.get(start)?;
    let anchor = start_node.parent.unwrap_or(start);
    let bones = collect_descendant_chain(scene, start);

    let world_pos: Vec<Vec3> = bones
        .iter()
        .map(|&b| scene.world_matrix(b).transform_point3(Vec3::ZERO))
        .collect();
    let mut lengths = vec![0.0f32];
    for i in 1..bones.len() {
        lengths.push((world_pos[i] - world_pos[i - 1]).length().max(1e-4));
    }
    let avg_len = if lengths.len() > 1 {
        lengths[1..].iter().sum::<f32>() / (lengths.len() - 1) as f32
    } else {
        // Single-bone chain (no child bone found): no real segment to
        // average, so fall back to the bone's own translation length as a
        // rough "one bone's worth" of reach for the virtual tip.
        scene
            .nodes
            .get(bones[0])
            .map(|n| n.local.translation.length())
            .filter(|l| *l > 1e-4)
            .unwrap_or(0.05)
    };
    let tip_length = (avg_len * 0.65).max(1e-3);

    let bind_rotations = bones
        .iter()
        .map(|&b| scene.nodes.get(b).map(|n| n.local.rotation).unwrap_or(Quat::IDENTITY))
        .collect();

    Some(SoftChain {
        name: name.into(),
        anchor,
        bones,
        lengths,
        tip_length,
        bind_rotations,
        support_normal_local: Vec3::ZERO,
        config,
        collider_radius,
        collide_with_others,
        sim: Chain::new(),
    })
}

fn collect_descendant_chain(scene: &Scene, start: Handle<Node>) -> Vec<Handle<Node>> {
    let mut bones = vec![start];
    let mut current = start;
    loop {
        let mut children = scene
            .nodes
            .iter()
            .filter(|(_, n)| n.parent.map(|p| p.key()) == Some(current.key()));
        let Some((only, _)) = children.next() else { break };
        if children.next().is_some() {
            break; // branches here — a separate chain would start at each child
        }
        bones.push(only);
        current = only;
    }
    bones
}

/// An active controller grab: pulls one particle of one chain toward a live
/// world-space target this frame (see [`pick_chain_tip`]).
pub struct GrabPull {
    pub chain_idx: usize,
    pub particle: usize,
    pub target: Vec3,
}

/// A particle hit by [`pick_chain_tip`], along with how the hand should
/// keep tracking it every subsequent frame while the grab is held — see
/// [`GrabAnchor`].
pub struct ChainHit {
    pub chain_idx: usize,
    pub particle: usize,
    pub anchor: GrabAnchor,
}

/// How a grab's pull target should be recomputed each frame it's held (set
/// once, at the moment [`pick_chain_tip`] finds the hit, then reused every
/// frame until release — see the caller's main loop).
#[derive(Clone, Copy)]
pub enum GrabAnchor {
    /// Grabbed by touch (grip near the capsule): track the hand directly,
    /// like actually holding it in your fist.
    Touch,
    /// Grabbed by aiming the laser through the capsule from `distance`
    /// meters away: track a point that far out along the *current* aim
    /// ray, like a tractor-beam/distance-grab — this is what makes moving
    /// or rotating the controller drag the object while grabbed from afar,
    /// instead of it barely budging because the pull target was stuck at
    /// the hand's (far-away) position the whole time.
    Ray { distance: f32 },
}

/// Advance every chain's jiggle simulation by `dt` and write the result back
/// into bone rotations. `grabs` lets 0+ controllers pull a specific chain
/// particle toward their live position this frame.
pub fn evaluate_soft_chains(scene: &mut Scene, chains: &mut [SoftChain], dt: f32, grabs: &[GrabPull]) {
    let dt = dt.clamp(1.0 / 240.0, 1.0 / 20.0);

    for chain in chains.iter() {
        restore_bind(scene, chain);
    }

    let mut scratches: Vec<Option<PreparedChain>> = Vec::with_capacity(chains.len());
    let mut tip_axes: Vec<Vec3> = Vec::with_capacity(chains.len());
    for chain in chains.iter_mut() {
        let (scratch, tip_axis) = prepare_chain(scene, chain, dt);
        scratches.push(scratch);
        tip_axes.push(tip_axis);
    }

    for _ in 0..SUBSTEPS {
        for (i, chain) in chains.iter_mut().enumerate() {
            if let Some(scratch) = scratches[i].as_ref() {
                chain.sim.integrate(scratch);
            }
        }
        for (i, chain) in chains.iter_mut().enumerate() {
            let Some(scratch) = scratches[i].as_ref() else { continue };
            let pull = grabs.iter().find(|g| g.chain_idx == i).map(|g| PullTarget {
                particle: g.particle,
                target: g.target,
                blend: GRAB_BLEND,
                chain_power: GRAB_CHAIN_POWER,
            });
            chain.sim.constrain(scratch, CONSTRAINT_ITERS, pull);
        }
        // Capsule↔capsule from *current* particle poses (lockstep) — stable,
        // no particle hits — so e.g. the left/right breast tip capsules
        // (see `tip_capsule`) push each other apart instead of passing
        // through.
        resolve_breast_capsules(chains, &scratches);
        // Held grab is kinematic: kill Verlet velocity so release can't shoot.
        for g in grabs {
            if let Some(chain) = chains.get_mut(g.chain_idx) {
                chain.sim.zero_velocity();
            }
        }
    }

    for (i, chain) in chains.iter().enumerate() {
        if let Some(scratch) = scratches[i].as_ref() {
            write_chain_bones(scene, chain, scratch, tip_axes[i]);
        }
    }
}

/// Pushes overlapping chain tip capsules (see [`tip_capsule`]'s doc for what
/// the tip segment represents) apart via `mega_physics::chain`'s soft
/// capsule↔capsule resolver — this is what actually stops e.g. the left and
/// right jiggle chains from freely interpenetrating (the [`Capsule`] used by
/// [`pick_chain_tip`] only tests hand/ray proximity; it never pushes
/// anything by itself). No-op with fewer than two live capsules.
fn resolve_breast_capsules(chains: &mut [SoftChain], scratches: &[Option<PreparedChain>]) {
    let caps: Vec<ChainCapsule> = chains
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if !c.collide_with_others {
                return None;
            }
            let radius = c.collider_radius?;
            let pos = c.sim.positions();
            let n = pos.len();
            if n < 2 {
                return None;
            }
            Some(ChainCapsule {
                a: pos[n - 2],
                b: pos[n - 1],
                radius,
                softness: COLLISION_SOFTNESS,
                chain: Some(i),
            })
        })
        .collect();
    if caps.len() < 2 {
        return;
    }
    // `resolve_capsule_collisions` operates on `&mut [Chain]` + scratch
    // slices aligned by index, not `&mut [SoftChain]` directly — unpack/
    // repack the `sim` field to satisfy borrowing (same trick model-rig's
    // `soft_chain.rs` uses).
    let mut sims: Vec<Chain> = chains.iter_mut().map(|c| std::mem::take(&mut c.sim)).collect();
    resolve_capsule_collisions(&mut sims, scratches, &caps, COLLISION_PASSES);
    for (c, sim) in chains.iter_mut().zip(sims.into_iter()) {
        c.sim = sim;
    }
}

fn restore_bind(scene: &mut Scene, chain: &SoftChain) {
    for (&b, &rot) in chain.bones.iter().zip(chain.bind_rotations.iter()) {
        if let Some(n) = scene.nodes.get_mut(b) {
            n.local.rotation = rot;
        }
    }
}

fn isometry_from_mat4(m: glam::Mat4) -> Isometry {
    let (_, rotation, translation) = m.to_scale_rotation_translation();
    Isometry::new(translation, rotation)
}

/// Builds this frame's [`ChainFrame`] from the (bind-restored) animated bone
/// pose and calls [`Chain::prepare`]. Returns the tip bone's local "aim
/// axis" too (needed by [`write_chain_bones`] to swing the tip bone toward
/// the virtual tip particle).
fn prepare_chain(scene: &Scene, chain: &mut SoftChain, dt: f32) -> (Option<PreparedChain>, Vec3) {
    let n_bones = chain.bones.len();
    if n_bones == 0 {
        return (None, Vec3::Y);
    }
    let n = n_bones + 1;

    let mut rest: Vec<Vec3> = chain
        .bones
        .iter()
        .map(|&b| scene.world_matrix(b).transform_point3(Vec3::ZERO))
        .collect();

    let mut tip_dir = if n_bones >= 2 {
        (rest[n_bones - 1] - rest[n_bones - 2]).normalize_or_zero()
    } else {
        Vec3::ZERO
    };
    if tip_dir.length_squared() < 1e-8 {
        let tip = chain.bones[n_bones - 1];
        tip_dir = scene.world_matrix(tip).transform_vector3(Vec3::Y).normalize_or_zero();
    }
    if tip_dir.length_squared() < 1e-8 {
        tip_dir = Vec3::Y;
    }
    rest.push(rest[n_bones - 1] + tip_dir * chain.tip_length.max(1e-4));

    let mut seg_len = Vec::with_capacity(n);
    seg_len.push(0.0);
    for i in 1..n_bones {
        seg_len.push(chain.lengths[i].max(1e-4));
    }
    seg_len.push(chain.tip_length.max(1e-4));

    let root = isometry_from_mat4(scene.world_matrix(chain.bones[0]));

    let anchor_rot = quat_from_matrix(scene.world_matrix(chain.anchor));
    let support_normal = (anchor_rot * chain.support_normal_local).normalize_or_zero();

    let tip_local_axis = {
        let tip = chain.bones[n_bones - 1];
        let tip_rot = quat_from_matrix(scene.world_matrix(tip));
        let rest_tip_dir = (rest[n - 1] - rest[n - 2]).normalize_or_zero();
        let axis = tip_rot.inverse() * rest_tip_dir;
        if axis.length_squared() > 1e-8 {
            axis.normalize()
        } else {
            Vec3::Y
        }
    };

    let support_point = rest[0];
    let frame = ChainFrame {
        rest,
        seg_len,
        root,
        gravity: chain.config.gravity,
        stiffness: chain.config.stiffness,
        damping: chain.config.damping,
        inertia: chain.config.inertia,
        max_angle: chain.config.max_angle,
        support_point,
        support_normal,
    };
    let scratch = chain.sim.prepare(&frame, dt, SUBSTEPS);
    (Some(scratch), tip_local_axis)
}

fn write_chain_bones(scene: &mut Scene, chain: &SoftChain, _scratch: &PreparedChain, tip_local_axis: Vec3) {
    let n_bones = chain.bones.len();
    let pos = chain.sim.positions();
    let n = pos.len();
    for i in 0..n_bones {
        if i + 1 < n_bones {
            swing_bone_to(scene, chain.bones[i], chain.bones[i + 1], pos[i + 1]);
        } else {
            let tip = chain.bones[i];
            let origin = scene.world_matrix(tip).transform_point3(Vec3::ZERO);
            let tip_rot = quat_from_matrix(scene.world_matrix(tip));
            let from = (tip_rot * tip_local_axis).normalize_or_zero();
            let to = (pos[n - 1] - origin).normalize_or_zero();
            apply_swing(scene, tip, from, to);
        }
    }
}

/// Swing `bone` so its child joint moves toward `desired_child` (preserves twist).
fn swing_bone_to(scene: &mut Scene, bone: Handle<Node>, child: Handle<Node>, desired_child: Vec3) {
    let origin = scene.world_matrix(bone).transform_point3(Vec3::ZERO);
    let cur_child = scene.world_matrix(child).transform_point3(Vec3::ZERO);
    let from = (cur_child - origin).normalize_or_zero();
    let to = (desired_child - origin).normalize_or_zero();
    apply_swing(scene, bone, from, to);
}

fn apply_swing(scene: &mut Scene, bone: Handle<Node>, from: Vec3, to: Vec3) {
    if from.length_squared() < 1e-10 || to.length_squared() < 1e-10 {
        return;
    }
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 0.999999 {
        return;
    }
    let q = if dot < -0.999999 {
        let axis = from.any_orthonormal_vector();
        Quat::from_axis_angle(axis, std::f32::consts::PI)
    } else {
        Quat::from_rotation_arc(from, to)
    };
    apply_world_delta_rot(scene, bone, q);
}

fn apply_world_delta_rot(scene: &mut Scene, bone: Handle<Node>, world_delta: Quat) {
    let parent_world = scene
        .nodes
        .get(bone)
        .and_then(|n| n.parent)
        .map(|p| scene.world_matrix(p))
        .unwrap_or(glam::Mat4::IDENTITY);
    let parent_rot = quat_from_matrix(parent_world);
    let Some(node) = scene.nodes.get_mut(bone) else { return };
    let world_r = parent_rot * node.local.rotation;
    node.local.rotation = (parent_rot.inverse() * (world_delta * world_r).normalize()).normalize();
}

/// A capsule (thick line segment) grab collider wrapped around a chain's
/// tip segment — the last real bone (e.g. `Nipple.L`/`Nipple.R`, when that's
/// the tip of a `Boob.*` descendant chain — see `collect_descendant_chain`)
/// and the virtual particle past it. This is deliberately *not* a real
/// physics-engine collider (no broadphase, no response) — it only exists so
/// [`pick_chain_tip`] has a proper shape (instead of a bare point) to test
/// hand proximity and aim-ray intersection against.
#[derive(Clone, Copy)]
struct Capsule {
    a: Vec3,
    b: Vec3,
    radius: f32,
}

impl Capsule {
    fn closest_point(&self, p: Vec3) -> Vec3 {
        let ab = self.b - self.a;
        let len_sq = ab.length_squared();
        if len_sq < 1e-12 {
            return self.a;
        }
        let t = ((p - self.a).dot(ab) / len_sq).clamp(0.0, 1.0);
        self.a + ab * t
    }

    /// Signed distance from `p` to the capsule surface (negative = inside).
    fn distance_to_point(&self, p: Vec3) -> f32 {
        (self.closest_point(p) - p).length() - self.radius
    }

    /// Tests the capsule against a ray (`origin + t * dir`, `t` clamped to
    /// `[0, max_dist]`, `dir` assumed normalized). Returns `(signed distance
    /// from the capsule surface to the ray — negative/zero means the ray
    /// passes through the capsule —, distance from `origin` to the closest
    /// point on the ray)`; the latter is what [`pick_chain_tip`] remembers
    /// as the grab's anchor distance (see [`GrabAnchor::Ray`]).
    fn distance_to_ray(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> (f32, f32) {
        let (_, c2, dist) = closest_segment_segment(self.a, self.b, origin, origin + dir * max_dist);
        (dist - self.radius, (c2 - origin).length())
    }
}

/// Closest points between two line segments (Ericson, *Real-Time Collision
/// Detection*, §5.1.9 `ClosestPtSegmentSegment`). Returns `(point on
/// p1-q1, point on p2-q2, distance between them)`.
fn closest_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3, f32) {
    const EPS: f32 = 1e-8;
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let (s, t) = if a <= EPS && e <= EPS {
        (0.0, 0.0)
    } else if a <= EPS {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= EPS {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom.abs() > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    let c1 = p1 + d1 * s;
    let c2 = p2 + d2 * t;
    (c1, c2, (c1 - c2).length())
}

/// Builds the live grab capsule for a chain's tip segment (the last real
/// bone and the virtual particle past it), or `None` if the chain has no
/// collider (`collider_radius: None` — see [`SoftChain::collider_radius`]).
/// Uses live sim positions once the chain has run at least one frame,
/// falling back to the bind-pose bone position before that (e.g. the very
/// first frame) — same fallback the old point-picker used.
fn tip_capsule(scene: &Scene, chain: &SoftChain) -> Option<Capsule> {
    let radius = chain.collider_radius?;
    let n_bones = chain.bones.len();
    if n_bones == 0 {
        return None;
    }
    let n = n_bones + 1;
    if chain.sim.is_initialized() && chain.sim.positions().len() == n {
        let pos = chain.sim.positions();
        Some(Capsule { a: pos[n - 2], b: pos[n - 1], radius })
    } else {
        let tip = chain.bones[n_bones - 1];
        let m = scene.world_matrix(tip);
        let origin = m.transform_point3(Vec3::ZERO);
        let axis = m.transform_vector3(Vec3::Y).normalize_or_zero();
        let b = origin + axis * chain.tip_length.max(1e-4);
        Some(Capsule { a: origin, b, radius })
    }
}

/// A single hand's grab-test inputs this frame: `hand_pos` for
/// reach-and-touch, plus an optional aim ray so pointing at a chain's tip
/// collider from a distance also counts as a grab — you no longer have to
/// fly the controller right up against it, matching either input triggers
/// a grab.
pub struct GrabProbe {
    pub hand_pos: Vec3,
    pub ray: Option<(Vec3, Vec3)>,
    pub ray_max_dist: f32,
}

/// Picks the chain whose tip capsule (see [`tip_capsule`]) is closest to
/// being touched or aimed-at by `probe`, within `touch_margin` extra reach
/// beyond the capsule surface for the touch test. Returns `None` if no
/// chain's capsule is touched or intersected by the ray.
pub fn pick_chain_tip(scene: &Scene, chains: &[SoftChain], probe: &GrabProbe, touch_margin: f32) -> Option<ChainHit> {
    let mut best: Option<(f32, ChainHit)> = None;
    for (ci, chain) in chains.iter().enumerate() {
        let Some(capsule) = tip_capsule(scene, chain) else { continue };
        let touch_d = capsule.distance_to_point(probe.hand_pos) - touch_margin;
        let ray_hit = probe
            .ray
            .map(|(origin, dir)| capsule.distance_to_ray(origin, dir, probe.ray_max_dist));
        // Whichever test actually landed closer wins — and its anchor kind
        // (touch vs. ray) is what decides how this grab tracks the hand
        // every frame afterward (see `GrabAnchor`).
        let (hit_d, anchor) = match ray_hit {
            Some((ray_d, ray_dist)) if ray_d <= touch_d => (ray_d, GrabAnchor::Ray { distance: ray_dist }),
            _ => (touch_d, GrabAnchor::Touch),
        };
        if hit_d <= 0.0 && best.as_ref().is_none_or(|(bd, _)| hit_d < *bd) {
            best = Some((hit_d, ChainHit { chain_idx: ci, particle: chain.bones.len(), anchor }));
        }
    }
    best.map(|(_, h)| h)
}

/// Kill grab-induced Verlet velocity and start the post-release settle
/// window, so letting go can't shoot the chain.
pub fn release_grab(chains: &mut [SoftChain], chain_idx: usize) {
    if let Some(chain) = chains.get_mut(chain_idx) {
        chain.sim.zero_velocity();
        chain.sim.begin_relax();
    }
}
