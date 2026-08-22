//! `XR_EXT_hand_tracking` — 26 joints per hand (wrist, palm, fingers).
//!
//! The runtime must advertise the extension *and* report
//! `supports_hand_tracking` for the HMD. Trackers are created on
//! [`crate::xr::XrContext`] and located each frame against stage space.

use super::session::XrFrame;
use glam::{Quat, Vec3};
use openxr as xr;

/// Joint count in the default `XR_EXT_hand_tracking` set.
pub const HAND_JOINT_COUNT: usize = xr::HAND_JOINT_COUNT;

/// One tracked joint in OpenXR stage space (right-handed, `-Z` forward).
#[derive(Clone, Copy, Debug)]
pub struct TrackedJoint {
    pub position: Vec3,
    pub orientation: Quat,
    /// Approximate bone radius at this joint, meters.
    pub radius: f32,
}

/// One hand's joints this frame. Missing entries were not valid this sample.
#[derive(Clone, Copy, Debug)]
pub struct TrackedHand {
    pub joints: [Option<TrackedJoint>; HAND_JOINT_COUNT],
}

impl TrackedHand {
    fn empty() -> Self {
        Self {
            joints: [None; HAND_JOINT_COUNT],
        }
    }

    pub fn get(&self, index: usize) -> Option<TrackedJoint> {
        self.joints.get(index).copied().flatten()
    }
}

/// Stick-figure bones as `(parent, child)` indices into [`TrackedHand::joints`].
/// Matches the default OpenXR hand joint set (`PALM=0` … `LITTLE_TIP=25`).
pub const HAND_BONES: &[(usize, usize)] = &[
    (1, 0), // wrist → palm
    // thumb
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    // index
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    // middle
    (1, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    // ring
    (1, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    // little
    (1, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
];

pub(crate) fn create_trackers(
    instance: &xr::Instance,
    system: xr::SystemId,
    session: &xr::Session<xr::Vulkan>,
    ext_present: bool,
) -> (Option<xr::HandTracker>, Option<xr::HandTracker>) {
    if !ext_present {
        eprintln!("xr: XR_EXT_hand_tracking not advertised by runtime");
        return (None, None);
    }
    match instance.supports_hand_tracking(system) {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("xr: hand tracking extension present, but this HMD reports supports_hand_tracking=false");
            return (None, None);
        }
        Err(e) => {
            eprintln!("xr: supports_hand_tracking query failed: {e}");
            return (None, None);
        }
    }

    let left = match session.create_hand_tracker(xr::Hand::LEFT) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("xr: create_hand_tracker(LEFT) failed: {e}");
            None
        }
    };
    let right = match session.create_hand_tracker(xr::Hand::RIGHT) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("xr: create_hand_tracker(RIGHT) failed: {e}");
            None
        }
    };
    match (left.is_some(), right.is_some()) {
        (true, true) => eprintln!("xr: hand tracking enabled (left+right)"),
        (true, false) => eprintln!("xr: hand tracking enabled (left only)"),
        (false, true) => eprintln!("xr: hand tracking enabled (right only)"),
        (false, false) => {}
    }
    (left, right)
}

pub(crate) fn locate_hand(
    stage: &xr::Space,
    tracker: Option<&xr::HandTracker>,
    time: xr::Time,
) -> Option<TrackedHand> {
    let tracker = tracker?;
    let locations = match stage.locate_hand_joints(tracker, time) {
        Ok(Some(locs)) => locs,
        Ok(None) => return None,
        Err(e) => {
            // Spam-prone if the runtime errors every frame — keep it quiet after
            // the first few would need a flag; a single eprint per call is too
            // much. Swallow transient locate failures.
            let _ = e;
            return None;
        }
    };

    let mut hand = TrackedHand::empty();
    let mut any = false;
    for (i, loc) in locations.iter().enumerate() {
        if !loc.location_flags.contains(
            xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID,
        ) {
            continue;
        }
        let p = loc.pose.position;
        let o = loc.pose.orientation;
        hand.joints[i] = Some(TrackedJoint {
            position: Vec3::new(p.x, p.y, p.z),
            orientation: Quat::from_xyzw(o.x, o.y, o.z, o.w),
            radius: loc.radius,
        });
        any = true;
    }
    any.then_some(hand)
}

pub(crate) fn locate_both(
    stage: &xr::Space,
    left: Option<&xr::HandTracker>,
    right: Option<&xr::HandTracker>,
    frame: &XrFrame,
) -> [Option<TrackedHand>; 2] {
    let time = frame.state.predicted_display_time;
    [
        locate_hand(stage, left, time),
        locate_hand(stage, right, time),
    ]
}
