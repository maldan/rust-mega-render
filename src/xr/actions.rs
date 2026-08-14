//! Controller input: grip poses (for later hand visualization) and thumbsticks
//! (for locomotion). Bindings are suggested for every common SteamVR controller
//! profile with an analog stick/trackpad so this works across Index, Touch,
//! WMR and Vive wands without per-headset code in the host app.

use super::session::XrError;
use glam::{Quat, Vec2, Vec3};
use openxr as xr;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Hand {
    Left,
    Right,
}

/// World-space (stage-relative) pose of a controller.
#[derive(Clone, Copy, Debug)]
pub struct HandPose {
    pub position: Vec3,
    pub orientation: Quat,
}

pub struct XrActions {
    set: xr::ActionSet,
    /// Kept alive for `left_space`/`right_space` (each holds a ref to the
    /// action's parent set); not read directly elsewhere.
    #[allow(dead_code)]
    grip_pose: xr::Action<xr::Posef>,
    stick: xr::Action<xr::Vector2f>,
    select: xr::Action<f32>,
    left_path: xr::Path,
    right_path: xr::Path,
    left_space: xr::Space,
    right_space: xr::Space,
}

impl XrActions {
    pub fn new(xr_instance: &xr::Instance, session: &xr::Session<xr::Vulkan>) -> Result<Self, XrError> {
        let set = xr_instance.create_action_set("input", "Input", 0)?;
        let left_path = xr_instance.string_to_path("/user/hand/left")?;
        let right_path = xr_instance.string_to_path("/user/hand/right")?;
        let hand_paths = [left_path, right_path];

        let grip_pose =
            set.create_action::<xr::Posef>("grip_pose", "Grip Pose", &hand_paths)?;
        let stick = set.create_action::<xr::Vector2f>("stick", "Thumbstick", &hand_paths)?;
        let select = set.create_action::<f32>("select", "Select", &hand_paths)?;

        // Cover the interaction profiles SteamVR is likely to expose a controller
        // through — thumbstick-based (Touch/Index/WMR) and trackpad-based (Vive).
        suggest(
            xr_instance,
            "/interaction_profiles/oculus/touch_controller",
            &grip_pose,
            &stick,
            &select,
            "thumbstick",
            "squeeze/value",
            left_path,
            right_path,
        )?;
        suggest(
            xr_instance,
            "/interaction_profiles/valve/index_controller",
            &grip_pose,
            &stick,
            &select,
            "thumbstick",
            "squeeze/value",
            left_path,
            right_path,
        )?;
        suggest(
            xr_instance,
            "/interaction_profiles/microsoft/motion_controller",
            &grip_pose,
            &stick,
            &select,
            "thumbstick",
            "squeeze/click",
            left_path,
            right_path,
        )?;
        suggest(
            xr_instance,
            "/interaction_profiles/htc/vive_controller",
            &grip_pose,
            &stick,
            &select,
            "trackpad",
            "squeeze/click",
            left_path,
            right_path,
        )?;

        session.attach_action_sets(&[&set])?;

        let left_space = grip_pose.create_space(session, left_path, xr::Posef::IDENTITY)?;
        let right_space = grip_pose.create_space(session, right_path, xr::Posef::IDENTITY)?;

        Ok(Self {
            set,
            grip_pose,
            stick,
            select,
            left_path,
            right_path,
            left_space,
            right_space,
        })
    }

    /// Call once per frame (after [`super::session::XrContext::poll_events`], before reading state).
    pub fn sync(&self, session: &xr::Session<xr::Vulkan>) -> Result<(), XrError> {
        session.sync_actions(&[(&self.set).into()])?;
        Ok(())
    }

    fn hand_path(&self, hand: Hand) -> xr::Path {
        match hand {
            Hand::Left => self.left_path,
            Hand::Right => self.right_path,
        }
    }

    fn hand_space(&self, hand: Hand) -> &xr::Space {
        match hand {
            Hand::Left => &self.left_space,
            Hand::Right => &self.right_space,
        }
    }

    /// Thumbstick/trackpad axes in `[-1, 1]` (x = left/right, y = forward/back push).
    pub fn stick(&self, session: &xr::Session<xr::Vulkan>, hand: Hand) -> Vec2 {
        match self.stick.state(session, self.hand_path(hand)) {
            Ok(s) if s.is_active => Vec2::new(s.current_state.x, s.current_state.y),
            _ => Vec2::ZERO,
        }
    }

    /// Trigger/squeeze pull, `0..1`.
    pub fn select(&self, session: &xr::Session<xr::Vulkan>, hand: Hand) -> f32 {
        match self.select.state(session, self.hand_path(hand)) {
            Ok(s) if s.is_active => s.current_state,
            _ => 0.0,
        }
    }

    /// Controller pose relative to `base` (normally [`super::session::XrContext::stage`]).
    pub fn hand_pose(
        &self,
        base: &xr::Space,
        hand: Hand,
        time: xr::Time,
    ) -> Option<HandPose> {
        let loc = self.hand_space(hand).locate(base, time).ok()?;
        if !loc
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID)
        {
            return None;
        }
        Some(HandPose {
            position: Vec3::new(loc.pose.position.x, loc.pose.position.y, loc.pose.position.z),
            orientation: Quat::from_xyzw(
                loc.pose.orientation.x,
                loc.pose.orientation.y,
                loc.pose.orientation.z,
                loc.pose.orientation.w,
            ),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn suggest(
    xr_instance: &xr::Instance,
    profile: &str,
    grip_pose: &xr::Action<xr::Posef>,
    stick: &xr::Action<xr::Vector2f>,
    select: &xr::Action<f32>,
    stick_component: &str,
    select_component: &str,
    left_path: xr::Path,
    right_path: xr::Path,
) -> Result<(), XrError> {
    let profile_path = xr_instance.string_to_path(profile)?;
    let mut bindings = Vec::new();
    for (hand, hand_path) in [("left", left_path), ("right", right_path)] {
        let grip = xr_instance.string_to_path(&format!("/user/hand/{hand}/input/grip/pose"))?;
        let stick_p =
            xr_instance.string_to_path(&format!("/user/hand/{hand}/input/{stick_component}"))?;
        let select_p =
            xr_instance.string_to_path(&format!("/user/hand/{hand}/input/{select_component}"))?;
        let _ = hand_path;
        bindings.push(xr::Binding::new(grip_pose, grip));
        bindings.push(xr::Binding::new(stick, stick_p));
        bindings.push(xr::Binding::new(select, select_p));
    }
    // Not every runtime knows every profile (e.g. no Vive driver loaded) — treat
    // that as a no-op rather than a hard error, matching the OpenXR spec's intent
    // that suggesting an unsupported profile is harmless.
    let _ = xr_instance.suggest_interaction_profile_bindings(profile_path, &bindings);
    Ok(())
}
