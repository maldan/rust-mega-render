use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    Perspective,
    Orthographic,
}

#[derive(Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub projection: Projection,
    /// Ortho half-height in world units (full height = 2 * ortho_size).
    pub ortho_size: f32,
    /// Smoothed focus distance used by DOF (world units).
    pub focus_distance: f32,
    /// Desired focus distance; [`Self::tick_focus`] eases `focus_distance` toward this.
    pub focus_target: f32,
    /// Focus pull speed (higher = snappier). `0` = snap each tick.
    pub focus_smooth: f32,
    /// Lens f-number. Smaller = stronger blur when DOF is enabled. Unused by the camera itself.
    pub f_stop: f32,
    /// VR: explicit view matrix (world -> view), bypassing `eye`/`target`/`up`.
    /// Set per-eye each frame by the XR host loop; `None` for normal desktop cameras.
    pub xr_view: Option<Mat4>,
    /// VR: explicit (usually asymmetric) projection matrix, bypassing `fov_y`/`ortho_size`.
    /// See [`Camera::asymmetric_perspective`].
    pub xr_proj: Option<Mat4>,
}

impl Camera {
    pub fn look_at(eye: Vec3, target: Vec3) -> Self {
        let focus_distance = (eye - target).length().max(0.01);
        let fov_y = 45f32.to_radians();
        let ortho_size = Self::ortho_size_from_distance(focus_distance, fov_y);
        Self {
            eye,
            target,
            up: Vec3::Y,
            fov_y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Perspective,
            ortho_size,
            focus_distance,
            focus_target: focus_distance,
            focus_smooth: 6.0,
            f_stop: 8.0,
            xr_view: None,
            xr_proj: None,
        }
    }

    pub fn orbit(yaw: f32, pitch: f32, dist: f32, target: Vec3) -> Self {
        let pitch = pitch.clamp(-1.55, 1.55);
        // LH: X+ right, Y+ up, Z+ forward
        let eye = target
            + Vec3::new(
                dist * yaw.sin() * pitch.cos(),
                dist * pitch.sin(),
                dist * yaw.cos() * pitch.cos(),
            );
        Self::look_at(eye, target)
    }

    /// Ortho half-height that matches a perspective frustum at `distance`.
    pub fn ortho_size_from_distance(distance: f32, fov_y: f32) -> f32 {
        (distance.max(0.01) * (fov_y * 0.5).tan()).max(0.01)
    }

    /// Distance that matches `ortho_size` under perspective `fov_y`.
    pub fn distance_from_ortho_size(ortho_size: f32, fov_y: f32) -> f32 {
        let half = (fov_y * 0.5).tan().max(1e-4);
        (ortho_size / half).max(0.05)
    }

    /// Keep `ortho_size` in sync with current eye↔target distance (for seamless toggles).
    pub fn sync_ortho_from_distance(&mut self) {
        let dist = (self.eye - self.target).length().max(0.05);
        self.ortho_size = Self::ortho_size_from_distance(dist, self.fov_y);
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        if let Some(p) = self.xr_proj {
            return p;
        }
        let aspect = aspect.max(1e-4);
        match self.projection {
            Projection::Perspective => glam::camera::lh::proj::directx::perspective(
                self.fov_y,
                aspect,
                self.near,
                self.far,
            ),
            Projection::Orthographic => {
                let h = self.ortho_size.max(0.01);
                let w = h * aspect;
                glam::camera::lh::proj::directx::orthographic(
                    -w, w, -h, h, self.near, self.far,
                )
            }
        }
    }

    /// World -> view matrix. Uses [`Self::xr_view`] when set (VR), otherwise
    /// the usual `eye`/`target`/`up` look-at.
    pub fn view(&self) -> Mat4 {
        self.xr_view
            .unwrap_or_else(|| glam::camera::lh::view::look_at_mat4(self.eye, self.target, self.up))
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }

    /// Asymmetric (off-axis) LH perspective matrix for a VR eye, built from
    /// the four half-angles OpenXR reports per view (`XrFovf`: `angle_left`,
    /// `angle_right`, `angle_up`, `angle_down`, all in radians — `angle_left`
    /// and `angle_down` are typically negative). Matches the convention of
    /// [`glam::camera::lh::proj::directx::perspective`] (Z in `[0, 1]`, Y not
    /// flipped) so it drops into the same view/proj pipeline as the desktop
    /// camera.
    pub fn asymmetric_perspective(
        angle_left: f32,
        angle_right: f32,
        angle_up: f32,
        angle_down: f32,
        near: f32,
        far: f32,
    ) -> Mat4 {
        let tan_left = angle_left.tan();
        let tan_right = angle_right.tan();
        let tan_up = angle_up.tan();
        let tan_down = angle_down.tan();

        let tan_width = (tan_right - tan_left).max(1e-6);
        let tan_height = (tan_up - tan_down).max(1e-6);

        let xx = 2.0 / tan_width;
        let yy = 2.0 / tan_height;
        // Off-axis (convergence) shift. Reference derivations of this formula
        // (e.g. OpenXR's `xr_linear.h`) target a right-handed, `-Z`-forward
        // view space where `w' = -view.z`, which flips the sign of this term
        // versus solving the same frustum-edge equations for this engine's
        // left-handed, `+Z`-forward view space (`w' = +view.z`, matching
        // `glam::camera::lh`). Using the RH sign here shifted each eye's
        // frustum outward (away from the nose) instead of inward, adding a
        // constant NDC divergence on top of real per-eye parallax — masked by
        // large natural disparity up close, but dominant (and growing) at
        // distance, where correctly-converged disparity should shrink toward
        // zero. Negate to converge instead of diverge.
        let ax = -(tan_right + tan_left) / tan_width;
        let ay = -(tan_up + tan_down) / tan_height;

        let z_range_inv = 1.0 / (far - near);
        let zz = far * z_range_inv;
        let tz = -near * far * z_range_inv;

        Mat4::from_cols(
            glam::Vec4::new(xx, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, yy, 0.0, 0.0),
            glam::Vec4::new(ax, ay, zz, 1.0),
            glam::Vec4::new(0.0, 0.0, tz, 0.0),
        )
    }

    /// Ease `focus_distance` toward `focus_target`.
    pub fn tick_focus(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        if self.focus_smooth <= 1e-4 || dt <= 0.0 {
            self.focus_distance = self.focus_target;
            return;
        }
        let t = 1.0 - (-self.focus_smooth * dt).exp();
        self.focus_distance += (self.focus_target - self.focus_distance) * t.clamp(0.0, 1.0);
    }

    /// Set `focus_target` to the view-ray hit on a horizontal plane at `plane_y`.
    pub fn autofocus_ground(&mut self, plane_y: f32) {
        let forward = (self.target - self.eye).normalize_or_zero();
        if forward.y.abs() < 1e-5 {
            return;
        }
        let t = (plane_y - self.eye.y) / forward.y;
        if t > self.near {
            self.focus_target = t.clamp(self.near * 2.0, self.far * 0.45);
        }
    }

    /// Set `focus_target` to distance toward a world point.
    pub fn autofocus_point(&mut self, point: Vec3) {
        self.focus_target = (point - self.eye).length().max(self.near * 2.0);
    }
}
