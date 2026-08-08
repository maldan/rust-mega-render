use glam::{Mat4, Vec3};

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    /// Smoothed focus distance used by DOF (world units).
    pub focus_distance: f32,
    /// Desired focus distance; [`Self::tick_focus`] eases `focus_distance` toward this.
    pub focus_target: f32,
    /// Focus pull speed (higher = snappier). `0` = snap each tick.
    pub focus_smooth: f32,
    /// Lens f-number. Smaller = stronger blur when DOF is enabled. Unused by the camera itself.
    pub f_stop: f32,
}

impl Camera {
    pub fn look_at(eye: Vec3, target: Vec3) -> Self {
        let focus_distance = (eye - target).length().max(0.01);
        Self {
            eye,
            target,
            up: Vec3::Y,
            fov_y: 45f32.to_radians(),
            near: 0.1,
            far: 100.0,
            focus_distance,
            focus_target: focus_distance,
            focus_smooth: 6.0,
            f_stop: 8.0,
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

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let proj =
            glam::camera::lh::proj::directx::perspective(self.fov_y, aspect, self.near, self.far);
        let view = glam::camera::lh::view::look_at_mat4(self.eye, self.target, self.up);
        proj * view
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
