use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    Perspective,
    Orthographic,
}

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

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = glam::camera::lh::view::look_at_mat4(self.eye, self.target, self.up);
        self.proj(aspect) * view
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
