use glam::{Mat4, Vec3};

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn look_at(eye: Vec3, target: Vec3) -> Self {
        Self {
            eye,
            target,
            up: Vec3::Y,
            fov_y: 45f32.to_radians(),
            near: 0.1,
            far: 100.0,
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
}
