use glam::Vec3;

pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
}

/// Direction the light travels (from source toward the scene).
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub enabled: bool,
    /// Whether this light may cast shadows (technique is up to the visualizer).
    pub cast_shadows: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.4, -1.0, -0.3).normalize(),
            color: [1.0, 0.98, 0.92],
            intensity: 1.0,
            enabled: true,
            cast_shadows: true,
        }
    }
}

pub struct PointLight {
    pub position: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub enabled: bool,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 0.0),
            color: [1.0, 0.85, 0.6],
            intensity: 3.0,
            range: 10.0,
            enabled: true,
        }
    }
}

impl Light {
    pub fn directional(dir: DirectionalLight) -> Self {
        Self::Directional(dir)
    }

    pub fn point(point: PointLight) -> Self {
        Self::Point(point)
    }
}
