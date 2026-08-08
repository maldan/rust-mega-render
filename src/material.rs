use super::store::Handle;
use super::texture::Texture;

pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// 0 = off. Pre-integrated SSS strength on diffuse.
    pub sss_strength: f32,
    /// Scatter tint (skin ≈ warm red).
    pub sss_color: [f32; 3],
    /// Curvature for the SSS LUT (0..1). Higher = thinner / softer wrap.
    pub sss_curvature: f32,
    pub albedo_map: Option<Handle<Texture>>,
    pub normal_map: Option<Handle<Texture>>,
    pub metallic_roughness_map: Option<Handle<Texture>>,
}

impl Material {
    pub fn new(albedo: [f32; 4], metallic: f32, roughness: f32) -> Self {
        Self {
            albedo,
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.04, 1.0),
            sss_strength: 0.0,
            sss_color: [1.0, 0.35, 0.2],
            sss_curvature: 0.3,
            albedo_map: None,
            normal_map: None,
            metallic_roughness_map: None,
        }
    }

    pub fn with_map(mut self, map: Handle<Texture>) -> Self {
        self.albedo_map = Some(map);
        self
    }

    pub fn with_sss(mut self, strength: f32, color: [f32; 3], curvature: f32) -> Self {
        self.sss_strength = strength.clamp(0.0, 1.0);
        self.sss_color = color;
        self.sss_curvature = curvature.clamp(0.001, 1.0);
        self
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0], 0.0, 0.5)
    }
}
