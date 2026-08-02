use super::store::Handle;
use super::texture::Texture;

pub struct Material {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
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
            albedo_map: None,
            normal_map: None,
            metallic_roughness_map: None,
        }
    }

    pub fn with_map(mut self, map: Handle<Texture>) -> Self {
        self.albedo_map = Some(map);
        self
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0], 0.0, 0.5)
    }
}
