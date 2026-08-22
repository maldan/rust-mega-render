use super::store::Handle;
use super::texture::Texture;

/// Distinguishes the fragment shading path a material is drawn with. `Standard`
/// materials go through the opaque G-buffer pipeline (alpha-cutout only);
/// `Hair` materials are routed to a dedicated depth-prepass + alpha-blend pass
/// with a Scheuermann-style dual-specular (shifted-tangent Kajiya-Kay) shading model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadingModel {
    Standard,
    Hair(HairShading),
}

/// Dual-specular hair shading parameters (Scheuermann / shifted-tangent Kajiya-Kay).
/// Each specular lobe is computed from a tangent shifted along the normal by
/// `*_shift`, giving the characteristic bright "primary" highlight plus a
/// broader, tinted "secondary" highlight seen on real hair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairShading {
    /// Tangent shift for the bright primary highlight (usually small negative).
    pub primary_shift: f32,
    /// Tangent shift for the tinted secondary highlight (usually more negative).
    pub secondary_shift: f32,
    /// Specular exponent (sharpness) of the primary lobe.
    pub primary_exponent: f32,
    /// Specular exponent (sharpness) of the secondary lobe.
    pub secondary_exponent: f32,
    /// Tint color of the secondary lobe (commonly warm/gold).
    pub secondary_tint: [f32; 3],
    /// Intensity multiplier of the secondary lobe.
    pub secondary_strength: f32,
    /// Soften ribbon tips over this fraction of strand length (0..1).
    /// Applied in the hair shader from UV.v — not baked into the mask texture,
    /// so it doesn't fight the depth-prepass alpha cutoff.
    pub tip_fade: f32,
    /// Skip the hair depth prepass so cards soft-alpha-blend over each other
    /// (draw-order composite). Depth against opaque scene still applies.
    pub soft_blend: bool,
    /// Soft fringe width below alpha_cutoff when soft_blend is off (0 = hard,
    /// 1 = fringe from ~0 to cutoff). Packed into object.params.w for the shader.
    pub cutout_fringe: f32,
}

impl Default for HairShading {
    fn default() -> Self {
        Self {
            primary_shift: 0.345,
            secondary_shift: 0.46,
            primary_exponent: 1400.0,
            secondary_exponent: 188.0,
            secondary_tint: [1.0, 1.0, 1.0],
            secondary_strength: 0.24,
            tip_fade: 0.0,
            soft_blend: false,
            cutout_fringe: 0.85,
        }
    }
}

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
    /// 0 = opaque. If > 0, fragments with albedo alpha below this are discarded (MASK).
    /// Deferred path has no alpha blend; this is the supported transparency mode.
    pub alpha_cutoff: f32,
    pub shading_model: ShadingModel,
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
            alpha_cutoff: 0.0,
            shading_model: ShadingModel::Standard,
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
