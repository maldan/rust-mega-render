use super::store::Handle;
use super::texture::{Texture, TextureStore};

pub use crate::io::material::{MaterialBytesError, MaterialFile, MaterialFileMaps};

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

/// How this material addresses albedo / normal / MR maps.
/// `Single` is wrap-sampling one texture per slot. `Udim` is opt-in tile sets.
#[derive(Clone)]
pub enum MaterialMaps {
    Single {
        albedo: Option<Handle<Texture>>,
        normal: Option<Handle<Texture>>,
        metallic_roughness: Option<Handle<Texture>>,
    },
    Udim {
        /// `(udim, texture)` e.g. `(1001, tex)`. `u = (id-1001)%10`, `v = (id-1001)/10`.
        albedo: Vec<(u32, Handle<Texture>)>,
        normal: Vec<(u32, Handle<Texture>)>,
        metallic_roughness: Vec<(u32, Handle<Texture>)>,
    },
}

impl Default for MaterialMaps {
    fn default() -> Self {
        Self::Single {
            albedo: None,
            normal: None,
            metallic_roughness: None,
        }
    }
}

impl MaterialMaps {
    pub fn remap_textures(&mut self, map: impl Fn(Handle<Texture>) -> Option<Handle<Texture>>) {
        match self {
            Self::Single {
                albedo,
                normal,
                metallic_roughness,
            } => {
                *albedo = albedo.and_then(&map);
                *normal = normal.and_then(&map);
                *metallic_roughness = metallic_roughness.and_then(&map);
            }
            Self::Udim {
                albedo,
                normal,
                metallic_roughness,
            } => {
                remap_udim_slot(albedo, &map);
                remap_udim_slot(normal, &map);
                remap_udim_slot(metallic_roughness, &map);
            }
        }
    }
}

fn remap_udim_slot(
    tiles: &mut Vec<(u32, Handle<Texture>)>,
    map: &impl Fn(Handle<Texture>) -> Option<Handle<Texture>>,
) {
    tiles.retain_mut(|(_, h)| {
        if let Some(next) = map(*h) {
            *h = next;
            true
        } else {
            false
        }
    });
}

/// How a height map is applied. Tessellation displaces real vertices (correct
/// silhouettes, extra geometry). Parallax only offsets UVs in the fragment
/// shader (cheap, silhouette stays the original mesh).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeightMode {
    #[default]
    Tessellate,
    Parallax,
}

impl HeightMode {
    pub const ALL: [Self; 2] = [Self::Tessellate, Self::Parallax];

    pub fn label(self) -> &'static str {
        match self {
            Self::Tessellate => "Tessellation",
            Self::Parallax => "Parallax",
        }
    }
}

/// LUT index for Mari UDIM in a 10×10 tile grid, or `None` if out of range.
pub(crate) fn udim_lut_index(udim: u32) -> Option<usize> {
    let n = udim.checked_sub(1001)?;
    let u = (n % 10) as usize;
    let v = (n / 10) as usize;
    (v < 10).then_some(u + v * 10)
}

pub(crate) fn first_udim_tile(tiles: &[(u32, Handle<Texture>)]) -> Option<Handle<Texture>> {
    tiles
        .iter()
        .find(|(id, _)| *id == 1001)
        .or(tiles.first())
        .map(|(_, h)| *h)
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
    pub maps: MaterialMaps,
    /// Height map for GPU displacement or parallax. `None` or
    /// [`Self::displacement_scale`] ≤ 0 skips both.
    pub height: Option<Handle<Texture>>,
    /// World-unit height at map value 1. Ignored when [`Self::height`] is `None`.
    pub displacement_scale: f32,
    /// GPU tessellation cap (1..=32), or parallax ray-march steps when
    /// [`Self::height_mode`] is [`HeightMode::Parallax`].
    pub tess_factor: u32,
    /// Tessellate vertices vs offset UVs. Ignored when height is off.
    pub height_mode: HeightMode,
    /// 0 = opaque. If > 0, fragments with albedo alpha below this are discarded (MASK).
    /// Deferred path has no alpha blend; this is the supported transparency mode.
    pub alpha_cutoff: f32,
    pub shading_model: ShadingModel,
}

impl Material {
    /// Lookup texture ids in `textures` and write a `MAT ` blob. Handles are not stored.
    pub fn to_bytes(&self, textures: &TextureStore) -> Vec<u8> {
        MaterialFile::from_material(self, textures).to_bytes()
    }

    pub fn new(albedo: [f32; 4], metallic: f32, roughness: f32) -> Self {
        Self {
            albedo,
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.04, 1.0),
            sss_strength: 0.0,
            sss_color: [1.0, 0.35, 0.2],
            sss_curvature: 0.3,
            maps: MaterialMaps::default(),
            height: None,
            displacement_scale: 0.0,
            tess_factor: 32,
            height_mode: HeightMode::Tessellate,
            alpha_cutoff: 0.0,
            shading_model: ShadingModel::Standard,
        }
    }

    pub fn with_height(mut self, map: Handle<Texture>, scale: f32) -> Self {
        self.height = Some(map);
        self.displacement_scale = scale.max(0.0);
        self.height_mode = HeightMode::Tessellate;
        self
    }

    pub fn with_parallax(mut self, map: Handle<Texture>, scale: f32) -> Self {
        self.height = Some(map);
        self.displacement_scale = scale.max(0.0);
        self.height_mode = HeightMode::Parallax;
        self
    }

    pub fn with_map(mut self, map: Handle<Texture>) -> Self {
        self.maps = MaterialMaps::Single {
            albedo: Some(map),
            normal: None,
            metallic_roughness: None,
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_index_mari() {
        assert_eq!(udim_lut_index(1001), Some(0));
        assert_eq!(udim_lut_index(1002), Some(1));
        assert_eq!(udim_lut_index(1011), Some(10));
        assert_eq!(udim_lut_index(1000), None);
        assert_eq!(udim_lut_index(1101), None);
    }
}
