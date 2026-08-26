use super::curve::{ease_in_out, HairColorStop, HairCurve, HairCurvePoint, HairCurvePreset};
use crate::material::HairShading;
use glam::Vec3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HairShape {
    Ribbon,
    Tube,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum HairStyle {
    #[default]
    Straight,
    Roll,
    Curl,
    Wave,
    Crimp,
    Coil,
    Braid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandRange {
    pub min: f32,
    pub max: f32,
}

impl RandRange {
    pub const ZERO: RandRange = RandRange { min: 0.0, max: 0.0 };

    pub fn is_active(&self) -> bool {
        self.min.abs() > 1e-5 || self.max.abs() > 1e-5
    }
}

impl Default for RandRange {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LayerRandom {
    pub length: RandRange,
    pub width: RandRange,
    pub roll: RandRange,
    pub offset: Vec3,
    pub rotate: Vec3,
}

#[derive(Clone, Copy)]
pub struct HairGuidePoint {
    pub pos: Vec3,
    pub normal: Vec3,
}

impl HairGuidePoint {
    fn mirrored_x(self) -> Self {
        Self {
            pos: Vec3::new(-self.pos.x, self.pos.y, self.pos.z),
            normal: Vec3::new(-self.normal.x, self.normal.y, self.normal.z).normalize_or_zero(),
        }
    }
}

#[derive(Clone)]
pub struct HairGuide {
    pub points: Vec<HairGuidePoint>,
    pub mirror_x: bool,
    pub lift: f32,
    pub width: f32,
    pub fill_with: Option<usize>,
    /// No bones. Cards go on a separate unskinned mesh (rest pose).
    pub is_static: bool,
}

impl HairGuide {
    pub fn mirrored_x(&self) -> Self {
        Self {
            points: self.points.iter().copied().map(HairGuidePoint::mirrored_x).collect(),
            mirror_x: false,
            lift: self.lift,
            width: self.width,
            fill_with: None,
            is_static: self.is_static,
        }
    }
}

/// Mesh + fiber-card + hair-material parameters used to rebuild a layer.
#[derive(Clone)]
pub struct HairParams {
    pub segments: u32,
    pub density: u32,
    pub lift_curve: HairCurve,
    pub width_curve: HairCurve,
    pub section_curve: HairCurve,
    pub lift_mult: f32,
    pub width_mult: f32,
    pub shape: HairShape,
    pub color_stops: Vec<HairColorStop>,
    pub card_strands: u32,
    pub tex_width: u32,
    pub tex_height: u32,
    pub fiber_gap: f32,
    pub fiber_gap_shade: f32,
    pub roughness: f32,
    pub rough_variance: f32,
    pub fiber_waviness: f32,
    pub fiber_width_variance: f32,
    pub fiber_overlap: f32,
    pub fiber_shade_variance: f32,
    pub fiber_blur: f32,
    pub normal_strength: f32,
    pub smooth: f32,
    pub fill_curve: f32,
    pub style: HairStyle,
    pub curl: f32,
    pub curl_start: f32,
    pub curl_radius: f32,
    pub roll: f32,
    pub roll_start: f32,
    pub wave_amp: f32,
    pub wave_freq: f32,
    pub wave_start: f32,
    pub crimp_amp: f32,
    pub crimp_freq: f32,
    pub crimp_start: f32,
    pub coil_turns: f32,
    pub coil_start: f32,
    pub coil_radius: f32,
    pub coil_taper: f32,
    /// Braid: ply count around the guide spine (2 = twist, 3 = classic).
    pub braid_strands: u32,
    pub braid_turns: f32,
    pub braid_start: f32,
    pub braid_radius: f32,
    /// 0 = round rope, 1 = flattened plait.
    pub braid_flatten: f32,
    pub tip_density: f32,
    pub multiply: u32,
    pub layers: u32,
    pub layer_gap: f32,
    pub layer_rand: Vec<LayerRandom>,
    pub layer_alpha: Vec<f32>,
    pub seed: u32,
    pub hair_shading: HairShading,
    pub tip_fade: f32,
    pub soft_blend: bool,
    pub cutout: f32,
    pub cutout_fringe: f32,
}

impl HairParams {
    pub fn tex_size(&self) -> (u32, u32) {
        (
            self.tex_width.clamp(8, 4096),
            self.tex_height.clamp(8, 4096),
        )
    }

    pub fn auto_stack_count(&self) -> usize {
        (self.layers.min(16) + 1) as usize
    }
}

impl Default for HairParams {
    fn default() -> Self {
        Self {
            segments: 8,
            density: 2,
            lift_curve: ease_in_out(),
            width_curve: default_width_curve(),
            section_curve: default_section_curve(),
            lift_mult: 1.0,
            width_mult: 1.0,
            shape: HairShape::Ribbon,
            card_strands: 256,
            tex_width: 512,
            tex_height: 1024,
            fiber_gap: 0.94,
            fiber_gap_shade: 0.41,
            roughness: 0.45,
            rough_variance: 0.35,
            fiber_waviness: 12.0,
            fiber_width_variance: 0.0,
            fiber_overlap: 0.5,
            fiber_shade_variance: 1.0,
            fiber_blur: 0.0,
            normal_strength: 0.6,
            smooth: 0.35,
            fill_curve: 0.0,
            style: HairStyle::Straight,
            curl: 1.5,
            curl_start: 0.45,
            curl_radius: 0.02,
            roll: 0.8,
            roll_start: 0.65,
            wave_amp: 0.018,
            wave_freq: 2.5,
            wave_start: 0.12,
            crimp_amp: 0.006,
            crimp_freq: 8.0,
            crimp_start: 0.04,
            coil_turns: 4.0,
            coil_start: 0.22,
            coil_radius: 0.012,
            coil_taper: 0.65,
            braid_strands: 3,
            braid_turns: 5.0,
            braid_start: 0.06,
            braid_radius: 0.014,
            braid_flatten: 0.72,
            tip_density: 2.0,
            multiply: 1,
            layers: 0,
            layer_gap: 0.05,
            layer_rand: vec![LayerRandom::default()],
            layer_alpha: vec![1.0],
            seed: 1,
            hair_shading: HairShading::default(),
            tip_fade: 0.09,
            soft_blend: false,
            cutout: 0.4,
            cutout_fringe: 0.85,
            color_stops: vec![
                HairColorStop {
                    t: 0.0,
                    color: [0.22, 0.09, 0.04, 1.0],
                },
                HairColorStop {
                    t: 1.0,
                    color: [0.08, 0.03, 0.02, 1.0],
                },
            ],
        }
    }
}

fn default_section_curve() -> HairCurve {
    HairCurve {
        points: vec![
            HairCurvePoint {
                t: 0.0,
                v: 0.2,
                tangent_out: 0.0,
            },
            HairCurvePoint {
                t: 0.5,
                v: 1.0,
                tangent_out: 0.0,
            },
            HairCurvePoint {
                t: 1.0,
                v: 0.2,
                tangent_out: 0.0,
            },
        ],
        preset: HairCurvePreset::Custom,
    }
}

fn default_width_curve() -> HairCurve {
    HairCurve {
        points: vec![
            HairCurvePoint {
                t: 0.0,
                v: 1.0,
                tangent_out: 0.0,
            },
            HairCurvePoint {
                t: 1.0,
                v: 0.35,
                tangent_out: 0.0,
            },
        ],
        preset: HairCurvePreset::Custom,
    }
}

pub fn fill_pairs_of(guides: &[HairGuide]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, g) in guides.iter().enumerate() {
        let Some(j) = g.fill_with else {
            continue;
        };
        if j >= guides.len() || j == i {
            continue;
        }
        let p = (i.min(j), i.max(j));
        if !pairs.contains(&p) {
            pairs.push(p);
        }
    }
    pairs
}
