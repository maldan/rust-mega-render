#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    pub ao: AoSettings,
    pub bloom: BloomSettings,
    pub tonemap: TonemapSettings,
    pub color_grade: ColorGradeSettings,
    pub vignette: VignetteSettings,
    pub grain: GrainSettings,
    pub fxaa: FxaaSettings,
    pub fog: FogSettings,
}

/// Which screen-space AO algorithm to run when [`AoSettings::enabled`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AoMethod {
    /// Classic hemisphere kernel SSAO (depth-only normals).
    Ssao,
    /// Ground-truth AO (horizon search) using G-buffer normals.
    #[default]
    Gtao,
}

#[derive(Clone, Debug)]
pub struct AoSettings {
    pub enabled: bool,
    pub method: AoMethod,
    pub radius: f32,
    /// SSAO sample bias along the normal.
    pub bias: f32,
    pub intensity: f32,
    /// GTAO: number of slice directions (2..=8).
    pub directions: u32,
    /// GTAO: steps per horizon ray (2..=16).
    pub steps: u32,
    /// GTAO: thickness / falloff scale in view space.
    pub thickness: f32,
}

#[derive(Clone, Debug)]
pub struct BloomSettings {
    pub enabled: bool,
    pub threshold: f32,
    pub intensity: f32,
}

/// Filmic tonemap applied in the composite pass (scene is linear when post is on).
#[derive(Clone, Debug)]
pub struct TonemapSettings {
    pub enabled: bool,
    /// Multiplier before the curve. `1.0` is neutral.
    pub exposure: f32,
    /// `true` = ACES approximation, `false` = Reinhard.
    pub aces: bool,
}

#[derive(Clone, Debug)]
pub struct ColorGradeSettings {
    pub enabled: bool,
    /// `1.0` = unchanged.
    pub contrast: f32,
    /// `1.0` = unchanged.
    pub saturation: f32,
    /// Additive lift after contrast. `0.0` = unchanged.
    pub brightness: f32,
}

#[derive(Clone, Debug)]
pub struct VignetteSettings {
    pub enabled: bool,
    pub intensity: f32,
    /// Higher = softer edge falloff.
    pub smoothness: f32,
}

#[derive(Clone, Debug)]
pub struct GrainSettings {
    pub enabled: bool,
    pub intensity: f32,
}

#[derive(Clone, Debug)]
pub struct FxaaSettings {
    pub enabled: bool,
}

#[derive(Clone, Debug)]
pub struct FogSettings {
    pub enabled: bool,
    pub color: [f32; 3],
    /// Exponential density along view distance.
    pub density: f32,
    /// World Y where fog is densest (floor).
    pub height: f32,
    /// How fast fog thins above `height`.
    pub height_falloff: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            ao: AoSettings {
                enabled: false,
                method: AoMethod::Gtao,
                radius: 0.55,
                bias: 0.05,
                intensity: 1.0,
                directions: 6,
                steps: 8,
                thickness: 1.0,
            },
            bloom: BloomSettings {
                enabled: false,
                threshold: 0.7,
                intensity: 0.6,
            },
            tonemap: TonemapSettings {
                enabled: true,
                exposure: 1.6,
                aces: true,
            },
            color_grade: ColorGradeSettings {
                enabled: false,
                contrast: 1.05,
                saturation: 1.02,
                brightness: 0.0,
            },
            vignette: VignetteSettings {
                enabled: false,
                intensity: 0.15,
                smoothness: 0.7,
            },
            grain: GrainSettings {
                enabled: false,
                intensity: 0.02,
            },
            fxaa: FxaaSettings { enabled: true },
            fog: FogSettings {
                enabled: false,
                color: [0.55, 0.62, 0.72],
                density: 0.035,
                height: 0.0,
                height_falloff: 0.15,
            },
        }
    }
}

impl PostProcessSettings {
    pub fn any_enabled(&self) -> bool {
        self.ao.enabled
            || self.bloom.enabled
            || self.tonemap.enabled
            || self.color_grade.enabled
            || self.vignette.enabled
            || self.grain.enabled
            || self.fxaa.enabled
            || self.fog.enabled
    }
}
