#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    pub env: EnvMapSettings,
    pub ao: AoSettings,
    pub contact_shadow: ContactShadowSettings,
    pub ssgi: SsgiSettings,
    pub ssr: SsrSettings,
    pub bloom: BloomSettings,
    pub tonemap: TonemapSettings,
    pub color_grade: ColorGradeSettings,
    pub vignette: VignetteSettings,
    pub grain: GrainSettings,
    pub fxaa: FxaaSettings,
    pub fog: FogSettings,
    pub dof: DofSettings,
}

/// Equirect env reflections + skybox (no heavy IBL bake).
#[derive(Clone, Debug)]
pub struct EnvMapSettings {
    pub enabled: bool,
    /// Scales reflections and skybox.
    pub intensity: f32,
    /// Yaw of the env map in degrees (0..=360).
    pub rotation_y: f32,
}

impl Default for EnvMapSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity: 1.0,
            rotation_y: 0.0,
        }
    }
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

/// Screen-space contact shadows along the primary directional light.
#[derive(Clone, Debug)]
pub struct ContactShadowSettings {
    pub enabled: bool,
    /// Ray length in world units.
    pub length: f32,
    /// Depth thickness tolerance (world units) for hit acceptance.
    pub thickness: f32,
    pub intensity: f32,
    /// March steps (4..=32).
    pub samples: u32,
    /// Start offset along the ray to reduce self-occlusion.
    pub bias: f32,
}

/// Screen-space reflections with env-map fallback (confidence blend).
#[derive(Clone, Debug)]
pub struct SsrSettings {
    pub enabled: bool,
    /// Max ray length in world / view units.
    pub max_distance: f32,
    /// Depth thickness tolerance (view units) for hit acceptance.
    pub thickness: f32,
    /// Additive strength in composite.
    pub intensity: f32,
    /// March steps (8..=64).
    pub max_steps: u32,
    /// Start offset along the ray to reduce self-hits.
    pub bias: f32,
    /// Skip screen-space march above this roughness (env-only specular).
    pub roughness_cutoff: f32,
    /// Camera-reprojection temporal accumulation.
    pub temporal: bool,
    /// Blend toward history (0 = current only, ~0.9 = stable).
    pub history: f32,
    /// Relative clip-depth rejection threshold for disocclusion.
    pub depth_reject: f32,
}

/// Spatial screen-space global illumination (optional temporal accumulation).
#[derive(Clone, Debug)]
pub struct SsgiSettings {
    pub enabled: bool,
    /// Max ray length in world units.
    pub radius: f32,
    /// Depth thickness tolerance (world units) for hit acceptance.
    pub thickness: f32,
    /// Additive strength in composite.
    pub intensity: f32,
    /// Hemisphere rays per pixel (4..=32).
    pub samples: u32,
    /// March steps along each ray (4..=32).
    pub max_steps: u32,
    /// Start offset along the ray to reduce self-hits.
    pub bias: f32,
    /// How much to reduce constant ambient (0..=1) to avoid double-counting with SSGI.
    pub ambient_dim: f32,
    /// Camera-reprojection temporal accumulation.
    pub temporal: bool,
    /// Blend toward history (0 = current only, ~0.9 = stable).
    pub history: f32,
    /// Relative clip-depth rejection threshold for disocclusion.
    pub depth_reject: f32,
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

/// Temporal screen-space depth of field (CoC gather + optional reprojection).
///
/// Focus / f-stop live on [`crate::Camera`]; these tune the post effect only.
#[derive(Clone, Debug)]
pub struct DofSettings {
    pub enabled: bool,
    /// Max blur radius in pixels (full-res).
    pub max_coc_px: f32,
    /// Overall CoC scale (pairs with camera `f_stop`).
    pub scale: f32,
    /// Soft in-focus half-range in world units around `focus_distance`.
    pub focus_range: f32,
    /// Spiral gather taps per pixel (4..=24).
    pub samples: u32,
    /// 0 = circular bokeh, 5..=8 = blade count (hex etc.).
    pub bokeh_blades: u32,
    /// Gather at half resolution (cheaper, still sharp in focus via upsample).
    pub half_res: bool,
    /// Continuously pull focus toward the view-ray / ground (see camera autofocus).
    pub auto_focus: bool,
    /// Camera-reprojection temporal accumulation.
    pub temporal: bool,
    /// Blend toward history (0 = current only, ~0.9 = stable).
    pub history: f32,
    /// Relative clip-depth rejection threshold for disocclusion.
    pub depth_reject: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            env: EnvMapSettings::default(),
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
            contact_shadow: ContactShadowSettings {
                enabled: false,
                length: 0.4,
                thickness: 0.08,
                intensity: 1.0,
                samples: 12,
                bias: 0.002,
            },
            ssgi: SsgiSettings {
                enabled: false,
                radius: 2.5,
                thickness: 0.45,
                intensity: 1.8,
                samples: 6,
                max_steps: 6,
                bias: 0.02,
                ambient_dim: 0.2,
                temporal: true,
                history: 0.88,
                depth_reject: 0.025,
            },
            ssr: SsrSettings {
                enabled: false,
                max_distance: 5.0,
                thickness: 0.24,
                intensity: 0.63,
                max_steps: 37,
                bias: 0.0,
                roughness_cutoff: 0.48,
                temporal: true,
                history: 0.98,
                depth_reject: 0.1,
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
            dof: DofSettings {
                enabled: false,
                max_coc_px: 28.0,
                scale: 12.0,
                focus_range: 0.25,
                samples: 12,
                bokeh_blades: 6,
                half_res: true,
                auto_focus: false,
                temporal: true,
                history: 0.9,
                depth_reject: 0.03,
            },
        }
    }
}

impl PostProcessSettings {
    pub fn any_enabled(&self) -> bool {
        self.ao.enabled
            || self.contact_shadow.enabled
            || self.ssgi.enabled
            || self.ssr.enabled
            || self.bloom.enabled
            || self.tonemap.enabled
            || self.color_grade.enabled
            || self.vignette.enabled
            || self.grain.enabled
            || self.fxaa.enabled
            || self.fog.enabled
            || self.dof.enabled
    }
}
