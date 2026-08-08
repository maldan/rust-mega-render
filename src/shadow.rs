//! Visualizer shadow technique settings (not part of scene lights).

/// Shadow map filtering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowFilter {
    /// Fixed 3×3 percentage-closer filtering (cheap).
    Pcf,
    /// Percentage-closer soft shadows (variable penumbra).
    #[default]
    Pcss,
}

/// Backend shadow quality / technique knobs.
#[derive(Clone, Debug)]
pub struct ShadowSettings {
    pub filter: ShadowFilter,
    /// Shadow map resolution: typically `1024`, `2048`, or `4096`.
    pub map_size: u32,
    /// Receiver depth bias scale (acne ↔ peter-panning). Used as
    /// `max(bias * (1 - n·l), bias * 0.3)`.
    pub bias: f32,
    /// PCSS softness `0` = sharp, `1` = max soft (quadratic curve).
    pub pcss_light_size: f32,
    /// PCSS blocker-search taps (4..=16).
    pub pcss_blocker_samples: u32,
    /// PCSS filter taps (8..=48).
    pub pcss_filter_samples: u32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            filter: ShadowFilter::Pcss,
            map_size: 4096,
            bias: 0.00001,
            pcss_light_size: 0.35,
            pcss_blocker_samples: 16,
            pcss_filter_samples: 48,
        }
    }
}
