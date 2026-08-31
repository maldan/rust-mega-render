//! Visualizer GPU tessellation quality settings (not part of scene/material data).
//!
//! [`Material::tess_factor`](crate::Material::tess_factor) still caps how far a
//! single draw is allowed to subdivide. Per-edge level is the minimum of:
//! screen-space size ([`TessSettings::target_px`]), height-map curvature
//! (linear interpolant already good enough → stay coarse), and height-texel
//! span (no extra verts beyond the map).

/// Backend tessellation quality knobs, shared by all height-displaced draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TessSettings {
    /// When `false`, GPU tessellation is skipped for every draw even if the
    /// material has a height map and [`crate::HeightMode::Tessellate`]. The base
    /// mesh is drawn as-is (albedo/normal/MR still apply). Parallax is unchanged.
    pub enabled: bool,
    /// Pixel budget for both projected edge length *and* leftover height
    /// error after linear interpolation. Smaller = denser where the height
    /// map actually bends / higher quality / more GPU work; larger = coarser.
    /// Typical range is roughly `4.0` (high quality) to `24.0` (low quality);
    /// `8.0..=12.0` is a reasonable default for most content.
    ///
    /// This is independent of [`Material::tess_factor`](crate::Material::tess_factor),
    /// which remains a hard per-draw ceiling regardless of how small
    /// `target_px` is set.
    pub target_px: f32,
}

impl Default for TessSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            target_px: 10.0,
        }
    }
}

impl TessSettings {
    /// Clamps to a sane range so a bad UI value can't blow up GPU buffers
    /// (extremely small `target_px` combined with a high `tess_factor` cap
    /// still costs at most `tess_factor`, but keeping this away from `0`
    /// avoids division-by-near-zero level computation on the GPU).
    pub fn sanitized_target_px(&self) -> f32 {
        self.target_px.clamp(1.0, 256.0)
    }
}
