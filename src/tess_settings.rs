//! Visualizer GPU tessellation quality settings (not part of scene/material data).
//!
//! [`Material::tess_factor`](crate::Material::tess_factor) still caps how far a
//! single draw is allowed to subdivide, but the actual level chosen per triangle
//! edge is driven by [`TessSettings::target_px`]: the tessellator keeps
//! subdividing until each edge is roughly that many screen pixels long, so
//! nearby-but-small geometry doesn't get over-tessellated and distant-but-large
//! geometry still gets enough detail.

/// Backend tessellation quality knobs, shared by all height-displaced draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TessSettings {
    /// Target on-screen edge length, in pixels, that the tessellator tries to
    /// keep sub-triangle edges under. Smaller = denser mesh / higher quality
    /// / more GPU work; larger = coarser mesh / cheaper. Typical range is
    /// roughly `4.0` (high quality) to `24.0` (low quality); `8.0..=12.0` is a
    /// reasonable default for most content.
    ///
    /// This is independent of [`Material::tess_factor`](crate::Material::tess_factor),
    /// which remains a hard per-draw ceiling regardless of how small
    /// `target_px` is set.
    pub target_px: f32,
}

impl Default for TessSettings {
    fn default() -> Self {
        Self { target_px: 10.0 }
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
