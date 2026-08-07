//! Visualizer debug output modes (not scene data).
//!
//! These select which internal buffer is shown. A backend may no-op unsupported
//! modes and fall back to [`DebugView::Final`].

/// Which buffer / channel the visualizer presents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DebugView {
    /// Lit scene through the normal present path (post / tonemap).
    #[default]
    Final,
    /// HDR scene color (simple exposure preview, no full post stack).
    SceneColor,
    /// World-space normals (`n * 0.5 + 0.5`).
    Normals,
    /// Metallic-roughness.G (roughness) as grayscale.
    Roughness,
    /// Metallic-roughness.B (metallic) as grayscale.
    Metallic,
    /// Linearized depth preview.
    Depth,
    /// SSAO / AO buffer (requires AO pass; white if unavailable).
    Ao,
}

impl DebugView {
    pub const ALL: &'static [DebugView] = &[
        DebugView::Final,
        DebugView::SceneColor,
        DebugView::Normals,
        DebugView::Roughness,
        DebugView::Metallic,
        DebugView::Depth,
        DebugView::Ao,
    ];

    pub fn label(self) -> &'static str {
        match self {
            DebugView::Final => "Final",
            DebugView::SceneColor => "Scene color",
            DebugView::Normals => "Normals",
            DebugView::Roughness => "Roughness",
            DebugView::Metallic => "Metallic",
            DebugView::Depth => "Depth",
            DebugView::Ao => "AO",
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            DebugView::Final => 0,
            DebugView::SceneColor => 1,
            DebugView::Normals => 2,
            DebugView::Roughness => 3,
            DebugView::Metallic => 4,
            DebugView::Depth => 5,
            DebugView::Ao => 6,
        }
    }
}
