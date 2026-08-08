use glam::Vec2;

/// Per-frame input snapshot for HUD / game logic.
///
/// Filled by the host from OS events — the engine never talks to winit.
#[derive(Clone, Debug, Default)]
pub struct InputFrame {
    /// Cursor in viewport pixels (origin top-left).
    pub cursor: Vec2,
    pub mouse_down: bool,
    pub mouse_pressed: bool,
    pub mouse_released: bool,
    /// Wheel delta in pixels (+y = scroll up).
    pub scroll_delta: Vec2,
    pub dt: f32,
}
