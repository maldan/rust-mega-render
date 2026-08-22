mod wgpu;

pub use wgpu::WgpuVisualizer;

use super::{DebugView, PostProcessSettings, Scene, ShadowSettings};

/// Optional GPU features the wgpu visualizer uses (wireframe debug view).
/// Intersect with `adapter.features()` when requesting a device.
pub const WGPU_FEATURES: ::wgpu::Features = ::wgpu::Features::POLYGON_MODE_LINE;

pub trait Visualizer {
    fn sync(&mut self, scene: &Scene);
    fn render(&mut self, scene: &Scene, aspect: f32);
    fn post_process(&mut self) -> &mut PostProcessSettings;
    fn shadow_settings(&mut self) -> &mut ShadowSettings;
    /// Post + shadow settings together (avoids double-borrow of the visualizer).
    fn effect_settings(&mut self) -> (&mut PostProcessSettings, &mut ShadowSettings);
    fn debug_view(&self) -> DebugView;
    fn set_debug_view(&mut self, view: DebugView);
}
