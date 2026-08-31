mod wgpu;

pub use wgpu::WgpuVisualizer;

use super::{DebugView, PostProcessSettings, Scene, ShadowSettings, TessSettings};

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
    /// GPU tessellation quality (screen-space edge-length target for height-displaced draws).
    fn tess_settings(&mut self) -> &mut TessSettings;
    /// Post + shadow + tessellation settings together (avoids double-borrow of the visualizer).
    fn all_settings(&mut self) -> (&mut PostProcessSettings, &mut ShadowSettings, &mut TessSettings);
    fn debug_view(&self) -> DebugView;
    fn set_debug_view(&mut self, view: DebugView);
}
