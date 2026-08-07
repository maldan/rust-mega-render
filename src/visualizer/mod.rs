mod wgpu;

pub use wgpu::WgpuVisualizer;

use super::{DebugView, PostProcessSettings, Scene};

pub trait Visualizer {
    fn sync(&mut self, scene: &Scene);
    fn render(&mut self, scene: &Scene, aspect: f32);
    fn post_process(&mut self) -> &mut PostProcessSettings;
    fn debug_view(&self) -> DebugView;
    fn set_debug_view(&mut self, view: DebugView);
}
