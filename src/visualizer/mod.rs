mod wgpu;

pub use wgpu::WgpuVisualizer;

use super::{PostProcessSettings, Scene};

pub trait Visualizer {
    fn sync(&mut self, scene: &Scene);
    fn render(&mut self, scene: &Scene, aspect: f32);
    fn post_process(&mut self) -> &mut PostProcessSettings;
}
