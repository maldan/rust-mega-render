#![allow(unused_imports)] // module facade re-exports

mod animation;
mod camera;
mod debug_draw;
mod debug_view;
mod gltf_load;
mod hud;
mod hud_font;
mod ibl;
mod input;
mod light;
mod material;
mod mesh;
mod node;
mod post_process;
mod primitives;
mod scene;
mod shadow;
mod skin;
mod store;
mod texture;
pub mod view_gizmo;
mod visualizer;
#[cfg(feature = "xr")]
pub mod xr;

pub use animation::{AnimChannel, AnimPath, AnimValues, AnimationClip, Animator};
pub use camera::{Camera, Projection};
pub use debug_draw::{
    gizmo_ring_basis, gizmo_screen_size, DebugDraw, DebugLine, DebugPoint, DebugTri, GizmoAxis,
    GizmoMode, GizmoOpts, GizmoRotateArc, LineOpts, PolyOpts,
};
pub use debug_view::DebugView;
pub use gltf_load::load_gltf;
pub use hud::{Hud, HudId, HudLine, HudOutput, HudQuad, Rect as HudRect};
pub use input::InputFrame;
pub use light::{DirectionalLight, Light, PointLight};
pub use material::Material;
pub use mesh::{Mesh, MorphTarget};
pub use node::{Node, Transform};
pub use post_process::{
    AoMethod, AoSettings, BloomSettings, ColorGradeSettings, ContactShadowSettings, DofSettings,
    EnvMapSettings, FogSettings, FxaaSettings, GrainSettings, MotionBlurSettings,
    PostProcessSettings, SsgiQuality, SsgiSettings, SsrSettings, TonemapSettings, VignetteSettings,
};
pub use primitives::{cube, plane, sphere};
pub use scene::Scene;
pub use shadow::{ShadowFilter, ShadowSettings};
pub use skin::{
    blend_skin_matrix, blend_skin_point, blend_skin_vector, skin_mesh_matrix, skin_mesh_point,
    DualQuat, Skin, SkinningMode,
};
pub use store::{Handle, Store};
pub use texture::Texture;
pub use view_gizmo::ViewAxis;
pub use visualizer::{Visualizer, WgpuVisualizer};
