//! GPU compute tessellation + height displacement (no skin / UDIM / shadow tess).
//!
//! Fly across the plane: near triangles tess more than far ones.
//!
//! ```text
//! cargo run --example tess_demo
//! ```

#[path = "framework.rs"]
mod framework;

use framework::{Demo, Host, SCENE_TEX, UiCtx};
use glam::Vec2;
use mega_render::{
    cube, DebugView, Light, Material, Mesh, Node, Scene, Texture, Transform, Visualizer,
};
use mega_ui::{DockNode, DockState, ScrollAxes, TextStyle, Ui};

/// `segs×segs` quads, same winding/UV as [`mega_render::plane`].
fn grid_plane(w: f32, h: f32, segs: u32) -> Mesh {
    let segs = segs.max(1);
    let hw = w * 0.5;
    let hh = h * 0.5;
    let n = [0.0, 1.0, 0.0];
    let cols = segs + 1;
    let mut positions = Vec::with_capacity((cols * cols) as usize);
    let mut normals = Vec::with_capacity((cols * cols) as usize);
    let mut uvs = Vec::with_capacity((cols * cols) as usize);
    for z in 0..cols {
        let tv = z as f32 / segs as f32;
        let pz = hh - tv * h;
        for x in 0..cols {
            let tu = x as f32 / segs as f32;
            positions.push([-hw + tu * w, 0.0, pz]);
            normals.push(n);
            uvs.push([tu, tv]);
        }
    }
    let mut indices = Vec::with_capacity((segs * segs * 6) as usize);
    for z in 0..segs {
        for x in 0..segs {
            let i = z * cols + x;
            let a = i;
            let b = i + 1;
            let c = i + cols + 1;
            let d = i + cols;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    Mesh::new(positions, normals, uvs, indices)
}

fn hills(w: u32, h: u32) -> Texture {
    let w = w.max(1);
    let h = h.max(1);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let u = x as f32 / w as f32;
            let v = y as f32 / h as f32;
            let a = (u * std::f32::consts::TAU * 3.0).sin() * 0.5 + 0.5;
            let b = (v * std::f32::consts::TAU * 2.0).sin() * 0.5 + 0.5;
            let c = ((u * 37.0 + v * 19.0).sin() * 0.5 + 0.5) * 0.25;
            let n = (a * b * 0.85 + c).clamp(0.0, 1.0);
            let g = (n * 255.0) as u8;
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = g;
            rgba[i + 1] = g;
            rgba[i + 2] = g;
            rgba[i + 3] = 255;
        }
    }
    Texture {
        id: Texture::new_id(),
        width: w,
        height: h,
        rgba,
        version: 1,
        srgb: false,
        dirty: None,
        gpu_resident: false,
    }
}

fn default_dock() -> DockState {
    DockState::new(DockNode::split_h(
        0.72,
        DockNode::leaf(&["Viewport"]),
        DockNode::leaf(&["Debug"]),
    ))
}

struct TessDemo;

impl Demo for TessDemo {
    fn title() -> &'static str {
        "mega-render tessellation"
    }

    fn build_scene() -> Scene {
        let mut scene = Scene::new();
        if let Some(Light::Directional(d)) = scene.lights.first_mut() {
            d.intensity = 3.0;
            d.direction = glam::Vec3::new(0.35, -0.55, 0.76);
        }
        scene.ambient = [0.06, 0.07, 0.08];
        scene.camera.eye = glam::Vec3::new(0.0, 7.0, 16.0);
        scene.camera.target = glam::Vec3::new(0.0, 1.0, 0.0);

        let height = scene.textures.insert(hills(256, 256));
        let ground = scene.meshes.insert(grid_plane(24.0, 24.0, 16));
        let box_m = scene.meshes.insert(cube(1.2));

        let mat_terrain = scene.materials.insert(
            Material::new([0.45, 0.52, 0.38, 1.0], 0.0, 0.7).with_height(height, 2.4),
        );
        let mat_box = scene
            .materials
            .insert(Material::new([0.75, 0.22, 0.18, 1.0], 0.0, 0.4));

        let root = scene.nodes.insert(Node {
            id: Node::new_id(),
            name: "root".into(),
            parent: None,
            local: Transform::default(),
            mesh: None,
            material: None,
            skin: None,
            visible: true,
        });
        scene.nodes.insert(Node {
            id: Node::new_id(),
            name: "terrain".into(),
            parent: Some(root),
            local: Transform::default(),
            mesh: Some(ground),
            material: Some(mat_terrain),
            skin: None,
            visible: true,
        });
        scene.nodes.insert(Node {
            id: Node::new_id(),
            name: "box".into(),
            parent: Some(root),
            local: Transform::from_translation(glam::Vec3::new(0.0, 0.6, 0.0)),
            mesh: Some(box_m),
            material: Some(mat_box),
            skin: None,
            visible: true,
        });
        scene
    }

    fn configure(visualizer: &mut mega_render::WgpuVisualizer) {
        visualizer.set_debug_view(DebugView::Wireframe);
        let post = visualizer.post_process();
        post.tonemap.enabled = true;
        post.tonemap.aces = true;
        post.ao.enabled = false;
        post.ssgi.enabled = false;
        post.ssr.enabled = false;
        post.bloom.enabled = false;
        post.dof.enabled = false;
    }

    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool {
        let fps = ctx.fps;
        let frame_ms = ctx.frame_ms;
        let status_h = 24.0 * ui.scale();
        let dock_size = Vec2::new(ctx.window_size.x, (ctx.window_size.y - status_h).max(1.0));
        let UiCtx {
            scene,
            debug_view,
            dock,
            viewport_size,
            stats,
            tess,
            ..
        } = ctx;

        ui.dock_space("main", dock_size, dock, |ui, tab| match tab {
            "Viewport" => {
                ui.label_styled(
                    &format!("Tess · {}", (**debug_view).label()),
                    TextStyle {
                        color: [0.85, 0.85, 0.85, 1.0],
                        size: 14.0,
                    },
                );
                ui.separator();
                let size = ui.available_size();
                **viewport_size = size;
                ui.texture(SCENE_TEX, size);
            }
            "Debug" => {
                let size = ui.available_size();
                ui.scroll_area("tess_dbg", size, ScrollAxes::Vertical, |ui| {
                    ui.label("Screen-space LOD: subdivides until each edge is ~target_px long.");
                    ui.label("Box has no height map → original mesh.");
                    ui.separator();
                    ui.label("Tess target (px/edge)");
                    ui.slider("tess_target_px", &mut tess.target_px, 1.0..=64.0);
                    ui.separator();
                    let mut idx = DebugView::ALL
                        .iter()
                        .position(|v| *v == **debug_view)
                        .unwrap_or(0);
                    let labels: Vec<&str> = DebugView::ALL.iter().map(|v| v.label()).collect();
                    if ui.select("debug_view", &mut idx, &labels).changed() {
                        **debug_view = DebugView::ALL[idx];
                    }
                    ui.separator();
                    if let Some((_, node)) = scene.nodes.iter().find(|(_, n)| n.name == "terrain") {
                        if let Some(mh) = node.material {
                            if let Some(mat) = scene.materials.get_mut(mh) {
                                ui.label("Displacement scale");
                                ui.slider("scale", &mut mat.displacement_scale, 0.0..=6.0);
                            }
                        }
                    }
                });
            }
            other => ui.label(other),
        });

        ui.status_bar(|ui| {
            ui.label("RMB look · WASD move · Esc quit");
            ui.label("·");
            ui.label(&format!("FPS {:.0}", fps));
            ui.label("·");
            ui.label(&format!("{:.1} ms", frame_ms));
            ui.label("·");
            ui.label(&format!("ui {} / {}", stats.batches, stats.quads));
        });
        true
    }
}

fn main() {
    Host::<TessDemo>::run(default_dock());
}
