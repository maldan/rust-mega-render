//! Bake a procedural brick graph from code and apply it to a cube + sphere.
//!
//! ```text
//! cargo run --example texgen_demo
//! ```

#[path = "framework.rs"]
mod framework;

use framework::{Demo, Host, SCENE_TEX, UiCtx};
use glam::Vec2;
use mega_render::texgen::{
    ColorRampParams, GpuEval, GradientStop, NodeKind, OpacityStop, TexGraph,
};
use mega_render::{
    cube_subdiv, sphere, DebugView, Light, Material, Node, Scene, Transform, Visualizer,
    WgpuVisualizer,
};
use mega_ui::{DockNode, DockState, ScrollAxes, TextStyle, Ui};

fn brick_graph() -> TexGraph {
    let mut g = TexGraph::new();
    let out = g.output_id.clone();
    if let Some(n) = g.node_mut(&out) {
        n.displacement = 0.06;
        n.tess_factor = 16;
    }

    let bricks = g.add(NodeKind::Bricks);
    if let Some(n) = g.node_mut(&bricks) {
        n.bricks.x_amount = 5;
        n.bricks.y_amount = 10;
        n.bricks.gap = 0.1;
        n.bricks.bevel = 0.18;
    }

    let ramp = g.add(NodeKind::ColorRamp);
    if let Some(n) = g.node_mut(&ramp) {
        n.color_ramp = ColorRampParams {
            colors: vec![
                GradientStop {
                    t: 0.0,
                    color: [0.22, 0.2, 0.18, 1.0],
                },
                GradientStop {
                    t: 0.45,
                    color: [0.55, 0.28, 0.2, 1.0],
                },
                GradientStop {
                    t: 1.0,
                    color: [0.72, 0.42, 0.28, 1.0],
                },
            ],
            opacities: vec![
                OpacityStop { t: 0.0, alpha: 1.0 },
                OpacityStop { t: 1.0, alpha: 1.0 },
            ],
        };
    }

    let nrm = g.add(NodeKind::HeightToNormal);
    if let Some(n) = g.node_mut(&nrm) {
        n.normal_strength = 2.2;
    }

    let noise = g.add(NodeKind::Noise);
    if let Some(n) = g.node_mut(&noise) {
        n.noise.scale = 8.0;
        n.noise.octaves = 3;
        n.noise.seed = 3.0;
    }

    g.connect(&bricks, "out", &ramp, "fac");
    g.connect(&bricks, "out", &nrm, "height");
    g.connect(&ramp, "out", &out, "albedo");
    g.connect(&nrm, "out", &out, "normal");
    g.connect(&bricks, "out", &out, "height");
    g.connect(&noise, "out", &out, "roughness");
    g
}

fn default_dock() -> DockState {
    DockState::new(DockNode::split_h(
        0.78,
        DockNode::leaf(&["Viewport"]),
        DockNode::leaf(&["Info"]),
    ))
}

struct TexgenDemo;

impl Demo for TexgenDemo {
    fn title() -> &'static str {
        "mega-render texgen"
    }

    fn build_scene() -> Scene {
        let mut scene = Scene::new();
        if let Some(Light::Directional(d)) = scene.lights.first_mut() {
            d.intensity = 3.2;
            d.direction = glam::Vec3::new(0.4, -0.6, 0.7);
        }
        scene.ambient = [0.07, 0.07, 0.08];
        scene.camera.eye = glam::Vec3::new(0.0, 1.4, 4.2);
        scene.camera.target = glam::Vec3::new(0.0, 0.55, 0.0);

        let placeholder = scene
            .materials
            .insert(Material::new([0.5, 0.5, 0.5, 1.0], 0.0, 0.5));
        let mesh_cube = scene.meshes.insert(cube_subdiv(1.1, 8));
        let mesh_sphere = scene.meshes.insert(sphere(0.55, 48, 28));

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
            name: "cube".into(),
            parent: Some(root),
            local: Transform::from_translation(glam::Vec3::new(-0.9, 0.55, 0.0)),
            mesh: Some(mesh_cube),
            material: Some(placeholder),
            skin: None,
            visible: true,
        });
        scene.nodes.insert(Node {
            id: Node::new_id(),
            name: "sphere".into(),
            parent: Some(root),
            local: Transform::from_translation(glam::Vec3::new(0.95, 0.55, 0.0)),
            mesh: Some(mesh_sphere),
            material: Some(placeholder),
            skin: None,
            visible: true,
        });
        scene
    }

    fn configure(visualizer: &mut WgpuVisualizer) {
        let post = visualizer.post_process();
        post.tonemap.enabled = true;
        post.tonemap.aces = true;
        post.ao.enabled = false;
        post.ssgi.enabled = false;
        post.ssr.enabled = false;
        post.bloom.enabled = false;
        post.dof.enabled = false;
    }

    fn on_gpu(visualizer: &mut WgpuVisualizer, scene: &mut Scene) {
        let graph = brick_graph();
        let mut eval = GpuEval::new(visualizer.device(), visualizer.queue());
        let mat = eval.bake_into_scene(
            visualizer.device(),
            visualizer.queue(),
            scene,
            &graph,
            512,
        );
        let handles: Vec<_> = scene
            .nodes
            .iter()
            .filter(|(_, n)| n.name == "cube" || n.name == "sphere")
            .map(|(h, _)| h)
            .collect();
        for h in handles {
            if let Some(n) = scene.nodes.get_mut(h) {
                n.material = Some(mat);
            }
        }
    }

    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool {
        let fps = ctx.fps;
        let frame_ms = ctx.frame_ms;
        let status_h = 24.0 * ui.scale();
        let dock_size = Vec2::new(ctx.window_size.x, (ctx.window_size.y - status_h).max(1.0));
        let UiCtx {
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
                    "Texgen · bricks graph baked on GPU",
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
            "Info" => {
                let size = ui.available_size();
                ui.scroll_area("texgen_info", size, ScrollAxes::Vertical, |ui| {
                    ui.label("Graph: Bricks → ColorRamp (albedo), Height→Normal, Noise (roughness).");
                    ui.label("Same material on cube and sphere.");
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
    Host::<TexgenDemo>::run(default_dock());
}
