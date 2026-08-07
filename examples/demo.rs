//! Material showcase demo with mega-ui dock (effects panel).
//!
//! Move: WASD · Up/Down: E/Q · Sprint: Shift · Look: RMB (over viewport) · Quit: Esc
//!
//! ```text
//! cargo run --example demo
//! ```
//!
//! `mega-ui` is linked only for this example (dev-dependency) — no crate cycle
//! with mega-render.

#[path = "framework.rs"]
mod framework;

use framework::{Demo, Host, SCENE_TEX, UiCtx};
use glam::{Vec2, Vec3};
use mega_render::{
    cube, plane, sphere, Light, Material, Node, PointLight, Scene, Transform, Visualizer,
    WgpuVisualizer,
};
use mega_ui::{DockNode, DockState, ScrollAxes, TextStyle, Ui};

fn build_demo_scene() -> Scene {
    let mut scene = Scene::new();
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        d.intensity = 2.5;
        d.color = [1.0, 0.98, 0.92];
    }
    scene.ambient = [0.03, 0.03, 0.04];
    scene.ibl_intensity = 1.0;
    scene.lights.push(Light::Point(PointLight {
        position: Vec3::new(-2.0, 3.0, 1.5),
        color: [0.5, 0.7, 1.0],
        intensity: 6.0,
        range: 10.0,
    }));

    let mesh_plane = scene.meshes.insert(plane(14.0, 14.0));
    let mesh_cube = scene.meshes.insert(cube(0.8));
    let mesh_sphere = scene.meshes.insert(sphere(0.45, 32, 20));

    let mat_ground = scene
        .materials
        .insert(Material::new([0.35, 0.38, 0.32, 1.0], 0.0, 0.85));

    let root = scene.nodes.insert(Node {
        name: "root".into(),
        parent: None,
        local: Transform::default(),
        mesh: None,
        material: None,
        skin: None,
        visible: true,
    });

    scene.nodes.insert(Node {
        name: "ground".into(),
        parent: Some(root),
        local: Transform::default(),
        mesh: Some(mesh_plane),
        material: Some(mat_ground),
        skin: None,
        visible: true,
    });

    let di_rough = [0.05, 0.15, 0.3, 0.5, 0.75, 1.0];
    for (i, &r) in di_rough.iter().enumerate() {
        let mat = scene
            .materials
            .insert(Material::new([0.85, 0.15, 0.12, 1.0], 0.0, r));
        let x = -4.0 + i as f32 * 1.6;
        scene.nodes.insert(Node {
            name: format!("di_sphere_{i}"),
            parent: Some(root),
            local: Transform::from_translation(Vec3::new(x, 0.45, 0.0)),
            mesh: Some(mesh_sphere),
            material: Some(mat),
            skin: None,
            visible: true,
        });
    }

    let metal_rough = [0.05, 0.15, 0.3, 0.5, 0.75, 1.0];
    for (i, &r) in metal_rough.iter().enumerate() {
        let mat = scene
            .materials
            .insert(Material::new([0.95, 0.78, 0.35, 1.0], 1.0, r));
        let x = -4.0 + i as f32 * 1.6;
        scene.nodes.insert(Node {
            name: format!("metal_sphere_{i}"),
            parent: Some(root),
            local: Transform::from_translation(Vec3::new(x, 0.45, 2.2)),
            mesh: Some(mesh_sphere),
            material: Some(mat),
            skin: None,
            visible: true,
        });
    }

    let cubes = [
        ([0.2, 0.45, 0.9, 1.0], 0.0, 0.2, Vec3::new(-3.2, 0.4, -2.2)),
        ([0.2, 0.45, 0.9, 1.0], 0.0, 0.8, Vec3::new(-1.6, 0.4, -2.2)),
        ([0.75, 0.75, 0.78, 1.0], 1.0, 0.15, Vec3::new(0.0, 0.4, -2.2)),
        ([0.75, 0.75, 0.78, 1.0], 1.0, 0.6, Vec3::new(1.6, 0.4, -2.2)),
        ([0.15, 0.7, 0.35, 1.0], 0.0, 0.4, Vec3::new(3.2, 0.4, -2.2)),
    ];
    for (i, &(albedo, metallic, roughness, pos)) in cubes.iter().enumerate() {
        let mat = scene
            .materials
            .insert(Material::new(albedo, metallic, roughness));
        scene.nodes.insert(Node {
            name: format!("cube_{i}"),
            parent: Some(root),
            local: Transform::from_translation(pos),
            mesh: Some(mesh_cube),
            material: Some(mat),
            skin: None,
            visible: true,
        });
    }

    scene
}

fn default_dock() -> DockState {
    DockState::new(DockNode::split_h(
        0.72,
        DockNode::leaf(&["Viewport"]),
        DockNode::leaf(&[
            "SSAO", "Bloom", "Tonemap", "Grade", "Vignette", "Grain", "FXAA", "Fog", "Lighting",
        ]),
    ))
}

struct MaterialDemo;

impl Demo for MaterialDemo {
    fn title() -> &'static str {
        "mega-render demo"
    }

    fn build_scene() -> Scene {
        build_demo_scene()
    }

    fn configure(visualizer: &mut WgpuVisualizer) {
        let post = visualizer.post_process();
        post.ssao.enabled = true;
        post.ssao.radius = 0.55;
        post.ssao.intensity = 0.3;
        post.bloom.enabled = true;
        post.bloom.threshold = 1.2;
        post.bloom.intensity = 0.2;
        post.tonemap.enabled = true;
        post.tonemap.aces = true;
        post.tonemap.exposure = 1.4;
        post.color_grade.enabled = false;
        post.vignette.enabled = true;
        post.vignette.intensity = 0.15;
        post.vignette.smoothness = 0.7;
        post.grain.enabled = false;
        post.fxaa.enabled = true;
        post.fog.enabled = false;
    }

    fn update(scene: &mut Scene, _dt: f32) -> bool {
        scene.debug.clear();
        scene.debug.axes(Vec3::ZERO, 1.5);
        false
    }

    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool {
        let fps = (1.0 / ctx.dt.max(1e-4)).min(999.0);
        let status_h = 24.0 * ui.scale();
        let dock_size = Vec2::new(ctx.window_size.x, (ctx.window_size.y - status_h).max(1.0));

        let UiCtx {
            scene,
            post,
            dock,
            viewport_size,
            stats,
            ..
        } = ctx;

        ui.dock_space("main", dock_size, dock, |ui, tab| match tab {
            "Viewport" => {
                ui.label_styled(
                    "Scene",
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
            "SSAO" => effect_panel(ui, "SSAO", |ui| {
                ui.checkbox("Enabled", &mut post.ssao.enabled);
                ui.add_enabled(post.ssao.enabled, |ui| {
                    ui.slider("Radius", &mut post.ssao.radius, 0.1..=3.0);
                    ui.slider("Bias", &mut post.ssao.bias, 0.0..=0.2);
                    ui.slider("Intensity", &mut post.ssao.intensity, 0.0..=2.0);
                });
            }),
            "Bloom" => effect_panel(ui, "Bloom", |ui| {
                ui.checkbox("Enabled", &mut post.bloom.enabled);
                ui.add_enabled(post.bloom.enabled, |ui| {
                    ui.slider("Threshold", &mut post.bloom.threshold, 0.0..=4.0);
                    ui.slider("Intensity", &mut post.bloom.intensity, 0.0..=2.0);
                });
            }),
            "Tonemap" => effect_panel(ui, "Tonemap", |ui| {
                ui.checkbox("Enabled", &mut post.tonemap.enabled);
                ui.add_enabled(post.tonemap.enabled, |ui| {
                    ui.checkbox("ACES", &mut post.tonemap.aces);
                    ui.slider("Exposure", &mut post.tonemap.exposure, 0.1..=4.0);
                });
            }),
            "Grade" => effect_panel(ui, "Color Grade", |ui| {
                ui.checkbox("Enabled", &mut post.color_grade.enabled);
                ui.add_enabled(post.color_grade.enabled, |ui| {
                    ui.slider("Contrast", &mut post.color_grade.contrast, 0.0..=2.0);
                    ui.slider("Saturation", &mut post.color_grade.saturation, 0.0..=2.0);
                    ui.slider("Brightness", &mut post.color_grade.brightness, -0.5..=0.5);
                });
            }),
            "Vignette" => effect_panel(ui, "Vignette", |ui| {
                ui.checkbox("Enabled", &mut post.vignette.enabled);
                ui.add_enabled(post.vignette.enabled, |ui| {
                    ui.slider("Intensity", &mut post.vignette.intensity, 0.0..=1.0);
                    ui.slider("Smoothness", &mut post.vignette.smoothness, 0.05..=1.5);
                });
            }),
            "Grain" => effect_panel(ui, "Film Grain", |ui| {
                ui.checkbox("Enabled", &mut post.grain.enabled);
                ui.add_enabled(post.grain.enabled, |ui| {
                    ui.slider("Intensity", &mut post.grain.intensity, 0.0..=0.2);
                });
            }),
            "FXAA" => effect_panel(ui, "FXAA", |ui| {
                ui.checkbox("Enabled", &mut post.fxaa.enabled);
                ui.label("Fast approximate anti-aliasing.");
            }),
            "Fog" => effect_panel(ui, "Fog", |ui| {
                ui.checkbox("Enabled", &mut post.fog.enabled);
                ui.add_enabled(post.fog.enabled, |ui| {
                    ui.label("Color");
                    let mut fog_col = [
                        post.fog.color[0],
                        post.fog.color[1],
                        post.fog.color[2],
                        1.0,
                    ];
                    if ui.color_edit("fog_color", &mut fog_col).changed() {
                        post.fog.color = [fog_col[0], fog_col[1], fog_col[2]];
                    }
                    ui.slider("Density", &mut post.fog.density, 0.0..=0.2);
                    ui.slider("Height", &mut post.fog.height, -5.0..=5.0);
                    ui.slider("Height falloff", &mut post.fog.height_falloff, 0.0..=2.0);
                });
            }),
            "Lighting" => effect_panel(ui, "Lighting", |ui| {
                ui.slider("IBL intensity", &mut scene.ibl_intensity, 0.0..=3.0);
                ui.label("Ambient");
                let mut amb = [scene.ambient[0], scene.ambient[1], scene.ambient[2], 1.0];
                if ui.color_edit("ambient", &mut amb).changed() {
                    scene.ambient = [amb[0], amb[1], amb[2]];
                }
                if let Some(Light::Directional(d)) = scene.lights.first_mut() {
                    ui.separator();
                    ui.label("Directional");
                    ui.slider("Intensity", &mut d.intensity, 0.0..=8.0);
                    let mut col = [d.color[0], d.color[1], d.color[2], 1.0];
                    if ui.color_edit("sun_color", &mut col).changed() {
                        d.color = [col[0], col[1], col[2]];
                    }
                }
                let mut point = scene.lights.iter_mut().find_map(|l| match l {
                    Light::Point(p) => Some(p),
                    _ => None,
                });
                if let Some(p) = point.as_mut() {
                    ui.separator();
                    ui.label("Point light");
                    ui.slider("Intensity", &mut p.intensity, 0.0..=20.0);
                    ui.slider("Range", &mut p.range, 1.0..=30.0);
                    let mut col = [p.color[0], p.color[1], p.color[2], 1.0];
                    if ui.color_edit("pt_color", &mut col).changed() {
                        p.color = [col[0], col[1], col[2]];
                    }
                }
            }),
            other => ui.label(other),
        });

        ui.status_bar(|ui| {
            ui.label("RMB look · WASD move · Esc quit");
            ui.label("·");
            ui.label(&format!("FPS {:.0}", fps));
            ui.label("·");
            ui.label(&format!("ui {} / {}", stats.batches, stats.quads));
        });

        true
    }
}

fn effect_panel(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui)) {
    let size = ui.available_size();
    ui.scroll_area(title, size, ScrollAxes::Vertical, |ui| {
        ui.label_styled(
            title,
            TextStyle {
                color: [0.85, 0.75, 0.35, 1.0],
                size: 16.0,
            },
        );
        ui.separator();
        add(ui);
    });
}

fn main() {
    Host::<MaterialDemo>::run(default_dock());
}
