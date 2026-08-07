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
    cube, plane, sphere, AoMethod, DebugView, Light, Material, Node, PointLight, Scene, Transform,
    Visualizer, WgpuVisualizer,
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
        enabled: true,
    }));
    scene.lights.push(Light::Point(PointLight {
        position: Vec3::new(2.5, 2.5, -1.0),
        color: [1.0, 0.55, 0.25],
        intensity: 5.0,
        range: 12.0,
        enabled: true,
    }));
    scene.lights.push(Light::Point(PointLight {
        position: Vec3::new(0.0, 2.0, 3.0),
        color: [0.85, 0.9, 1.0],
        intensity: 4.0,
        range: 10.0,
        enabled: true,
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

    let models = [
        (r"C:\Users\black\OneDrive\Desktop\tank.glb", Vec3::new(0.0, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\boa.glb", Vec3::new(0.0, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\bulma_full.glb", Vec3::new(0.5, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\cammy_full.glb", Vec3::new(1.0, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\chunli_full.glb", Vec3::new(1.5, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\zangya_full.glb", Vec3::new(2.0, 0.0, -4.5)),
        (r"F:/csharp/VR_Waifu/asset/model/Bulma/Bulma2.gltf", Vec3::new(2.5, 0.0, -4.5)),
    ];
    for &(path, offset) in &models {
        scene.load_gltf_async(path, Some(root), move |scene, h| {
            if let Some(n) = scene.nodes.get_mut(h) {
                n.local.translation += offset;
            }
        });
    }

    scene
}

fn default_dock() -> DockState {
    DockState::new(DockNode::split_h(
        0.72,
        DockNode::leaf(&["Viewport"]),
        DockNode::leaf(&["Debug", "Effects", "Lights"]),
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
        post.ao.enabled = true;
        post.ao.method = AoMethod::Gtao;
        post.ao.radius = 0.55;
        post.ao.intensity = 0.4;
        post.ao.directions = 6;
        post.ao.steps = 8;
        post.ao.thickness = 1.0;
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
        let fps = ctx.fps;
        let status_h = 24.0 * ui.scale();
        let dock_size = Vec2::new(ctx.window_size.x, (ctx.window_size.y - status_h).max(1.0));

        let UiCtx {
            scene,
            post,
            debug_view,
            dock,
            viewport_size,
            stats,
            ..
        } = ctx;

        ui.dock_space("main", dock_size, dock, |ui, tab| match tab {
            "Viewport" => {
                ui.label_styled(
                    &format!("Scene · {}", (**debug_view).label()),
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
            "Debug" => effect_panel(ui, "Debug View", |ui| {
                ui.label("G-buffer / pass visualization");
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
                ui.label("Final = lit + post stack.");
                ui.label("Others blit thin G-buffer / AO.");
            }),
            "Effects" => effect_panel(ui, "Effects", |ui| {
                ui.collapsing_header("AO", |ui| {
                    ui.checkbox("Enabled", &mut post.ao.enabled);
                    ui.add_enabled(post.ao.enabled, |ui| {
                        let mut method = match post.ao.method {
                            AoMethod::Ssao => 0usize,
                            AoMethod::Gtao => 1,
                        };
                        ui.label("Method — алгоритм AO");
                        if ui
                            .select("ao_method", &mut method, &["SSAO", "GTAO"])
                            .changed()
                        {
                            post.ao.method = if method == 0 {
                                AoMethod::Ssao
                            } else {
                                AoMethod::Gtao
                            };
                        }
                        match post.ao.method {
                            AoMethod::Ssao => {
                                ui.label("SSAO: hemisphere kernel, нормаль из depth.");
                            }
                            AoMethod::Gtao => {
                                ui.label("GTAO: obscurance + нормали из G-buffer.");
                            }
                        }
                        ui.separator();
                        ui.label("Radius — радиус выборки в мире");
                        ui.slider("Radius", &mut post.ao.radius, 0.1..=3.0);
                        ui.label("Intensity — сила AO");
                        ui.slider("Intensity", &mut post.ao.intensity, 0.0..=2.0);
                        match post.ao.method {
                            AoMethod::Ssao => {
                                ui.label("Bias — отступ вдоль нормали");
                                ui.slider("Bias", &mut post.ao.bias, 0.0..=0.2);
                            }
                            AoMethod::Gtao => {
                                let mut dirs = post.ao.directions as f32;
                                let mut steps = post.ao.steps as f32;
                                ui.label("Directions — число лучей");
                                if ui.slider("Directions", &mut dirs, 2.0..=8.0).changed() {
                                    post.ao.directions = dirs.round() as u32;
                                }
                                ui.label("Steps — шагов вдоль луча");
                                if ui.slider("Steps", &mut steps, 2.0..=12.0).changed() {
                                    post.ao.steps = steps.round() as u32;
                                }
                                ui.label("Thickness — контраст");
                                ui.slider("Thickness", &mut post.ao.thickness, 0.2..=3.0);
                            }
                        }
                    });
                });
                ui.collapsing_header("Bloom", |ui| {
                    ui.checkbox("Enabled", &mut post.bloom.enabled);
                    ui.add_enabled(post.bloom.enabled, |ui| {
                        ui.slider("Threshold", &mut post.bloom.threshold, 0.0..=4.0);
                        ui.slider("Intensity", &mut post.bloom.intensity, 0.0..=2.0);
                    });
                });
                ui.collapsing_header("Tonemap", |ui| {
                    ui.checkbox("Enabled", &mut post.tonemap.enabled);
                    ui.add_enabled(post.tonemap.enabled, |ui| {
                        ui.checkbox("ACES", &mut post.tonemap.aces);
                        ui.slider("Exposure", &mut post.tonemap.exposure, 0.1..=4.0);
                    });
                });
                ui.collapsing_header("Color Grade", |ui| {
                    ui.checkbox("Enabled", &mut post.color_grade.enabled);
                    ui.add_enabled(post.color_grade.enabled, |ui| {
                        ui.slider("Contrast", &mut post.color_grade.contrast, 0.0..=2.0);
                        ui.slider("Saturation", &mut post.color_grade.saturation, 0.0..=2.0);
                        ui.slider("Brightness", &mut post.color_grade.brightness, -0.5..=0.5);
                    });
                });
                ui.collapsing_header("Vignette", |ui| {
                    ui.checkbox("Enabled", &mut post.vignette.enabled);
                    ui.add_enabled(post.vignette.enabled, |ui| {
                        ui.slider("Intensity", &mut post.vignette.intensity, 0.0..=1.0);
                        ui.slider("Smoothness", &mut post.vignette.smoothness, 0.05..=1.5);
                    });
                });
                ui.collapsing_header("Film Grain", |ui| {
                    ui.checkbox("Enabled", &mut post.grain.enabled);
                    ui.add_enabled(post.grain.enabled, |ui| {
                        ui.slider("Intensity", &mut post.grain.intensity, 0.0..=0.2);
                    });
                });
                ui.collapsing_header("FXAA", |ui| {
                    ui.checkbox("Enabled", &mut post.fxaa.enabled);
                    ui.label("Fast Approximate AA после post.");
                });
                ui.collapsing_header("Fog", |ui| {
                    ui.checkbox("Enabled", &mut post.fog.enabled);
                    ui.add_enabled(post.fog.enabled, |ui| {
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
                });
            }),
            "Lights" => effect_panel(ui, "Lights", |ui| {
                ui.label("Ambient / IBL");
                ui.slider("IBL intensity", &mut scene.ibl_intensity, 0.0..=3.0);
                let mut amb = [scene.ambient[0], scene.ambient[1], scene.ambient[2], 1.0];
                if ui.color_edit("ambient", &mut amb).changed() {
                    scene.ambient = [amb[0], amb[1], amb[2]];
                }

                if let Some(Light::Directional(d)) = scene.lights.first_mut() {
                    ui.separator();
                    ui.collapsing_header("Directional (sun)", |ui| {
                        ui.checkbox("Enabled", &mut d.enabled);
                        ui.checkbox("Cast shadows", &mut d.cast_shadows);
                        ui.add_enabled(d.enabled, |ui| {
                            ui.slider("Intensity", &mut d.intensity, 0.0..=8.0);
                            let mut col = [d.color[0], d.color[1], d.color[2], 1.0];
                            if ui.color_edit("sun_color", &mut col).changed() {
                                d.color = [col[0], col[1], col[2]];
                            }
                            ui.label("Direction (world)");
                            ui.slider("Dir X", &mut d.direction.x, -1.0..=1.0);
                            ui.slider("Dir Y", &mut d.direction.y, -1.0..=1.0);
                            ui.slider("Dir Z", &mut d.direction.z, -1.0..=1.0);
                        });
                    });
                }

                ui.separator();
                ui.label("Point lights");
                let point_indices: Vec<usize> = scene
                    .lights
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| matches!(l, Light::Point(_)).then_some(i))
                    .take(3)
                    .collect();
                for (n, idx) in point_indices.into_iter().enumerate() {
                    let Light::Point(p) = &mut scene.lights[idx] else {
                        continue;
                    };
                    let header = format!("Point {}", n + 1);
                    ui.collapsing_header(&header, |ui| {
                        ui.checkbox("Enabled", &mut p.enabled);
                        ui.add_enabled(p.enabled, |ui| {
                            ui.slider("Intensity", &mut p.intensity, 0.0..=20.0);
                            ui.slider("Range", &mut p.range, 1.0..=30.0);
                            let mut col = [p.color[0], p.color[1], p.color[2], 1.0];
                            if ui.color_edit("color", &mut col).changed() {
                                p.color = [col[0], col[1], col[2]];
                            }
                            ui.label("Position");
                            ui.slider("X", &mut p.position.x, -10.0..=10.0);
                            ui.slider("Y", &mut p.position.y, 0.0..=10.0);
                            ui.slider("Z", &mut p.position.z, -10.0..=10.0);
                        });
                    });
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
