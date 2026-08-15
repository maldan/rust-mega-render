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
    cube, plane, sphere, AoMethod, DebugView, Handle, Light, Material, Node, PointLight, Scene,
    SsgiQuality,
    ShadowFilter, Transform, Visualizer, WgpuVisualizer,
};
use mega_ui::{DockNode, DockState, ScrollAxes, TextStyle, Ui};

fn build_demo_scene() -> Scene {
    let mut scene = Scene::new();
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        d.intensity = 3.3;
        d.color = [1.0, 1.0, 1.0];
        d.direction = Vec3::new(0.28, -0.30, 0.93);
    }
    scene.ambient = [0.03, 0.03, 0.04];
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
        enabled: false,
    }));
    scene.lights.push(Light::Point(PointLight {
        position: Vec3::new(0.0, 2.0, 3.0),
        color: [0.85, 0.9, 1.0],
        intensity: 4.0,
        range: 10.0,
        enabled: false,
    }));

    let mesh_plane = scene.meshes.insert(plane(14.0, 14.0));
    let mesh_cube = scene.meshes.insert(cube(0.8));
    let mesh_sphere = scene.meshes.insert(sphere(0.45, 32, 20));

    let mat_ground = scene
        .materials
        .insert(Material::new([0.72, 0.72, 0.75, 1.0], 1.0, 0.35));

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
        /*(r"C:\Users\black\OneDrive\Desktop\tank.glb", Vec3::new(0.0, 0.0, -4.5)),
        (r"C:\Users\black\OneDrive\Desktop\tank_2.glb", Vec3::new(1.5, 0.0, -4.5)),
        (r"C:\Users\black\OneDrive\Desktop\ammo.glb", Vec3::new(4.0, 0.0, -4.5)),
        (r"C:\Users\black\OneDrive\Desktop\cyborg.glb", Vec3::new(-1.0, 0.0, -4.5)),*/
        
        /*(r"F:\3d\tripo_ai\boa.glb", Vec3::new(0.0, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\bulma_full.glb", Vec3::new(0.5, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\cammy_full.glb", Vec3::new(1.0, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\chunli_full.glb", Vec3::new(1.5, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\zangya_full.glb", Vec3::new(2.0, 0.0, -4.5)),
        (r"F:\3d\tripo_ai\rangiku_full.glb", Vec3::new(2.5, 0.0, -4.5)),*/
        (r"F:\3d\tripo_ai\2b.glb", Vec3::new(3.0, 0.0, -4.5)),
        
        /*(r"F:/csharp/VR_Waifu/asset/model/Bulma/Bulma2.gltf", Vec3::new(2.5, 0.0, -4.5)),
        //(r"F:/csharp/VR_Waifu/asset/model/Miruko/Miruko.gltf", Vec3::new(3.0, 0.0, -4.5)),
        (r"F:/csharp/VR_Waifu/asset/model/ChunLi/ChunLi.gltf", Vec3::new(3.5, 0.0, -4.5)),
        (r"F:/csharp/VR_Waifu/asset/model/Tatsumaki/Tatsumaki_2.gltf", Vec3::new(4.0, 0.0, -4.5)),
        (r"F:/csharp/VR_Waifu/asset/model/Zangya/Zangya2.gltf", Vec3::new(4.5, 0.0, -4.5)),
        (r"F:/csharp/VR_Waifu/asset/model/Yoruichi/Yoruichi.gltf", Vec3::new(5.0, 0.0, -4.5)),*/
        
        
    ];
    for &(path, offset) in &models {
        scene.load_gltf_async(path, Some(root), move |scene, h| {
            if let Some(n) = scene.nodes.get_mut(h) {
                n.local.translation += offset;
            }
            enable_sss_on_subtree(scene, h);
        });
    }

    scene.camera.focus_distance = 0.7;
    scene.camera.focus_target = 0.7;
    scene.camera.focus_smooth = 2.1;
    scene.camera.f_stop = 0.9;

    scene
}

/// Enable pre-integrated SSS on every material under a loaded model (test hack).
fn enable_sss_on_subtree(scene: &mut Scene, root: Handle<Node>) {
    let mut stack = vec![root];
    let mut mats = Vec::new();
    while let Some(h) = stack.pop() {
        if let Some(n) = scene.nodes.get(h) {
            if let Some(m) = n.material {
                mats.push(m);
            }
        }
        for (child, node) in scene.nodes.iter() {
            if node.parent.map(|p| p.key()) == Some(h.key()) {
                stack.push(child);
            }
        }
    }
    mats.sort_by_key(|m| m.key());
    mats.dedup_by_key(|m| m.key());
    for m in mats {
        if let Some(mat) = scene.materials.get_mut(m) {
            // Skip obvious metals so chrome bits don't go waxy.
            if mat.metallic > 0.85 {
                continue;
            }
            mat.sss_strength = 1.0;
            mat.sss_color = [1.0, 0.32, 0.18];
            mat.sss_curvature = 0.7;
        }
    }
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
        visualizer.set_env_map_async(r"F:\3d\garbage\hdri\ferndale_studio_12_4k.exr");

        let shadow = visualizer.shadow_settings();
        shadow.filter = ShadowFilter::Pcss;
        shadow.map_size = 4096;
        shadow.pcss_light_size = 0.35;
        shadow.pcss_blocker_samples = 16;
        shadow.pcss_filter_samples = 48;

        let post = visualizer.post_process();
        post.env.enabled = true;
        post.env.intensity = 1.0;
        post.env.rotation_y = 0.0;
        post.ao.enabled = true;
        post.contact_shadow.enabled = true;
        post.ssgi.enabled = true;
        post.ssr.enabled = true;
        post.ssr.max_distance = 5.0;
        post.ssr.thickness = 0.24;
        post.ssr.intensity = 0.63;
        post.ssr.max_steps = 37;
        post.ssr.bias = 0.0;
        post.ssr.roughness_cutoff = 0.48;
        post.ssr.temporal = true;
        post.ssr.history = 0.98;
        post.ssr.depth_reject = 0.1;
        post.bloom.enabled = true;
        post.bloom.threshold = 1.2;
        post.bloom.intensity = 0.2;
        post.dof.enabled = true;
        post.dof.max_coc_px = 40.1;
        post.dof.scale = 15.6;
        post.dof.focus_range = 0.21;
        post.dof.samples = 14;
        post.dof.bokeh_blades = 5;
        post.dof.half_res = false;
        post.dof.auto_focus = false;
        post.dof.temporal = true;
        post.dof.history = 0.58;
        post.dof.depth_reject = 0.02;
        post.motion_blur.enabled = false;
        post.motion_blur.intensity = 1.0;
        post.motion_blur.max_blur_px = 64.0;
        post.motion_blur.samples = 16;
        post.motion_blur.dilate_radius = 2;
        post.motion_blur.depth_sigma = 0.02;
        post.motion_blur.shutter = 1.0 / 24.0;
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
        let frame_ms = ctx.frame_ms;
        let status_h = 24.0 * ui.scale();
        let dock_size = Vec2::new(ctx.window_size.x, (ctx.window_size.y - status_h).max(1.0));

        let UiCtx {
            scene,
            post,
            shadow,
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
                ui.collapsing_header("Env Map", |ui| {
                    ui.checkbox("Enabled", &mut post.env.enabled);
                    ui.label("Equirect HDR/EXR reflections + skybox.");
                    ui.add_enabled(post.env.enabled, |ui| {
                        ui.label("Intensity — сила отражений / sky");
                        ui.slider("Intensity", &mut post.env.intensity, 0.0..=3.0);
                        ui.label("Rotation Y — поворот карты (°)");
                        ui.slider("Rotation Y", &mut post.env.rotation_y, 0.0..=360.0);
                    });
                });
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
                ui.collapsing_header("Contact Shadows", |ui| {
                    ui.checkbox("Enabled", &mut post.contact_shadow.enabled);
                    ui.label("Короткие SS-тени вдоль directional light.");
                    ui.add_enabled(post.contact_shadow.enabled, |ui| {
                        ui.label("Length — длина луча (мир)");
                        ui.slider("Length", &mut post.contact_shadow.length, 0.05..=1.5);
                        ui.label("Thickness — допуск по глубине");
                        ui.slider("Thickness", &mut post.contact_shadow.thickness, 0.01..=2.0);
                        ui.label("Intensity — сила затемнения");
                        ui.slider("Intensity", &mut post.contact_shadow.intensity, 0.0..=2.0);
                        let mut samples = post.contact_shadow.samples as f32;
                        ui.label("Samples — шагов марша");
                        if ui.slider("Samples", &mut samples, 4.0..=32.0).changed() {
                            post.contact_shadow.samples = samples.round() as u32;
                        }
                        ui.label("Bias — отступ от поверхности");
                        ui.slider("Bias", &mut post.contact_shadow.bias, 0.0..=0.05);
                    });
                });
                ui.collapsing_header("SSGI", |ui| {
                    ui.checkbox("Enabled", &mut post.ssgi.enabled);
                    ui.label("Screen-space GI: Hi-Z + à-trous + velocity temporal + 2nd bounce.");
                    ui.add_enabled(post.ssgi.enabled, |ui| {
                        let mut q = match (post.ssgi.samples, post.ssgi.max_steps) {
                            (s, t) if s <= 4 && t <= 4 => 0usize,
                            (s, t) if s >= 12 || t >= 12 => 2,
                            _ => 1,
                        };
                        ui.label("Quality — rays × march steps");
                        if ui
                            .select("ssgi_quality", &mut q, &["Low (4×4)", "Medium (8×8)", "High (12×12)"])
                            .changed()
                        {
                            SsgiQuality::ALL[q].apply(&mut post.ssgi);
                        }
                        ui.separator();
                        ui.label("Radius — длина луча (мир)");
                        ui.slider("Radius", &mut post.ssgi.radius, 0.2..=4.0);
                        ui.label("Thickness — допуск по глубине");
                        ui.slider("Thickness", &mut post.ssgi.thickness, 0.02..=1.0);
                        ui.label("Intensity — сила в composite (× albedo × kd)");
                        ui.slider("Intensity", &mut post.ssgi.intensity, 0.0..=5.0);
                        ui.label("Energy — масштаб irradiance в spatial pass");
                        ui.slider("Energy", &mut post.ssgi.energy, 0.25..=3.0);
                        ui.label("2nd bounce — дешёвый screen-space bleed");
                        ui.slider("2nd bounce", &mut post.ssgi.second_bounce, 0.0..=1.5);
                        let mut samples = post.ssgi.samples as f32;
                        let mut steps = post.ssgi.max_steps as f32;
                        ui.label("Samples — лучей на пиксель");
                        if ui.slider("Samples", &mut samples, 4.0..=32.0).changed() {
                            post.ssgi.samples = samples.round() as u32;
                        }
                        ui.label("Steps — шагов марша");
                        if ui.slider("Steps", &mut steps, 4.0..=32.0).changed() {
                            post.ssgi.max_steps = steps.round() as u32;
                        }
                        ui.label("Bias — отступ от поверхности");
                        ui.slider("Bias", &mut post.ssgi.bias, 0.0..=0.1);
                        ui.label("Ambient dim — ослабление constant ambient при SSGI");
                        ui.slider("Ambient dim", &mut post.ssgi.ambient_dim, 0.0..=1.0);
                        ui.separator();
                        ui.checkbox("Temporal", &mut post.ssgi.temporal);
                        ui.label("Velocity reprojection + AABB clamp + soft depth/normal reject.");
                        ui.add_enabled(post.ssgi.temporal, |ui| {
                            ui.label("History — вес прошлого кадра");
                            ui.slider("History", &mut post.ssgi.history, 0.5..=0.98);
                            ui.label("Depth reject — порог disocclusion");
                            ui.slider("Depth reject", &mut post.ssgi.depth_reject, 0.005..=0.1);
                        });
                    });
                });
                ui.collapsing_header("SSR", |ui| {
                    ui.checkbox("Enabled", &mut post.ssr.enabled);
                    ui.label("Screen reflections; misses fall back to env map.");
                    ui.add_enabled(post.ssr.enabled, |ui| {
                        ui.label("Max distance — длина луча");
                        ui.slider("Max distance", &mut post.ssr.max_distance, 0.5..=20.0);
                        ui.label("Thickness — допуск по глубине");
                        ui.slider("Thickness", &mut post.ssr.thickness, 0.02..=1.0);
                        ui.label("Intensity — сила specular");
                        ui.slider("Intensity", &mut post.ssr.intensity, 0.0..=2.0);
                        ui.label("Steps — шагов марша");
                        let mut steps = post.ssr.max_steps as f32;
                        if ui.slider("Steps", &mut steps, 8.0..=64.0).changed() {
                            post.ssr.max_steps = steps.round() as u32;
                        }
                        ui.label("Bias — отступ от поверхности");
                        ui.slider("Bias", &mut post.ssr.bias, 0.0..=0.2);
                        ui.label("Roughness cutoff — выше только env");
                        ui.slider(
                            "Roughness cutoff",
                            &mut post.ssr.roughness_cutoff,
                            0.1..=1.0,
                        );
                        ui.separator();
                        ui.checkbox("Temporal", &mut post.ssr.temporal);
                        ui.label("Camera reprojection + depth rejection.");
                        ui.add_enabled(post.ssr.temporal, |ui| {
                            ui.label("History — вес прошлого кадра");
                            ui.slider("History", &mut post.ssr.history, 0.5..=0.98);
                            ui.label("Depth reject — порог disocclusion");
                            ui.slider("Depth reject", &mut post.ssr.depth_reject, 0.005..=0.1);
                        });
                    });
                });
                ui.collapsing_header("Bloom", |ui| {
                    ui.checkbox("Enabled", &mut post.bloom.enabled);
                    ui.add_enabled(post.bloom.enabled, |ui| {
                        ui.label("Threshold — порог яркости");
                        ui.slider("Threshold", &mut post.bloom.threshold, 0.0..=4.0);
                        ui.label("Intensity — сила свечения");
                        ui.slider("Intensity", &mut post.bloom.intensity, 0.0..=2.0);
                    });
                });
                ui.collapsing_header("DOF", |ui| {
                    ui.checkbox("Enabled", &mut post.dof.enabled);
                    ui.label("Temporal dual-field CoC + bokeh (optics на камере).");
                    ui.add_enabled(post.dof.enabled, |ui| {
                        ui.checkbox("Auto focus (ground)", &mut post.dof.auto_focus);
                        ui.label("Focus distance — плоскость фокуса (мир)");
                        if ui
                            .slider(
                                "Focus distance",
                                &mut scene.camera.focus_target,
                                0.2..=40.0,
                            )
                            .changed()
                        {
                            // Keep display distance in sync when not smoothing hard.
                            if scene.camera.focus_smooth <= 1e-3 {
                                scene.camera.focus_distance = scene.camera.focus_target;
                            }
                        }
                        ui.label("Focus smooth — скорость focus pull");
                        ui.slider("Focus smooth", &mut scene.camera.focus_smooth, 0.0..=20.0);
                        if ui.button("Focus → scene center").clicked() {
                            let center = Vec3::new(0.0, 0.5, 0.0);
                            scene.camera.autofocus_point(center);
                        }
                        ui.label("F-stop — меньше = сильнее blur");
                        ui.slider("F-stop", &mut scene.camera.f_stop, 0.8..=22.0);
                        ui.label("Focus range — зона резкости (мир)");
                        ui.slider("Focus range", &mut post.dof.focus_range, 0.0..=2.0);
                        ui.label("Max CoC — радиус в пикселях");
                        ui.slider("Max CoC", &mut post.dof.max_coc_px, 4.0..=48.0);
                        ui.label("Scale — сила относительно f-stop");
                        ui.slider("Scale", &mut post.dof.scale, 1.0..=40.0);
                        let mut samples = post.dof.samples as f32;
                        ui.label("Samples — тапов gather");
                        if ui.slider("Samples", &mut samples, 4.0..=24.0).changed() {
                            post.dof.samples = samples.round() as u32;
                        }
                        let mut blades = post.dof.bokeh_blades as f32;
                        ui.label("Bokeh blades — 0=круг, 6=hex");
                        if ui.slider("Bokeh blades", &mut blades, 0.0..=8.0).changed() {
                            let b = blades.round() as u32;
                            post.dof.bokeh_blades = if b > 0 && b < 5 { 5 } else { b };
                        }
                        ui.checkbox("Half-res gather", &mut post.dof.half_res);
                        ui.separator();
                        ui.checkbox("Temporal", &mut post.dof.temporal);
                        ui.label("Camera reprojection + depth rejection.");
                        ui.add_enabled(post.dof.temporal, |ui| {
                            ui.label("History — вес прошлого кадра");
                            ui.slider("History", &mut post.dof.history, 0.5..=0.98);
                            ui.label("Depth reject — порог disocclusion");
                            ui.slider("Depth reject", &mut post.dof.depth_reject, 0.005..=0.1);
                        });
                    });
                });
                ui.collapsing_header("Motion Blur", |ui| {
                    ui.checkbox("Enabled", &mut post.motion_blur.enabled);
                    ui.label("Shutter масштабирует velocity под FPS. Debug → Velocity.");
                    ui.add_enabled(post.motion_blur.enabled, |ui| {
                        ui.label("Intensity — доп. сила");
                        ui.slider("Intensity", &mut post.motion_blur.intensity, 0.0..=3.0);
                        let mut shutter_ms = post.motion_blur.shutter * 1000.0;
                        ui.label("Shutter — экспозиция (ms), кино ~42ms");
                        if ui.slider("Shutter ms", &mut shutter_ms, 4.0..=80.0).changed() {
                            post.motion_blur.shutter = shutter_ms / 1000.0;
                        }
                        ui.label("Max blur — clamp в пикселях");
                        ui.slider("Max blur", &mut post.motion_blur.max_blur_px, 8.0..=128.0);
                        let mut samples = post.motion_blur.samples as f32;
                        ui.label("Samples — тапов gather");
                        if ui.slider("Samples", &mut samples, 4.0..=24.0).changed() {
                            post.motion_blur.samples = samples.round() as u32;
                        }
                        let mut dilate = post.motion_blur.dilate_radius as f32;
                        ui.label("Dilate — силуэт (px)");
                        if ui.slider("Dilate", &mut dilate, 1.0..=3.0).changed() {
                            post.motion_blur.dilate_radius = dilate.round() as u32;
                        }
                        ui.label("Depth sigma — reject ближнего плана");
                        ui.slider("Depth sigma", &mut post.motion_blur.depth_sigma, 0.005..=0.1);
                    });
                });
                ui.collapsing_header("Tonemap", |ui| {
                    ui.checkbox("Enabled", &mut post.tonemap.enabled);
                    ui.add_enabled(post.tonemap.enabled, |ui| {
                        ui.checkbox("ACES", &mut post.tonemap.aces);
                        ui.label("Exposure — экспозиция");
                        ui.slider("Exposure", &mut post.tonemap.exposure, 0.1..=4.0);
                    });
                });
                ui.collapsing_header("Color Grade", |ui| {
                    ui.checkbox("Enabled", &mut post.color_grade.enabled);
                    ui.add_enabled(post.color_grade.enabled, |ui| {
                        ui.label("Contrast — контраст");
                        ui.slider("Contrast", &mut post.color_grade.contrast, 0.0..=2.0);
                        ui.label("Saturation — насыщенность");
                        ui.slider("Saturation", &mut post.color_grade.saturation, 0.0..=2.0);
                        ui.label("Brightness — яркость");
                        ui.slider("Brightness", &mut post.color_grade.brightness, -0.5..=0.5);
                    });
                });
                ui.collapsing_header("Vignette", |ui| {
                    ui.checkbox("Enabled", &mut post.vignette.enabled);
                    ui.add_enabled(post.vignette.enabled, |ui| {
                        ui.label("Intensity — затемнение краёв");
                        ui.slider("Intensity", &mut post.vignette.intensity, 0.0..=1.0);
                        ui.label("Smoothness — мягкость перехода");
                        ui.slider("Smoothness", &mut post.vignette.smoothness, 0.05..=1.5);
                    });
                });
                ui.collapsing_header("Film Grain", |ui| {
                    ui.checkbox("Enabled", &mut post.grain.enabled);
                    ui.add_enabled(post.grain.enabled, |ui| {
                        ui.label("Intensity — сила зерна");
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
                        ui.label("Color — цвет тумана");
                        let mut fog_col = [
                            post.fog.color[0],
                            post.fog.color[1],
                            post.fog.color[2],
                            1.0,
                        ];
                        if ui.color_edit("fog_color", &mut fog_col).changed() {
                            post.fog.color = [fog_col[0], fog_col[1], fog_col[2]];
                        }
                        ui.label("Density — плотность");
                        ui.slider("Density", &mut post.fog.density, 0.0..=0.2);
                        ui.label("Height — высота пола тумана");
                        ui.slider("Height", &mut post.fog.height, -5.0..=5.0);
                        ui.label("Height falloff — затухание вверх");
                        ui.slider("Height falloff", &mut post.fog.height_falloff, 0.0..=2.0);
                    });
                });
            }),
            "Lights" => effect_panel(ui, "Lights", |ui| {
                ui.label("Ambient — плоский ambient");
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
                            ui.label("Intensity — яркость");
                            ui.slider("Intensity", &mut d.intensity, 0.0..=8.0);
                            ui.label("Color");
                            let mut col = [d.color[0], d.color[1], d.color[2], 1.0];
                            if ui.color_edit("sun_color", &mut col).changed() {
                                d.color = [col[0], col[1], col[2]];
                            }
                            ui.label("Direction (world)");
                            ui.slider("Dir X", &mut d.direction.x, -1.0..=1.0);
                            ui.slider("Dir Y", &mut d.direction.y, -1.0..=1.0);
                            ui.slider("Dir Z", &mut d.direction.z, -1.0..=1.0);
                            if d.cast_shadows {
                                ui.separator();
                                ui.label("Shadows (visualizer)");
                                ui.label("Bias — acne ↔ peter-panning");
                                ui.slider("Shadow bias", &mut shadow.bias, 0.0..=0.01);
                                ui.label("Shadow map size");
                                let mut map_idx = match shadow.map_size {
                                    1024 => 0usize,
                                    4096 => 2,
                                    _ => 1,
                                };
                                if ui
                                    .select("shadow_map_size", &mut map_idx, &["1024", "2048", "4096"])
                                    .changed()
                                {
                                    shadow.map_size = match map_idx {
                                        0 => 1024,
                                        2 => 4096,
                                        _ => 2048,
                                    };
                                }
                                ui.label("Shadow filter");
                                let mut filter = match shadow.filter {
                                    ShadowFilter::Pcf => 0usize,
                                    ShadowFilter::Pcss => 1,
                                };
                                if ui
                                    .select("shadow_filter", &mut filter, &["PCF", "PCSS"])
                                    .changed()
                                {
                                    shadow.filter = if filter == 0 {
                                        ShadowFilter::Pcf
                                    } else {
                                        ShadowFilter::Pcss
                                    };
                                }
                                ui.add_enabled(shadow.filter == ShadowFilter::Pcss, |ui| {
                                    ui.label("Light size — мягкость (0 = острая, 1 = макс)");
                                    ui.slider(
                                        "Light size",
                                        &mut shadow.pcss_light_size,
                                        0.0..=1.0,
                                    );
                                    let mut blockers = shadow.pcss_blocker_samples as f32;
                                    ui.label("Blocker samples — поиск окклюдеров");
                                    if ui.slider("Blocker samples", &mut blockers, 4.0..=16.0).changed()
                                    {
                                        shadow.pcss_blocker_samples = blockers.round() as u32;
                                    }
                                    let mut filters = shadow.pcss_filter_samples as f32;
                                    ui.label("Filter samples — сэмплы penumbra");
                                    if ui.slider("Filter samples", &mut filters, 8.0..=48.0).changed()
                                    {
                                        shadow.pcss_filter_samples = filters.round() as u32;
                                    }
                                });
                            }
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
                            ui.label("Intensity — яркость");
                            ui.slider("Intensity", &mut p.intensity, 0.0..=20.0);
                            ui.label("Range — дальность");
                            ui.slider("Range", &mut p.range, 1.0..=30.0);
                            ui.label("Color");
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
            ui.label(&format!("{:.1} ms", frame_ms));
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
