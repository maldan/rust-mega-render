//! Debug draw showcase + interactive transform gizmos.
//!
//! Showcase: lines / points / boxes / spheres / bones (library helpers).
//! Three cubes: translate · rotate · scale gizmos (no mode switching).
//! View gizmo (top-right): click an axis tip to snap the fly cam.
//!
//! `Space` local/world · LMB drag · RMB look · WASD move · Esc quit
//!
//! ```text
//! cargo run --example debug_draw
//! ```

use std::f32::consts::TAU;
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec2, Vec3};
use mega_render::view_gizmo::{self, ViewAxis};
use mega_render::{
    cube, gizmo_ring_basis, gizmo_screen_size, plane, Camera, GizmoAxis, GizmoMode, GizmoOpts,
    GizmoRotateArc, Handle, InputFrame, Light, LineOpts, Material, Node, Scene, Transform,
    Visualizer, WgpuVisualizer,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

struct ViewTween {
    pivot: Vec3,
    from_dir: Vec3,
    to_dir: Vec3,
    dist: f32,
    t: f32,
}

struct FlyCam {
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    looking: bool,
    keys: [bool; 6],
    sprint: bool,
    look_delta: Vec2,
    tween: Option<ViewTween>,
}

impl FlyCam {
    fn new(eye: Vec3, target: Vec3) -> Self {
        let d = (target - eye).normalize_or_zero();
        Self {
            eye,
            yaw: d.x.atan2(d.z),
            pitch: d.y.clamp(-1.0, 1.0).asin(),
            looking: false,
            keys: [false; 6],
            sprint: false,
            look_delta: Vec2::ZERO,
            tween: None,
        }
    }

    fn set_key(&mut self, key: KeyCode, down: bool) {
        let i = match key {
            KeyCode::KeyW => Some(0),
            KeyCode::KeyS => Some(1),
            KeyCode::KeyA => Some(2),
            KeyCode::KeyD => Some(3),
            KeyCode::KeyE => Some(4),
            KeyCode::KeyQ => Some(5),
            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                self.sprint = down;
                None
            }
            _ => None,
        };
        if let Some(i) = i {
            self.keys[i] = down;
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
    }

    fn look_at(&mut self, eye: Vec3, target: Vec3) {
        self.eye = eye;
        let d = (target - eye).normalize_or_zero();
        self.yaw = d.x.atan2(d.z);
        self.pitch = d.y.clamp(-1.0, 1.0).asin();
    }

    /// Orbit toward `axis` around `pivot` (keeps distance).
    fn snap_view(&mut self, axis: ViewAxis, pivot: Vec3) {
        let from = (self.eye - pivot).normalize_or_zero();
        let to = axis.dir();
        if from.length_squared() < 1e-6 {
            self.look_at(pivot + to, pivot);
            self.tween = None;
            return;
        }
        self.tween = Some(ViewTween {
            pivot,
            from_dir: from,
            to_dir: to,
            dist: (self.eye - pivot).length().max(1.0),
            t: 0.0,
        });
    }

    fn apply(&mut self, dt: f32, scene: &mut Scene) {
        const SENS: f32 = 0.0025;
        const TWEEN_SPEED: f32 = 3.2; // ~0.3s with smoothstep

        // Manual look / move cancels the orbit tween.
        let steering =
            self.looking || self.look_delta.length_squared() > 0.0 || self.keys.iter().any(|&k| k);
        if steering {
            self.tween = None;
        }

        if let Some(tw) = self.tween.as_mut() {
            tw.t = (tw.t + dt * TWEEN_SPEED).min(1.0);
            let e = tw.t * tw.t * (3.0 - 2.0 * tw.t);
            let rot = Quat::IDENTITY.slerp(Quat::from_rotation_arc(tw.from_dir, tw.to_dir), e);
            let eye = tw.pivot + rot * tw.from_dir * tw.dist;
            let pivot = tw.pivot;
            let done = tw.t >= 1.0;
            self.look_at(eye, pivot);
            if done {
                self.tween = None;
            }
            self.look_delta = Vec2::ZERO;
            scene.camera.eye = self.eye;
            scene.camera.target = pivot;
            return;
        }

        self.yaw += self.look_delta.x * SENS;
        self.pitch = (self.pitch - self.look_delta.y * SENS).clamp(-1.55, 1.55);
        self.look_delta = Vec2::ZERO;

        let forward = self.forward();
        let right = Vec3::Y.cross(forward).normalize_or_zero();
        let mut vel = Vec3::ZERO;
        if self.keys[0] {
            vel += forward;
        }
        if self.keys[1] {
            vel -= forward;
        }
        if self.keys[2] {
            vel -= right;
        }
        if self.keys[3] {
            vel += right;
        }
        if self.keys[4] {
            vel += Vec3::Y;
        }
        if self.keys[5] {
            vel -= Vec3::Y;
        }
        let speed = if self.sprint { 12.0 } else { 5.0 };
        if vel.length_squared() > 0.0 {
            self.eye += vel.normalize() * speed * dt;
        }
        scene.camera.eye = self.eye;
        scene.camera.target = self.eye + forward;
    }
}

#[derive(Clone, Copy)]
struct GizmoTarget {
    node: Handle<Node>,
    mode: GizmoMode,
}

#[derive(Clone, Copy)]
struct Drag {
    target: usize,
    axis: GizmoAxis,
    start_t: Transform,
    axis_dir: Vec3,
    plane_n: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
    grab: Vec3,
    start_angle: f32,
    current_angle: f32,
    start_scale_along: f32,
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    visualizer: WgpuVisualizer,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    scene: Scene,
    fly: FlyCam,
    last: Instant,
    time: f32,
    fps: f32,
    fps_accum: f32,
    fps_frames: u32,
    cursor: Vec2,
    mouse_down: bool,
    mouse_pressed: bool,
    mouse_released: bool,
    targets: [GizmoTarget; 3],
    local_space: bool,
    hover: Option<(usize, GizmoAxis)>,
    drag: Option<Drag>,
}

impl App {
    fn new() -> Self {
        let mut scene = Scene::new();
        if let Some(Light::Directional(d)) = scene.lights.first_mut() {
            d.intensity = 2.6;
            d.direction = Vec3::new(0.35, -0.55, 0.75);
        }
        scene.ambient = [0.05, 0.05, 0.06];
        scene.camera.eye = Vec3::new(4.5, 3.2, 6.0);
        scene.camera.target = Vec3::new(0.0, 0.8, 0.0);

        let mesh_plane = scene.meshes.insert(plane(14.0, 14.0));
        let mesh_cube = scene.meshes.insert(cube(1.0));
        let mat_ground = scene
            .materials
            .insert(Material::new([0.4, 0.42, 0.46, 1.0], 0.0, 0.9));
        let mats = [
            scene
                .materials
                .insert(Material::new([0.85, 0.35, 0.28, 1.0], 0.1, 0.4)),
            scene
                .materials
                .insert(Material::new([0.35, 0.75, 0.4, 1.0], 0.1, 0.4)),
            scene
                .materials
                .insert(Material::new([0.35, 0.5, 0.9, 1.0], 0.1, 0.4)),
        ];

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

        let specs = [
            ("move", Vec3::new(-2.6, 0.5, 0.0), GizmoMode::Translate, mats[0]),
            ("rotate", Vec3::new(0.0, 0.5, 0.0), GizmoMode::Rotate, mats[1]),
            ("scale", Vec3::new(2.6, 0.5, 0.0), GizmoMode::Scale, mats[2]),
        ];
        let targets = specs.map(|(name, pos, mode, mat)| {
            let node = scene.nodes.insert(Node {
                name: name.into(),
                parent: Some(root),
                local: Transform {
                    translation: pos,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
                mesh: Some(mesh_cube),
                material: Some(mat),
                skin: None,
                visible: true,
            });
            GizmoTarget { node, mode }
        });

        let fly = FlyCam::new(scene.camera.eye, scene.camera.target);
        Self {
            window: None,
            gpu: None,
            scene,
            fly,
            last: Instant::now(),
            time: 0.0,
            fps: 0.0,
            fps_accum: 0.0,
            fps_frames: 0,
            cursor: Vec2::ZERO,
            mouse_down: false,
            mouse_pressed: false,
            mouse_released: false,
            targets,
            local_space: true,
            hover: None,
            drag: None,
        }
    }

    fn init_gpu(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("mega-render · debug_draw")
                        .with_inner_size(LogicalSize::new(1280.0, 720.0)),
                )
                .expect("window"),
        );
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");

        let mut limits = wgpu::Limits::default();
        let adapter_bytes = adapter.limits().max_color_attachment_bytes_per_sample;
        assert!(
            adapter_bytes >= 36,
            "GPU max_color_attachment_bytes_per_sample={adapter_bytes}, need ≥36"
        );
        limits.max_color_attachment_bytes_per_sample = adapter_bytes.min(64);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("debug_draw"),
            required_features: adapter.features() & mega_render::WGPU_FEATURES,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba8UnormSrgb)
            .expect("Rgba8UnormSrgb");
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let mut visualizer = WgpuVisualizer::new(&device, &queue);
        visualizer.ensure_target(config.width, config.height);
        visualizer.post_process().tonemap.exposure = 1.05;

        self.gpu = Some(Gpu {
            device,
            queue,
            surface,
            config,
            visualizer,
        });
        self.window = Some(window);
    }

    fn tf(&self, i: usize) -> Transform {
        self.scene.nodes.get(self.targets[i].node).unwrap().local
    }

    fn gizmo_rotation(&self, i: usize) -> Quat {
        if self.local_space {
            self.tf(i).rotation
        } else {
            Quat::IDENTITY
        }
    }

    fn axes(&self, i: usize) -> (Vec3, Vec3, Vec3) {
        let r = self.gizmo_rotation(i);
        (r * Vec3::X, r * Vec3::Y, r * Vec3::Z)
    }

    fn size_for(&self, i: usize, viewport_h: f32) -> f32 {
        let pos = self.tf(i).translation;
        let dist = (self.scene.camera.eye - pos).length();
        gizmo_screen_size(dist, self.scene.camera.fov_y, viewport_h, 100.0)
    }

    fn pick(&self, ray_o: Vec3, ray_d: Vec3, viewport_h: f32) -> Option<(usize, GizmoAxis)> {
        let mut best: Option<(f32, usize, GizmoAxis)> = None;
        let mut consider = |score: f32, ti: usize, axis: GizmoAxis| {
            if best.map(|(s, _, _)| score < s).unwrap_or(true) {
                best = Some((score, ti, axis));
            }
        };

        for (ti, target) in self.targets.iter().enumerate() {
            let tf = self.tf(ti);
            let pos = tf.translation;
            let (x, y, z) = self.axes(ti);
            let size = self.size_for(ti, viewport_h);
            let thresh = size * 0.1;

            match target.mode {
                GizmoMode::Translate => {
                    for (axis, dir) in [(GizmoAxis::X, x), (GizmoAxis::Y, y), (GizmoAxis::Z, z)] {
                        let d = ray_segment_dist(ray_o, ray_d, pos, pos + dir * size);
                        if d < thresh {
                            consider(d, ti, axis);
                        }
                    }
                    for (axis, a, b) in [
                        (GizmoAxis::Xy, x, y),
                        (GizmoAxis::Yz, y, z),
                        (GizmoAxis::Zx, z, x),
                    ] {
                        if let Some(d) = ray_plane_quad_dist(ray_o, ray_d, pos, a, b, size) {
                            consider(d, ti, axis);
                        }
                    }
                }
                GizmoMode::Rotate => {
                    for (axis, n) in [(GizmoAxis::X, x), (GizmoAxis::Y, y), (GizmoAxis::Z, z)] {
                        if let Some(d) = ray_ring_dist(ray_o, ray_d, pos, n, size) {
                            consider(d, ti, axis);
                        }
                    }
                }
                GizmoMode::Scale => {
                    // Uniform center first — prefer over axes near the pivot
                    if ray_sphere_hit(ray_o, ray_d, pos, size * 0.2).is_some() {
                        consider(0.0, ti, GizmoAxis::Uniform);
                    } else {
                        for (axis, dir) in [(GizmoAxis::X, x), (GizmoAxis::Y, y), (GizmoAxis::Z, z)]
                        {
                            let d = ray_segment_dist(ray_o, ray_d, pos, pos + dir * size);
                            if d < thresh {
                                consider(d, ti, axis);
                            }
                        }
                    }
                }
            }
        }
        best.map(|(_, ti, a)| (ti, a))
    }

    fn begin_drag(&mut self, ti: usize, axis: GizmoAxis, ray_o: Vec3, ray_d: Vec3) {
        let tf = self.tf(ti);
        let pos = tf.translation;
        let (x, y, z) = self.axes(ti);
        let mode = self.targets[ti].mode;

        let (axis_dir, plane_n, plane_u, plane_v) = match axis {
            GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z if mode == GizmoMode::Rotate => {
                let (u, v) = gizmo_ring_basis(axis, x, y, z);
                let n = match axis {
                    GizmoAxis::X => x,
                    GizmoAxis::Y => y,
                    _ => z,
                };
                (n, n, u, v)
            }
            GizmoAxis::X => (x, x, y, z),
            GizmoAxis::Y => (y, y, x, z),
            GizmoAxis::Z => (z, z, x, y),
            GizmoAxis::Xy => ((x + y).normalize_or_zero(), z, x, y),
            GizmoAxis::Yz => ((y + z).normalize_or_zero(), x, y, z),
            GizmoAxis::Zx => ((z + x).normalize_or_zero(), y, z, x),
            GizmoAxis::Uniform => {
                let view = (self.scene.camera.eye - pos).normalize_or_zero();
                let u = view.any_orthonormal_vector();
                let v = view.cross(u);
                (view, view, u, v)
            }
        };

        let grab = match mode {
            GizmoMode::Translate => match axis {
                GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                    let view = (self.scene.camera.eye - pos).normalize_or_zero();
                    let n = axis_dir.cross(view.cross(axis_dir)).normalize_or_zero();
                    ray_plane_point(ray_o, ray_d, pos, n).unwrap_or(pos)
                }
                _ => ray_plane_point(ray_o, ray_d, pos, plane_n).unwrap_or(pos),
            },
            GizmoMode::Rotate => {
                ray_plane_point(ray_o, ray_d, pos, axis_dir).unwrap_or(pos + plane_u)
            }
            GizmoMode::Scale => {
                let view = (self.scene.camera.eye - pos).normalize_or_zero();
                let n = if axis == GizmoAxis::Uniform {
                    view
                } else {
                    axis_dir.cross(view.cross(axis_dir)).normalize_or_zero()
                };
                ray_plane_point(ray_o, ray_d, pos, n).unwrap_or(pos + axis_dir)
            }
        };

        let start_angle = {
            let d = (grab - pos).normalize_or_zero();
            d.dot(plane_v).atan2(d.dot(plane_u))
        };

        self.drag = Some(Drag {
            target: ti,
            axis,
            start_t: tf,
            axis_dir,
            plane_n,
            plane_u,
            plane_v,
            grab,
            start_angle,
            current_angle: start_angle,
            start_scale_along: (grab - pos).dot(axis_dir),
        });
    }

    fn update_drag(&mut self, ray_o: Vec3, ray_d: Vec3) {
        let Some(drag) = self.drag.as_ref() else {
            return;
        };
        let drag = *drag;
        let ti = drag.target;
        let mode = self.targets[ti].mode;
        let pos0 = drag.start_t.translation;
        let mut tf = drag.start_t;
        let mut current_angle = drag.current_angle;

        match mode {
            GizmoMode::Translate => {
                let hit = match drag.axis {
                    GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                        let view = (self.scene.camera.eye - pos0).normalize_or_zero();
                        let n = drag
                            .axis_dir
                            .cross(view.cross(drag.axis_dir))
                            .normalize_or_zero();
                        ray_plane_point(ray_o, ray_d, pos0, n)
                    }
                    _ => ray_plane_point(ray_o, ray_d, pos0, drag.plane_n),
                };
                if let Some(hit) = hit {
                    let delta = hit - drag.grab;
                    tf.translation = match drag.axis {
                        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                            pos0 + drag.axis_dir * delta.dot(drag.axis_dir)
                        }
                        GizmoAxis::Xy | GizmoAxis::Yz | GizmoAxis::Zx => {
                            pos0
                                + drag.plane_u * delta.dot(drag.plane_u)
                                + drag.plane_v * delta.dot(drag.plane_v)
                        }
                        GizmoAxis::Uniform => pos0,
                    };
                }
            }
            GizmoMode::Rotate => {
                if let Some(hit) = ray_plane_point(ray_o, ray_d, pos0, drag.axis_dir) {
                    let d = (hit - pos0).normalize_or_zero();
                    current_angle = d.dot(drag.plane_v).atan2(d.dot(drag.plane_u));
                    let mut delta = current_angle - drag.start_angle;
                    while delta > std::f32::consts::PI {
                        delta -= TAU;
                    }
                    while delta < -std::f32::consts::PI {
                        delta += TAU;
                    }
                    tf.rotation =
                        Quat::from_axis_angle(drag.axis_dir, delta) * drag.start_t.rotation;
                }
            }
            GizmoMode::Scale => {
                let view = (self.scene.camera.eye - pos0).normalize_or_zero();
                let n = if drag.axis == GizmoAxis::Uniform {
                    view
                } else {
                    drag.axis_dir
                        .cross(view.cross(drag.axis_dir))
                        .normalize_or_zero()
                };
                if let Some(hit) = ray_plane_point(ray_o, ray_d, pos0, n) {
                    match drag.axis {
                        GizmoAxis::Uniform => {
                            let along = (hit - pos0).length();
                            let start = (drag.grab - pos0).length().max(1e-4);
                            let s = (along / start).clamp(0.05, 20.0);
                            tf.scale = drag.start_t.scale * s;
                        }
                        _ => {
                            let along = (hit - pos0).dot(drag.axis_dir);
                            let start = drag.start_scale_along.signum()
                                * drag.start_scale_along.abs().max(1e-4);
                            let s = (along / start).clamp(0.05, 20.0);
                            let mut scale = drag.start_t.scale;
                            match drag.axis {
                                GizmoAxis::X => scale.x = drag.start_t.scale.x * s,
                                GizmoAxis::Y => scale.y = drag.start_t.scale.y * s,
                                GizmoAxis::Z => scale.z = drag.start_t.scale.z * s,
                                _ => {}
                            }
                            tf.scale = scale;
                        }
                    }
                }
            }
        }

        if let Some(n) = self.scene.nodes.get_mut(self.targets[ti].node) {
            n.local = tf;
        }
        if let Some(d) = self.drag.as_mut() {
            d.current_angle = current_angle;
        }
    }

    fn frame(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        self.time += dt;
        self.fps_accum += dt;
        self.fps_frames += 1;
        if self.fps_accum >= 0.5 {
            self.fps = self.fps_frames as f32 / self.fps_accum;
            self.fps_accum = 0.0;
            self.fps_frames = 0;
        }

        let (w, h) = {
            let size = self.window.as_ref().unwrap().inner_size();
            (size.width.max(1), size.height.max(1))
        };
        let aspect = w as f32 / h as f32;

        self.fly.apply(dt, &mut self.scene);

        let viewport = Vec2::new(w as f32, h as f32);
        let over_view = view_gizmo::contains_cursor(viewport, self.cursor);
        let (ray_o, ray_d) = cursor_ray(&self.scene.camera, aspect, self.cursor, viewport);

        if self.drag.is_some() {
            self.hover = self.drag.as_ref().map(|d| (d.target, d.axis));
            self.update_drag(ray_o, ray_d);
            if self.mouse_released {
                self.drag = None;
            }
        } else if !self.fly.looking && over_view {
            // View gizmo eats LMB — snap fly cam to the picked axis.
            self.hover = None;
            if self.mouse_pressed {
                if let Some(axis) = view_gizmo::hit_test(&self.scene.camera, viewport, self.cursor)
                {
                    self.fly.snap_view(axis, Vec3::new(0.0, 0.8, 0.0));
                }
            }
        } else if !self.fly.looking {
            self.hover = self.pick(ray_o, ray_d, h as f32);
            if self.mouse_pressed {
                if let Some((ti, axis)) = self.hover {
                    self.begin_drag(ti, axis, ray_o, ray_d);
                }
            }
        } else {
            self.hover = None;
        }

        self.draw_debug(h as f32);

        // HUD
        let input = InputFrame {
            cursor: self.cursor,
            mouse_down: self.mouse_down,
            mouse_pressed: self.mouse_pressed,
            mouse_released: self.mouse_released,
            scroll_delta: Vec2::ZERO,
            dt,
        };
        self.scene.hud.begin(&input, viewport);
        let text = [0.92, 0.93, 0.96, 1.0];
        let space = if self.local_space { "Local" } else { "World" };
        self.scene
            .hud
            .text(Vec2::new(16.0, 16.0), "DEBUG DRAW", text, 2.0);
        self.scene.hud.text(
            Vec2::new(16.0, 40.0),
            &format!("fps {:.0}  ·  {space}", self.fps),
            text,
            2.0,
        );
        self.scene.hud.text(
            Vec2::new(16.0, 64.0),
            "cubes: move · rotate · scale   Space local/world",
            [0.78, 0.8, 0.86, 0.95],
            2.0,
        );
        self.scene.hud.text(
            Vec2::new(16.0, 88.0),
            "orange=depth  green=overlay  ·  LMB gizmo  ·  RMB look",
            [0.78, 0.8, 0.86, 0.95],
            2.0,
        );
        self.scene.hud.text(
            Vec2::new(16.0, h as f32 - 28.0),
            "WASD move · view gizmo (top-right) · Esc quit",
            [0.7, 0.74, 0.8, 0.9],
            2.0,
        );
        view_gizmo::draw(
            &mut self.scene.hud,
            &self.scene.camera,
            viewport,
            self.cursor,
        );
        let _ = self.scene.hud.end();

        self.mouse_pressed = false;
        self.mouse_released = false;

        let window = self.window.as_ref().unwrap().clone();
        let gpu = self.gpu.as_mut().unwrap();
        gpu.visualizer.ensure_target(w, h);
        gpu.visualizer.sync(&self.scene);

        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        gpu.visualizer.render_to(&self.scene, aspect, &view);
        window.pre_present_notify();
        gpu.queue.present(frame);
        window.request_redraw();
    }

    fn draw_debug(&mut self, viewport_h: f32) {
        let t = self.time;
        let eye = self.scene.camera.eye;
        let hover = self.hover;
        let drag = self.drag;
        let local = self.local_space;
        let targets = self.targets;

        // Snapshot transforms / sizes before borrowing debug mutably
        let gizmos: [(Vec3, Quat, GizmoMode, f32); 3] = std::array::from_fn(|i| {
            let tf = self.scene.nodes.get(targets[i].node).unwrap().local;
            let rot = if local {
                tf.rotation
            } else {
                Quat::IDENTITY
            };
            let size = gizmo_screen_size(
                (eye - tf.translation).length(),
                self.scene.camera.fov_y,
                viewport_h,
                100.0,
            );
            (tf.translation, rot, targets[i].mode, size)
        });

        let dbg = &mut self.scene.debug;
        dbg.clear();

        // Ground grid + origin axes
        dbg.grid(
            Vec3::ZERO,
            6.0,
            1.0,
            [0.35, 0.38, 0.42, 0.55],
            5,
            [0.55, 0.58, 0.65, 0.9],
        );
        dbg.axes(Vec3::ZERO, 1.5);

        // Width samples (left)
        let widths = [1.0, 2.0, 4.0, 8.0];
        for (i, &w) in widths.iter().enumerate() {
            let y = 0.4 + i as f32 * 0.45;
            let a = Vec3::new(-5.5, y, -1.5);
            let b = Vec3::new(-3.0, y, -1.5);
            dbg.line(a, b, LineOpts::color([1.0, 0.85, 0.2, 1.0]).width(w));
            dbg.point_sized(a, [1.0, 0.85, 0.2, 1.0], 4.0 + w);
        }

        // Color gradient + taper (right)
        dbg.line(
            Vec3::new(3.0, 0.3, -1.5),
            Vec3::new(5.5, 2.2, -1.5),
            LineOpts {
                color_from: [1.0, 0.15, 0.2, 1.0],
                color_to: [0.2, 0.45, 1.0, 1.0],
                width_from: 10.0,
                width_to: 1.5,
                ..Default::default()
            },
        );

        // Depth vs overlay through the cubes row
        let through_a = Vec3::new(-4.0, 0.9, 0.0);
        let through_b = Vec3::new(4.0, 0.9, 0.0);
        dbg.line(
            through_a,
            through_b,
            LineOpts::color([1.0, 0.35, 0.15, 1.0]).width(3.0),
        );
        dbg.line(
            through_a + Vec3::Y * 0.35,
            through_b + Vec3::Y * 0.35,
            LineOpts::color([0.2, 1.0, 0.55, 1.0]).width(3.0).overlay(),
        );

        let orbit = Vec3::new(t.cos() * 2.2, 1.4, t.sin() * 2.2);
        dbg.sphere(orbit, 0.55, [0.4, 0.85, 1.0, 1.0], true);
        dbg.point_sized(orbit, [1.0, 1.0, 1.0, 1.0], 10.0);

        // Animated bone chain
        let base = Vec3::new(-4.0, 0.15, 2.5);
        let mid = base
            + Quat::from_axis_angle(Vec3::Y, t * 1.2)
                * Vec3::new(0.0, 1.1 + 0.2 * (t * 2.0).sin(), 0.8);
        let tip = mid
            + Quat::from_axis_angle(Vec3::Z, (t * 1.7).sin() * 0.7) * Vec3::new(0.0, 0.9, 0.2);
        dbg.bone(base, mid, [1.0, 0.55, 0.2, 1.0], true);
        dbg.bone(mid, tip, [0.95, 0.35, 0.55, 1.0], true);
        dbg.point_sized(base, [1.0, 0.55, 0.2, 1.0], 8.0);
        dbg.point_sized(mid, [1.0, 0.8, 0.3, 1.0], 7.0);
        dbg.point_sized(tip, [0.95, 0.35, 0.55, 1.0], 7.0);

        // Rainbow helix
        let helix_origin = Vec3::new(4.0, 0.2, 2.5);
        let segs = 48;
        for i in 0..segs {
            let u0 = i as f32 / segs as f32;
            let u1 = (i + 1) as f32 / segs as f32;
            let p = |u: f32| {
                let a = u * TAU * 3.0 + t;
                helix_origin + Vec3::new(a.cos() * 0.7, u * 2.4, a.sin() * 0.7)
            };
            let hue = |u: f32| {
                let h = (u + t * 0.1) % 1.0;
                let x = (h * 6.0).fract();
                match (h * 6.0) as i32 {
                    0 => [1.0, x, 0.0, 1.0],
                    1 => [1.0 - x, 1.0, 0.0, 1.0],
                    2 => [0.0, 1.0, x, 1.0],
                    3 => [0.0, 1.0 - x, 1.0, 1.0],
                    4 => [x, 0.0, 1.0, 1.0],
                    _ => [1.0, 0.0, 1.0 - x, 1.0],
                }
            };
            dbg.line(
                p(u0),
                p(u1),
                LineOpts {
                    color_from: hue(u0),
                    color_to: hue(u1),
                    width_from: 2.0 + 4.0 * (1.0 - u0),
                    width_to: 2.0 + 4.0 * (1.0 - u1),
                    ..Default::default()
                },
            );
        }

        // Point size ramp
        for i in 0..8 {
            let x = -2.8 + i as f32 * 0.8;
            let psz = 3.0 + i as f32 * 3.0;
            let pulse = 0.5 + 0.5 * (t * 3.0 + i as f32 * 0.4).sin();
            dbg.point_sized(
                Vec3::new(x, 0.15, 4.2),
                [0.3 + pulse * 0.5, 0.7, 1.0, 1.0],
                psz * (0.7 + 0.3 * pulse),
            );
        }

        dbg.axes_overlay(tip, 0.45);

        let spin = Mat4::from_translation(Vec3::new(0.0, 2.4, 3.2))
            * Mat4::from_quat(Quat::from_euler(
                glam::EulerRot::YXZ,
                t,
                t * 0.7,
                t * 0.35,
            ))
            * Mat4::from_scale(Vec3::splat(0.9));
        dbg.box_transform(spin, [0.7, 0.9, 1.0, 1.0], false);

        // Three interactive gizmos
        for (i, &(pos, rot, mode, size)) in gizmos.iter().enumerate() {
            let highlight = hover.and_then(|(ti, a)| (ti == i).then_some(a));
            let rotate_arc = drag.and_then(|d| {
                if d.target == i && mode == GizmoMode::Rotate {
                    Some(GizmoRotateArc {
                        axis: d.axis,
                        u: d.plane_u,
                        v: d.plane_v,
                        start: d.start_angle,
                        current: d.current_angle,
                    })
                } else {
                    None
                }
            });
            dbg.gizmo(
                pos,
                rot,
                GizmoOpts {
                    mode,
                    size,
                    highlight,
                    eye: Some(eye),
                    rotate_arc,
                    depth_test: false,
                },
            );
        }
    }
}

fn cursor_ray(cam: &Camera, aspect: f32, cursor: Vec2, viewport: Vec2) -> (Vec3, Vec3) {
    let vp = cam.view_proj(aspect);
    let inv = vp.inverse();
    let ndc = Vec2::new(
        (cursor.x / viewport.x) * 2.0 - 1.0,
        1.0 - (cursor.y / viewport.y) * 2.0,
    );
    let near = inv.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
    let far = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    let dir = (far - near).normalize_or_zero();
    (cam.eye, dir)
}

fn ray_plane_point(ro: Vec3, rd: Vec3, pos: Vec3, n: Vec3) -> Option<Vec3> {
    let n = n.normalize_or_zero();
    let denom = rd.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (pos - ro).dot(n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ro + rd * t)
}

fn ray_segment_dist(ro: Vec3, rd: Vec3, a: Vec3, b: Vec3) -> f32 {
    let u = b - a;
    let w0 = ro - a;
    let aa = rd.dot(rd);
    let bb = u.dot(u);
    let ab = rd.dot(u);
    let aw = rd.dot(w0);
    let bw = u.dot(w0);
    let den = aa * bb - ab * ab;
    let (s, t) = if den.abs() < 1e-8 {
        (0.0, (bw / bb).clamp(0.0, 1.0))
    } else {
        let s = ((ab * bw - bb * aw) / den).max(0.0);
        let t = ((aa * bw - ab * aw) / den).clamp(0.0, 1.0);
        (s, t)
    };
    let p = ro + rd * s;
    let q = a + u * t;
    (p - q).length()
}

fn ray_plane_quad_dist(
    ro: Vec3,
    rd: Vec3,
    pos: Vec3,
    a: Vec3,
    b: Vec3,
    size: f32,
) -> Option<f32> {
    let n = a.cross(b).normalize_or_zero();
    let hit = ray_plane_point(ro, rd, pos, n)?;
    let d = hit - pos;
    let ua = d.dot(a);
    let ub = d.dot(b);
    let u0 = size * 0.22;
    let u1 = size * 0.42;
    if ua >= u0 && ua <= u1 && ub >= u0 && ub <= u1 {
        Some((hit - ro).length() * 0.001) // prefer planes slightly when inside
    } else {
        None
    }
}

fn ray_ring_dist(ro: Vec3, rd: Vec3, pos: Vec3, axis: Vec3, radius: f32) -> Option<f32> {
    let hit = ray_plane_point(ro, rd, pos, axis)?;
    let r = (hit - pos).length();
    let radial = (r - radius).abs();
    if radial < radius * 0.12 {
        Some(radial)
    } else {
        None
    }
}

fn ray_sphere_hit(ro: Vec3, rd: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ro - center;
    let b = oc.dot(rd);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t > 0.0 {
        Some(0.0)
    } else {
        None
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            self.init_gpu(event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    if size.width > 0 && size.height > 0 {
                        gpu.config.width = size.width;
                        gpu.config.height = size.height;
                        gpu.surface.configure(&gpu.device, &gpu.config);
                    }
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Vec2::new(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    let down = state == ElementState::Pressed;
                    if down && !self.mouse_down {
                        self.mouse_pressed = true;
                    }
                    if !down && self.mouse_down {
                        self.mouse_released = true;
                    }
                    self.mouse_down = down;
                }
                if button == MouseButton::Right {
                    let down = state == ElementState::Pressed;
                    self.fly.looking = down;
                    if down {
                        self.drag = None;
                    }
                    if let Some(win) = self.window.as_ref() {
                        let _ = win.set_cursor_grab(if down {
                            CursorGrabMode::Locked
                        } else {
                            CursorGrabMode::None
                        });
                        win.set_cursor_visible(!down);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::Escape && down {
                        event_loop.exit();
                        return;
                    }
                    if down && self.drag.is_none() {
                        if code == KeyCode::Space {
                            self.local_space = !self.local_space;
                        }
                    }
                    self.fly.set_key(code, down);
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.fly.looking {
                self.fly.look_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(win) = self.window.as_ref() {
            win.request_redraw();
        }
    }
}
