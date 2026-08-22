//! Simple scene + in-engine HUD (no mega-ui).
//!
//! Move: WASD · Look: RMB · Quit: Esc
//!
//! ```text
//! cargo run --example hud_demo
//! ```

use std::sync::Arc;
use std::time::Instant;

use glam::{Vec2, Vec3};
use mega_render::{
    cube, plane, sphere, Handle, HudRect, InputFrame, Light, Material, Node, Scene, Transform,
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

struct FlyCam {
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    looking: bool,
    keys: [bool; 6], // W S A D E Q
    sprint: bool,
    look_delta: Vec2,
}

impl FlyCam {
    fn new(eye: Vec3, target: Vec3) -> Self {
        let d = (target - eye).normalize_or_zero();
        let pitch = d.y.clamp(-1.0, 1.0).asin();
        let yaw = d.x.atan2(d.z);
        Self {
            eye,
            yaw,
            pitch,
            looking: false,
            keys: [false; 6],
            sprint: false,
            look_delta: Vec2::ZERO,
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

    fn apply(&mut self, dt: f32, scene: &mut Scene) {
        const SENS: f32 = 0.0025;
        self.yaw += self.look_delta.x * SENS;
        self.pitch = (self.pitch - self.look_delta.y * SENS).clamp(-1.55, 1.55);
        self.look_delta = Vec2::ZERO;

        let forward = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );
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
    input: InputFrame,
    cursor: Vec2,
    mouse_down: bool,
    want_capture: bool,
    last: Instant,
    fps: f32,
    fps_accum: f32,
    fps_frames: u32,
    cube_mesh: Handle<mega_render::Mesh>,
    sphere_mesh: Handle<mega_render::Mesh>,
    root: Handle<Node>,
    spawn_count: u32,
    light_on: bool,
    status: String,
}

impl App {
    fn new() -> Self {
        let mut scene = Scene::new();
        if let Some(Light::Directional(d)) = scene.lights.first_mut() {
            d.intensity = 2.8;
            d.direction = Vec3::new(0.35, -0.55, 0.75);
        }
        scene.ambient = [0.04, 0.04, 0.05];
        scene.camera.eye = Vec3::new(4.0, 3.0, 6.0);
        scene.camera.target = Vec3::new(0.0, 0.5, 0.0);

        let mesh_plane = scene.meshes.insert(plane(12.0, 12.0));
        let cube_mesh = scene.meshes.insert(cube(0.7));
        let sphere_mesh = scene.meshes.insert(sphere(0.4, 24, 16));
        let mat_ground = scene
            .materials
            .insert(Material::new([0.55, 0.55, 0.58, 1.0], 0.0, 0.85));
        let mat_cube = scene
            .materials
            .insert(Material::new([0.85, 0.25, 0.2, 1.0], 0.0, 0.4));

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
        scene.nodes.insert(Node {
            name: "cube0".into(),
            parent: Some(root),
            local: Transform::from_translation(Vec3::new(0.0, 0.35, 0.0)),
            mesh: Some(cube_mesh),
            material: Some(mat_cube),
            skin: None,
            visible: true,
        });

        let fly = FlyCam::new(scene.camera.eye, scene.camera.target);
        Self {
            window: None,
            gpu: None,
            scene,
            fly,
            input: InputFrame::default(),
            cursor: Vec2::ZERO,
            mouse_down: false,
            want_capture: false,
            last: Instant::now(),
            fps: 0.0,
            fps_accum: 0.0,
            fps_frames: 0,
            cube_mesh,
            sphere_mesh,
            root,
            spawn_count: 0,
            light_on: true,
            status: "готово".into(),
        }
    }

    fn init_gpu(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("mega-render · hud_demo")
                        .with_inner_size(LogicalSize::new(1280.0, 720.0)),
                )
                .expect("window"),
        );
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("surface");
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
            label: Some("hud_demo"),
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
        // Light post so the scene isn't raw HDR.
        visualizer.post_process().tonemap.exposure = 1.0;

        self.gpu = Some(Gpu {
            device,
            queue,
            surface,
            config,
            visualizer,
        });
        self.window = Some(window);
    }

    fn spawn_object(&mut self, sphere: bool) {
        self.spawn_count += 1;
        let i = self.spawn_count;
        let x = ((i % 5) as f32 - 2.0) * 1.2;
        let z = ((i / 5) as f32) * -1.2;
        let hue = (i as f32 * 0.17) % 1.0;
        let (r, g, b) = match (hue * 6.0) as i32 {
            0 => (1.0, hue * 6.0, 0.0),
            1 => (1.0 - (hue * 6.0 - 1.0), 1.0, 0.0),
            2 => (0.0, 1.0, hue * 6.0 - 2.0),
            3 => (0.0, 1.0 - (hue * 6.0 - 3.0), 1.0),
            4 => (hue * 6.0 - 4.0, 0.0, 1.0),
            _ => (1.0, 0.0, 1.0 - (hue * 6.0 - 5.0)),
        };
        let mat = self.scene.materials.insert(Material::new(
            [0.25 + r * 0.7, 0.25 + g * 0.7, 0.25 + b * 0.7, 1.0],
            if sphere { 0.8 } else { 0.05 },
            0.35,
        ));
        let mesh = if sphere {
            self.sphere_mesh
        } else {
            self.cube_mesh
        };
        let y = if sphere { 0.4 } else { 0.35 };
        self.scene.nodes.insert(Node {
            name: format!("spawn_{i}"),
            parent: Some(self.root),
            local: Transform::from_translation(Vec3::new(x, y, z)),
            mesh: Some(mesh),
            material: Some(mat),
            skin: None,
            visible: true,
        });
        self.status = format!("спавн #{i}");
    }

    fn clear_spawns(&mut self) {
        let root = self.root;
        let kill: Vec<_> = self
            .scene
            .nodes
            .iter()
            .filter(|(h, n)| {
                h.key() != root.key()
                    && n.parent.map(|p| p.key()) == Some(root.key())
                    && n.name.starts_with("spawn_")
            })
            .map(|(h, _)| h)
            .collect();
        for h in kill {
            self.scene.nodes.remove(h);
        }
        self.spawn_count = 0;
        self.status = "очищено".into();
    }

    fn build_hud(&mut self, w: f32, h: f32) {
        let capture_before = self.want_capture;
        self.scene.hud.begin(&self.input, Vec2::new(w, h));

        // Info panel
        let panel = HudRect::from_min_size(Vec2::new(16.0, 16.0), Vec2::new(300.0, 140.0));
        self.scene.hud.panel(panel, [0.06, 0.07, 0.1, 0.82]);
        let text = [0.92, 0.93, 0.95, 1.0];
        self.scene
            .hud
            .text(Vec2::new(28.0, 28.0), "HUD ДЕМО", text, 2.0);
        self.scene.hud.text(
            Vec2::new(28.0, 52.0),
            &format!("fps  {:.0}", self.fps),
            text,
            2.0,
        );
        self.scene.hud.text(
            Vec2::new(28.0, 72.0),
            &format!("ноды {}", self.scene.nodes.iter().count()),
            text,
            2.0,
        );
        self.scene.hud.text(
            Vec2::new(28.0, 92.0),
            &format!("ui capture {}", if capture_before { "ДА" } else { "нет" }),
            text,
            2.0,
        );
        self.scene.hud.text(
            Vec2::new(28.0, 112.0),
            &format!("статус: {}", self.status),
            text,
            2.0,
        );

        // Buttons
        let bw = 160.0;
        let bh = 36.0;
        let bx = 16.0;
        let mut by = 170.0;
        if self
            .scene
            .hud
            .button("add_cube", HudRect::from_min_size(Vec2::new(bx, by), Vec2::new(bw, bh)), "Куб")
        {
            self.spawn_object(false);
        }
        by += bh + 10.0;
        if self.scene.hud.button(
            "add_sphere",
            HudRect::from_min_size(Vec2::new(bx, by), Vec2::new(bw, bh)),
            "Сфера",
        ) {
            self.spawn_object(true);
        }
        by += bh + 10.0;
        if self.scene.hud.button(
            "clear",
            HudRect::from_min_size(Vec2::new(bx, by), Vec2::new(bw, bh)),
            "Очистить",
        ) {
            self.clear_spawns();
        }
        by += bh + 10.0;
        let light_label = if self.light_on {
            "Свет: ВКЛ"
        } else {
            "Свет: ВЫКЛ"
        };
        if self.scene.hud.button(
            "toggle_light",
            HudRect::from_min_size(Vec2::new(bx, by), Vec2::new(bw, bh)),
            light_label,
        ) {
            self.light_on = !self.light_on;
            if let Some(Light::Directional(d)) = self.scene.lights.first_mut() {
                d.enabled = self.light_on;
            }
            self.status = light_label.into();
        }

        // Hint bottom
        self.scene.hud.text(
            Vec2::new(16.0, h - 28.0),
            "ПКМ обзор · WASD ход · Esc выход",
            [0.75, 0.78, 0.85, 0.9],
            2.0,
        );

        let out = self.scene.hud.end();
        self.want_capture = out.want_capture_mouse;
    }

    fn frame(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
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

        self.input = InputFrame {
            cursor: self.cursor,
            mouse_down: self.mouse_down,
            mouse_pressed: self.input.mouse_pressed,
            mouse_released: self.input.mouse_released,
            scroll_delta: Vec2::ZERO,
            dt,
        };

        self.build_hud(w as f32, h as f32);

        // Keyboard / fly always work — want_capture_mouse only blocks world clicks.
        self.fly.apply(dt, &mut self.scene);

        self.input.mouse_pressed = false;
        self.input.mouse_released = false;

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
        let aspect = w as f32 / h as f32;
        gpu.visualizer.render_to(&self.scene, aspect, &view);
        window.pre_present_notify();
        gpu.queue.present(frame);
        window.request_redraw();
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
                        self.input.mouse_pressed = true;
                    }
                    if !down && self.mouse_down {
                        self.input.mouse_released = true;
                    }
                    self.mouse_down = down;
                }
                if button == MouseButton::Right {
                    let down = state == ElementState::Pressed;
                    self.fly.looking = down;
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
