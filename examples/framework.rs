//! Shared winit + wgpu host for mega-render examples.

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Vec2, Vec3};
use mega_render::{Camera, Scene, Visualizer, WgpuVisualizer};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

/// Per-example scene builder.
pub trait Demo {
    fn title() -> &'static str;
    fn window_size() -> (f64, f64) {
        (1280.0, 800.0)
    }
    fn build_scene() -> Scene;
    /// Optional post-process / visualizer tweaks after GPU init.
    fn configure(_visualizer: &mut WgpuVisualizer) {}
    /// Optional per-frame update. Return `true` to keep animating.
    fn update(_scene: &mut Scene, _dt: f32) -> bool {
        false
    }
}

struct FlyCam {
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    looking: bool,
    move_f: bool,
    move_b: bool,
    move_l: bool,
    move_r: bool,
    move_u: bool,
    move_d: bool,
    sprint: bool,
    look_delta: Vec2,
}

impl FlyCam {
    fn from_scene(scene: &Scene) -> Self {
        let mut cam = Self {
            eye: scene.camera.eye,
            yaw: 0.0,
            pitch: 0.0,
            looking: false,
            move_f: false,
            move_b: false,
            move_l: false,
            move_r: false,
            move_u: false,
            move_d: false,
            sprint: false,
            look_delta: Vec2::ZERO,
        };
        cam.sync_look(scene.camera.eye, scene.camera.target);
        cam
    }

    fn sync_look(&mut self, eye: Vec3, target: Vec3) {
        self.eye = eye;
        let d = (target - eye).normalize_or_zero();
        if d.length_squared() < 1e-8 {
            return;
        }
        self.pitch = d.y.clamp(-1.0, 1.0).asin();
        self.yaw = d.x.atan2(d.z);
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
    }

    fn moving(&self) -> bool {
        self.move_f || self.move_b || self.move_l || self.move_r || self.move_u || self.move_d
    }

    fn active(&self) -> bool {
        self.looking || self.moving()
    }

    fn set_key(&mut self, key: KeyCode, down: bool) {
        match key {
            KeyCode::KeyW => self.move_f = down,
            KeyCode::KeyS => self.move_b = down,
            KeyCode::KeyA => self.move_l = down,
            KeyCode::KeyD => self.move_r = down,
            KeyCode::KeyE => self.move_u = down,
            KeyCode::KeyQ => self.move_d = down,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.sprint = down,
            _ => {}
        }
    }

    fn add_look(&mut self, dx: f32, dy: f32) {
        if self.looking {
            self.look_delta += Vec2::new(dx, dy);
        }
    }

    fn update(&mut self, dt: f32) {
        const SENS: f32 = 0.0025;
        self.yaw += self.look_delta.x * SENS;
        self.pitch = (self.pitch - self.look_delta.y * SENS).clamp(-1.55, 1.55);
        self.look_delta = Vec2::ZERO;

        if !self.moving() {
            return;
        }

        let f = self.forward();
        let r = Vec3::Y.cross(f).normalize_or_zero();
        let mut dir = Vec3::ZERO;
        if self.move_f {
            dir += f;
        }
        if self.move_b {
            dir -= f;
        }
        if self.move_r {
            dir += r;
        }
        if self.move_l {
            dir -= r;
        }
        if self.move_u {
            dir += Vec3::Y;
        }
        if self.move_d {
            dir -= Vec3::Y;
        }
        if dir.length_squared() > 0.0 {
            let speed = if self.sprint { 12.0 } else { 4.0 };
            self.eye += dir.normalize() * speed * dt;
        }
    }

    fn apply(&self, scene: &mut Scene) {
        let eye = self.eye;
        let target = eye + self.forward();
        scene.camera = Camera::look_at(eye, target);
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    visualizer: WgpuVisualizer,
}

pub struct Host<D: Demo> {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    scene: Scene,
    fly: FlyCam,
    last_frame: Instant,
    animating: bool,
    _marker: std::marker::PhantomData<D>,
}

impl<D: Demo> Host<D> {
    fn new() -> Self {
        let scene = D::build_scene();
        let fly = FlyCam::from_scene(&scene);
        Self {
            window: None,
            gpu: None,
            scene,
            fly,
            last_frame: Instant::now(),
            animating: true,
            _marker: std::marker::PhantomData,
        }
    }

    fn set_looking(&mut self, looking: bool) {
        if self.fly.looking == looking {
            return;
        }
        self.fly.looking = looking;
        let Some(window) = &self.window else {
            return;
        };
        let mode = if looking {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        if window.set_cursor_grab(mode).is_err() && looking {
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
        }
        window.set_cursor_visible(!looking);
    }

    pub fn run() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
        let event_loop = EventLoop::new().expect("event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut host = Self::new();
        event_loop.run_app(&mut host).expect("run app");
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: true,
        }))
        .expect("no suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mega-render demo"),
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
        // Mesh pipeline is Rgba8UnormSrgb — match the swapchain to it.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba8UnormSrgb)
            .expect("adapter must support Rgba8UnormSrgb swapchain");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: caps
                .present_modes
                .iter()
                .copied()
                .find(|m| *m == wgpu::PresentMode::Fifo)
                .unwrap_or(caps.present_modes[0]),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let mut visualizer = WgpuVisualizer::new(&device, &queue);
        visualizer.ensure_target(width, height);
        D::configure(&mut visualizer);

        self.gpu = Some(Gpu {
            device,
            queue,
            surface,
            config,
            visualizer,
        });
        self.window = Some(window);
    }

    fn resize(&mut self, width: u32, height: u32) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        if width == 0 || height == 0 {
            return;
        }
        gpu.config.width = width;
        gpu.config.height = height;
        gpu.surface.configure(&gpu.device, &gpu.config);
        gpu.visualizer.ensure_target(width, height);
    }

    fn redraw(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.fly.update(dt);
        self.scene.update_animations(dt);
        self.animating = D::update(&mut self.scene, dt) || self.fly.active();
        self.fly.apply(&mut self.scene);

        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        gpu.visualizer.ensure_target(gpu.config.width, gpu.config.height);

        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                window.request_redraw();
                return;
            }
            _ => {
                window.request_redraw();
                return;
            }
        };

        let view = frame.texture.create_view(&Default::default());
        gpu.visualizer.sync(&self.scene);
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        gpu.visualizer.render_to(&self.scene, aspect, &view);

        window.pre_present_notify();
        gpu.queue.present(frame);
    }
}

impl<D: Demo> ApplicationHandler for Host<D> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let (w, h) = D::window_size();
        let attrs = Window::default_attributes()
            .with_title(D::title())
            .with_inner_size(LogicalSize::new(w, h));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.init_gpu(window.clone());
        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let playing =
            self.animating || self.fly.active() || self.scene.animators.iter().any(|a| a.playing);
        let ms = if playing { 8 } else { 33 };
        event_loop.set_control_flow(ControlFlow::wait_duration(Duration::from_millis(ms)));
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.fly.add_look(delta.0 as f32, delta.1 as f32);
            if self.fly.looking {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.set_looking(state == ElementState::Pressed);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Focused(false) => self.set_looking(false),
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.fly.set_key(code, down);
                    if down && code == KeyCode::Escape {
                        if self.fly.looking {
                            self.set_looking(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
