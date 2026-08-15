//! Shared winit + wgpu host for mega-render examples.
//!
//! Renders the 3D scene into an offscreen target shown inside a mega-ui dock
//! viewport. `mega-ui` is a **dev-dependency only** — the library itself does
//! not depend on UI (no cyclic crates).

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Vec2, Vec3};
use mega_render::{
    Camera, DebugView, PostProcessSettings, Scene, ShadowSettings, Visualizer, WgpuVisualizer,
};
use mega_ui::wgpu::{DrawStats, UiRenderer};
use mega_ui::{CursorIcon, DockState, Ui, UiInput};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

/// Texture slot for the 3D viewport (`ui.texture(SCENE_TEX, …)`).
pub const SCENE_TEX: u32 = 0;

/// Per-frame UI context passed to [`Demo::build_ui`].
pub struct UiCtx<'a> {
    pub scene: &'a mut Scene,
    pub post: &'a mut PostProcessSettings,
    pub shadow: &'a mut ShadowSettings,
    pub debug_view: &'a mut DebugView,
    pub dock: &'a mut DockState,
    /// Full window size in pixels (for dock layout).
    pub window_size: Vec2,
    /// Set this to the scene pane size in **pixels** (from `ui.available_size()`).
    pub viewport_size: &'a mut Vec2,
    pub dt: f32,
    /// Frames-per-second averaged over the last ~1 second.
    pub fps: f32,
    /// Mean frame time in milliseconds over the last ~1 second.
    pub frame_ms: f32,
    pub stats: DrawStats,
}

/// Per-example scene + dock UI.
pub trait Demo {
    fn title() -> &'static str;
    fn window_size() -> (f64, f64) {
        (1440.0, 900.0)
    }
    fn build_scene() -> Scene;
    fn configure(_visualizer: &mut WgpuVisualizer) {}
    fn init_ui(ui: &mut Ui) {
        ui.load_builtin_icons();
    }
    /// Optional per-frame scene update. Return `true` to keep animating.
    fn update(_scene: &mut Scene, _dt: f32) -> bool {
        false
    }
    /// Build dock / widgets. Return `true` to keep redrawing.
    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool;
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
        let focus_distance = scene.camera.focus_distance;
        let focus_target = scene.camera.focus_target;
        let focus_smooth = scene.camera.focus_smooth;
        let f_stop = scene.camera.f_stop;
        scene.camera = Camera::look_at(eye, target);
        scene.camera.focus_distance = focus_distance;
        scene.camera.focus_target = focus_target;
        scene.camera.focus_smooth = focus_smooth;
        scene.camera.f_stop = f_stop;
    }
}

struct SceneTarget {
    _texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    size: (u32, u32),
}

impl SceneTarget {
    fn ensure(device: &wgpu::Device, ui: &mut UiRenderer, size: (u32, u32)) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("demo scene color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_view = texture.create_view(&Default::default());
        ui.bind_texture_view(device, SCENE_TEX, texture.create_view(&Default::default()));
        Self {
            _texture: texture,
            render_view,
            size: (w, h),
        }
    }

    fn resize(&mut self, device: &wgpu::Device, ui: &mut UiRenderer, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.size == (w, h) {
            return;
        }
        *self = Self::ensure(device, ui, (w, h));
    }
}

#[derive(Default)]
struct FrameInput {
    mouse_pos: Vec2,
    mouse_down: bool,
    mouse_pressed: bool,
    mouse_released: bool,
    mouse_right_down: bool,
    mouse_right_pressed: bool,
    mouse_right_released: bool,
    scroll_delta: Vec2,
    text: String,
    key_backspace: bool,
    key_enter: bool,
    key_left: bool,
    key_right: bool,
    key_up: bool,
    key_down: bool,
    key_home: bool,
    key_end: bool,
    key_shift: bool,
    key_ctrl: bool,
    key_copy: bool,
    key_paste: bool,
    key_cut: bool,
    key_select_all: bool,
    modifiers: winit::keyboard::ModifiersState,
    clipboard_paste: String,
}

impl FrameInput {
    fn clear_edges(&mut self) {
        self.mouse_pressed = false;
        self.mouse_released = false;
        self.mouse_right_pressed = false;
        self.mouse_right_released = false;
        self.scroll_delta = Vec2::ZERO;
        self.text.clear();
        self.key_backspace = false;
        self.key_enter = false;
        self.key_left = false;
        self.key_right = false;
        self.key_up = false;
        self.key_down = false;
        self.key_home = false;
        self.key_end = false;
        self.key_copy = false;
        self.key_paste = false;
        self.key_cut = false;
        self.key_select_all = false;
        self.clipboard_paste.clear();
    }

    fn shortcut_mod(&self) -> bool {
        self.modifiers.control_key() || self.modifiers.super_key()
    }

    fn to_ui(&self, viewport: Vec2, dt: f32) -> UiInput {
        UiInput {
            mouse_pos: self.mouse_pos,
            mouse_down: self.mouse_down,
            mouse_pressed: self.mouse_pressed,
            mouse_released: self.mouse_released,
            mouse_right_down: self.mouse_right_down,
            mouse_right_pressed: self.mouse_right_pressed,
            mouse_right_released: self.mouse_right_released,
            viewport,
            scroll_delta: self.scroll_delta,
            dt,
            text: self.text.clone(),
            key_backspace: self.key_backspace,
            key_enter: self.key_enter,
            key_left: self.key_left,
            key_right: self.key_right,
            key_up: self.key_up,
            key_down: self.key_down,
            key_home: self.key_home,
            key_end: self.key_end,
            key_shift: self.key_shift,
            key_ctrl: self.key_ctrl || self.modifiers.super_key(),
            key_copy: self.key_copy,
            key_paste: self.key_paste,
            key_cut: self.key_cut,
            key_select_all: self.key_select_all,
            clipboard: self.clipboard_paste.clone(),
            mouse_middle_down: false,
            mouse_middle_pressed: false,
            mouse_middle_released: false,
            key_delete: false,
            key_duplicate: false,
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    visualizer: WgpuVisualizer,
    ui_renderer: UiRenderer,
    scene_target: SceneTarget,
}

pub struct Host<D: Demo> {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    scene: Scene,
    fly: FlyCam,
    ui: Ui,
    dock: DockState,
    input: FrameInput,
    last_frame: Instant,
    fps_accum_dt: f32,
    fps_frames: u32,
    fps: f32,
    frame_ms: f32,
    animating: bool,
    cursor: CursorIcon,
    draw_stats: DrawStats,
    clipboard: Option<arboard::Clipboard>,
    want_capture_mouse: bool,
    want_capture_keyboard: bool,
    viewport_size: Vec2,
    _marker: std::marker::PhantomData<D>,
}

impl<D: Demo> Host<D> {
    fn new(dock: DockState) -> Self {
        let scene = D::build_scene();
        let fly = FlyCam::from_scene(&scene);
        let mut ui = Ui::new();
        D::init_ui(&mut ui);
        Self {
            window: None,
            gpu: None,
            scene,
            fly,
            ui,
            dock,
            input: FrameInput::default(),
            last_frame: Instant::now(),
            fps_accum_dt: 0.0,
            fps_frames: 0,
            fps: 0.0,
            frame_ms: 0.0,
            animating: true,
            cursor: CursorIcon::Default,
            draw_stats: DrawStats::default(),
            clipboard: arboard::Clipboard::new().ok(),
            want_capture_mouse: false,
            want_capture_keyboard: false,
            viewport_size: Vec2::new(1280.0, 720.0),
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

    pub fn run(dock: DockState) {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
        let event_loop = EventLoop::new().expect("event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut host = Self::new(dock);
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
            // Need real hardware limits — buckets pin color-attachment bytes to 32,
            // which is already full for our 4-target G-buffer before velocity.
            apply_limit_buckets: false,
        }))
        .expect("no suitable GPU adapter");

        let mut limits = wgpu::Limits::default();
        // G-buffer is already 2×rgba16float + 2×rgba8unorm = 32 bytes/sample
        // (rgba8 counts as 8 in the WebGPU table). Velocity (rg16float = 4) needs ≥36.
        let adapter_bytes = adapter.limits().max_color_attachment_bytes_per_sample;
        assert!(
            adapter_bytes >= 36,
            "GPU max_color_attachment_bytes_per_sample={adapter_bytes}, need ≥36 for velocity MRT"
        );
        limits.max_color_attachment_bytes_per_sample = adapter_bytes.min(64);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mega-render demo"),
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
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
        let vp = (
            self.viewport_size.x.max(1.0) as u32,
            self.viewport_size.y.max(1.0) as u32,
        );
        visualizer.ensure_target(vp.0, vp.1);
        D::configure(&mut visualizer);

        let mut ui_renderer = UiRenderer::new(&device, &queue, format, &self.ui);
        ui_renderer.set_viewport(&queue, width as f32, height as f32);
        let scene_target = SceneTarget::ensure(&device, &mut ui_renderer, vp);

        self.gpu = Some(Gpu {
            device,
            queue,
            surface,
            config,
            visualizer,
            ui_renderer,
            scene_target,
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
        gpu.ui_renderer
            .set_viewport(&gpu.queue, width as f32, height as f32);
    }

    fn begin_paste(&mut self) {
        self.input.key_paste = true;
        if !self.input.clipboard_paste.is_empty() {
            return;
        }
        if let Some(cb) = self.clipboard.as_mut() {
            if let Ok(text) = cb.get_text() {
                self.input.clipboard_paste = text;
            }
        }
    }

    fn apply_cursor(&mut self, window: &Window, cursor: CursorIcon) {
        if self.fly.looking || cursor == self.cursor {
            return;
        }
        self.cursor = cursor;
        window.set_cursor(map_cursor(cursor));
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
        self.fps_accum_dt += dt;
        self.fps_frames += 1;
        if self.fps_accum_dt >= 1.0 {
            self.fps = self.fps_frames as f32 / self.fps_accum_dt;
            self.frame_ms = (self.fps_accum_dt / self.fps_frames.max(1) as f32) * 1000.0;
            self.fps_accum_dt = 0.0;
            self.fps_frames = 0;
        }

        if !self.want_capture_keyboard || self.fly.looking {
            self.fly.update(dt);
        }
        self.scene.poll_loads();
        self.scene.update_animations(dt);
        let demo_anim = D::update(&mut self.scene, dt);
        self.fly.apply(&mut self.scene);
        if let Some(gpu) = self.gpu.as_mut() {
            if gpu.visualizer.post_process().dof.auto_focus {
                self.scene.camera.autofocus_ground(0.0);
            }
        }
        self.scene.camera.tick_focus(dt);

        let viewport = Vec2::new(size.width as f32, size.height as f32);
        // While looking, starve UI of mouse so dock widgets don't steal the grab.
        let ui_input = if self.fly.looking {
            let mut starved = self.input.to_ui(viewport, dt);
            starved.mouse_down = false;
            starved.mouse_pressed = false;
            starved.mouse_released = false;
            starved.mouse_right_down = false;
            starved.mouse_right_pressed = false;
            starved.mouse_right_released = false;
            starved.scroll_delta = Vec2::ZERO;
            starved
        } else {
            self.input.to_ui(viewport, dt)
        };

        self.ui.begin_frame(ui_input);

        let cursor = {
            let Some(gpu) = self.gpu.as_mut() else {
                return;
            };

            let mut viewport_size = self.viewport_size;
            let mut debug_view = gpu.visualizer.debug_view();
            let keep_ui = {
                let (post, shadow) = gpu.visualizer.effect_settings();
                let mut ctx = UiCtx {
                    scene: &mut self.scene,
                    post,
                    shadow,
                    debug_view: &mut debug_view,
                    dock: &mut self.dock,
                    window_size: viewport,
                    viewport_size: &mut viewport_size,
                    dt,
                    fps: self.fps,
                    frame_ms: self.frame_ms,
                    stats: self.draw_stats,
                };
                D::build_ui(&mut self.ui, &mut ctx)
            };
            gpu.visualizer.set_debug_view(debug_view);
            self.viewport_size = viewport_size;

            let out = self.ui.end_frame();
            self.want_capture_mouse = out.want_capture_mouse;
            self.want_capture_keyboard = out.want_capture_keyboard;
            self.animating = demo_anim || self.fly.active() || keep_ui || out.needs_repaint;

            if let Some(text) = out.clipboard {
                if let Some(cb) = self.clipboard.as_mut() {
                    let _ = cb.set_text(text);
                }
            }
            let cursor = out.cursor;
            let draw_list = out.draw_list;

            let vp_w = self.viewport_size.x.round().max(1.0) as u32;
            let vp_h = self.viewport_size.y.round().max(1.0) as u32;
            gpu.scene_target
                .resize(&gpu.device, &mut gpu.ui_renderer, vp_w, vp_h);
            gpu.visualizer.ensure_target(vp_w, vp_h);

            gpu.visualizer.sync(&self.scene);
            let aspect = vp_w as f32 / vp_h as f32;
            gpu.visualizer
                .render_to(&self.scene, aspect, &gpu.scene_target.render_view);

            gpu.ui_renderer
                .sync_atlases(&gpu.device, &gpu.queue, &mut self.ui);

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
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("demo frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.04,
                                g: 0.04,
                                b: 0.04,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                self.draw_stats = gpu.ui_renderer.draw(&gpu.queue, &mut pass, &draw_list);
            }

            gpu.queue.submit(Some(encoder.finish()));
            window.pre_present_notify();
            gpu.queue.present(frame);
            cursor
        };

        self.input.clear_edges();
        self.apply_cursor(&window, cursor);
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
        let playing = self.animating
            || self.fly.active()
            || self.scene.animators.iter().any(|a| a.playing);
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
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.as_ref().map(|w| w.inner_size());
                if let Some(size) = size {
                    self.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if down && !self.input.mouse_down {
                            self.input.mouse_pressed = true;
                        }
                        if !down && self.input.mouse_down {
                            self.input.mouse_released = true;
                        }
                        self.input.mouse_down = down;
                    }
                    MouseButton::Right => {
                        if down && !self.input.mouse_right_down {
                            self.input.mouse_right_pressed = true;
                        }
                        if !down && self.input.mouse_right_down {
                            self.input.mouse_right_released = true;
                        }
                        self.input.mouse_right_down = down;

                        // Look only when UI is not eating the mouse (viewport / empty chrome).
                        if down && !self.want_capture_mouse && !self.fly.looking {
                            self.set_looking(true);
                        } else if !down {
                            self.set_looking(false);
                        }
                    }
                    _ => {}
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => Vec2::new(x * 40.0, y * 40.0),
                    MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x as f32, p.y as f32),
                };
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.input.modifiers = mods.state();
                self.input.key_shift = mods.state().shift_key();
                self.input.key_ctrl = mods.state().control_key();
            }
            WindowEvent::Focused(false) => self.set_looking(false),
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    if !self.want_capture_keyboard || self.fly.looking {
                        self.fly.set_key(code, down);
                    } else if !down {
                        // Always release movement keys so they don't stick.
                        self.fly.set_key(code, false);
                    }
                    if down && code == KeyCode::Escape {
                        if self.fly.looking {
                            self.set_looking(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                }

                if down {
                    let shortcut = self.input.shortcut_mod();
                    match &event.logical_key {
                        Key::Named(NamedKey::Backspace) => self.input.key_backspace = true,
                        Key::Named(NamedKey::Enter) => self.input.key_enter = true,
                        Key::Named(NamedKey::ArrowLeft) => self.input.key_left = true,
                        Key::Named(NamedKey::ArrowRight) => self.input.key_right = true,
                        Key::Named(NamedKey::ArrowUp) => self.input.key_up = true,
                        Key::Named(NamedKey::ArrowDown) => self.input.key_down = true,
                        Key::Named(NamedKey::Home) => self.input.key_home = true,
                        Key::Named(NamedKey::End) => self.input.key_end = true,
                        Key::Character(c) if shortcut => match c.to_lowercase().as_str() {
                            "c" => self.input.key_copy = true,
                            "v" => self.begin_paste(),
                            "x" => self.input.key_cut = true,
                            "a" => self.input.key_select_all = true,
                            _ => {}
                        },
                        _ => {}
                    }
                    if shortcut {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::KeyC => self.input.key_copy = true,
                                KeyCode::KeyV => self.begin_paste(),
                                KeyCode::KeyX => self.input.key_cut = true,
                                KeyCode::KeyA => self.input.key_select_all = true,
                                _ => {}
                            }
                        }
                    }
                    if !shortcut {
                        if let Some(text) = event.text.as_ref() {
                            for ch in text.chars() {
                                if !ch.is_control() {
                                    self.input.text.push(ch);
                                }
                            }
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                self.input.text.push_str(&text);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn map_cursor(icon: CursorIcon) -> winit::window::CursorIcon {
    match icon {
        CursorIcon::Default => winit::window::CursorIcon::Default,
        CursorIcon::Pointer => winit::window::CursorIcon::Pointer,
        CursorIcon::Move => winit::window::CursorIcon::Move,
        CursorIcon::ResizeNwse => winit::window::CursorIcon::NwseResize,
        CursorIcon::ResizeEw => winit::window::CursorIcon::EwResize,
        CursorIcon::ResizeNs => winit::window::CursorIcon::NsResize,
        CursorIcon::Text => winit::window::CursorIcon::Text,
    }
}
