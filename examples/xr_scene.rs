//! Stage 2: a real scene (ground plane + cubes + spheres) rendered in VR via
//! `WgpuVisualizer`, with flight locomotion on the controller thumbstick —
//! push forward to fly wherever the headset is looking, pull back to reverse.
//!
//! Architecture: [`mega_render::xr::XrWgpu`] bridges the OpenXR-owned Vulkan
//! device into a real `wgpu::Device`/`Queue` (via `wgpu-hal`), so the exact
//! same [`WgpuVisualizer`] used for desktop rendering draws each eye — no
//! separate VR render path to maintain. Each OpenXR swapchain image (a
//! 2-layer texture array) is wrapped once as a `wgpu::Texture`; we render
//! into it twice per frame, once per eye, with an asymmetric per-eye
//! projection built from the FOV OpenXR reports.
//!
//! ```text
//! cargo run --example xr_scene --features xr
//! ```
//!
//! Ctrl-C to request a clean exit.

use glam::{Quat, Vec2, Vec3};
use mega_render::xr::{wrap_swapchain_images, Hand, XrActions, XrContext, XrPollResult, XrWgpu, XR_COLOR_FORMAT_VK};
use mega_render::{cube, plane, sphere, Camera, Light, Material, Node, Scene, Transform, Visualizer, WgpuVisualizer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Flight speed in meters/second at full stick deflection.
const FLY_SPEED: f32 = 3.0;

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.ambient = [0.05, 0.05, 0.06];
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        d.intensity = 3.0;
        d.direction = Vec3::new(0.3, -0.6, 0.5);
    }

    let mesh_plane = scene.meshes.insert(plane(30.0, 30.0));
    let mesh_cube = scene.meshes.insert(cube(0.6));
    let mesh_sphere = scene.meshes.insert(sphere(0.35, 32, 20));

    let mat_ground = scene
        .materials
        .insert(Material::new([0.5, 0.55, 0.6, 1.0], 0.0, 0.8));

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

    // A ring of cubes and spheres around the origin so there's always something
    // to fly toward regardless of which way you start facing.
    let n = 10;
    for i in 0..n {
        let a = (i as f32 / n as f32) * std::f32::consts::TAU;
        let r = 4.0;
        let pos = Vec3::new(a.cos() * r, 0.3, a.sin() * r);
        let hue = i as f32 / n as f32;
        let color = hsv_to_rgb(hue, 0.65, 0.9);

        if i % 2 == 0 {
            let mat = scene
                .materials
                .insert(Material::new([color[0], color[1], color[2], 1.0], 0.1, 0.4));
            scene.nodes.insert(Node {
                name: format!("cube_{i}"),
                parent: Some(root),
                local: Transform::from_translation(pos),
                mesh: Some(mesh_cube),
                material: Some(mat),
                skin: None,
                visible: true,
            });
        } else {
            let mat = scene
                .materials
                .insert(Material::new([color[0], color[1], color[2], 1.0], 0.8, 0.25));
            scene.nodes.insert(Node {
                name: format!("sphere_{i}"),
                parent: Some(root),
                local: Transform::from_translation(pos + Vec3::Y * 0.05),
                mesh: Some(mesh_sphere),
                material: Some(mat),
                skin: None,
                visible: true,
            });
        }
    }

    // A tall marker at the origin (stage center) so orientation is obvious.
    let mat_marker = scene
        .materials
        .insert(Material::new([0.9, 0.85, 0.2, 1.0], 0.0, 0.5));
    scene.nodes.insert(Node {
        name: "marker".into(),
        parent: Some(root),
        local: Transform {
            translation: Vec3::new(0.0, 1.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(0.3, 2.0, 0.3),
        },
        mesh: Some(mesh_cube),
        material: Some(mat_marker),
        skin: None,
        visible: true,
    });

    scene
}

/// Converts a vector (position or direction) from OpenXR's right-handed,
/// `-Z`-forward convention into this engine's left-handed, `+Z`-forward world
/// convention (see `Camera::orbit`'s "LH: X+ right, Y+ up, Z+ forward" and
/// `glam::camera::lh`, which expects world input in that convention).
///
/// This is a mirror across the XY-plane (negate Z) — the standard boundary
/// conversion between a RH and LH coordinate system that share the same X
/// (right) and Y (up) axes. Skipping this for OpenXR poses mirrors each eye's
/// view horizontally, which is invisible per-eye but reverses the *sense* of
/// stereo parallax between the two eyes (pseudostereo: no depth fusion,
/// "doubling"), and also reverses the perceived yaw/strafe direction.
fn to_engine(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, -v.z)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || r.store(false, Ordering::Relaxed))
        .expect("setting Ctrl-C handler");

    let mut xr = XrContext::new("mega-render xr_scene").expect("XrContext::new");
    // SAFETY: `xr_wgpu` does not outlive `xr` — both are dropped together at
    // the end of `main`, and nothing else destroys the underlying Vulkan
    // instance/device in the meantime.
    let xr_wgpu = unsafe { XrWgpu::from_xr(&xr) }.expect("XrWgpu::from_xr");
    let actions = XrActions::new(&xr.xr_instance, xr.session()).expect("XrActions::new");

    let mut visualizer = WgpuVisualizer::new(&xr_wgpu.device, &xr_wgpu.queue);
    let mut scene = build_scene();
    visualizer.sync(&scene);

    let mut swapchain: Option<(mega_render::xr::XrSwapchain, Vec<mega_render::xr::XrSwapchainImage>)> = None;

    // Player position offset in stage space — this is what "flying" moves.
    let mut player_pos = Vec3::ZERO;
    let mut last_frame = Instant::now();

    println!("xr_scene: waiting for session to become ready (put the headset on)...");

    'main: loop {
        if !running.load(Ordering::Relaxed) {
            println!("xr_scene: requesting exit");
            xr.request_exit();
        }

        match xr.poll_events() {
            XrPollResult::Continue => {}
            XrPollResult::Exit => break 'main,
        }

        if !xr.session_running() {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        if let Err(e) = actions.sync(xr.session()) {
            eprintln!("xr_scene: action sync failed: {e}");
        }

        let Some(frame) = xr.wait_frame().expect("wait_frame") else {
            continue;
        };

        let (sc_xr, sc_images) = swapchain.get_or_insert_with(|| {
            let resolution = xr.recommended_resolution().expect("recommended_resolution");
            let sc = xr
                .create_swapchain(resolution, XR_COLOR_FORMAT_VK)
                .expect("create_swapchain");
            let images = unsafe { wrap_swapchain_images(&xr_wgpu, &sc.images, resolution) };
            (sc, images)
        });

        let image_index = sc_xr.handle.acquire_image().expect("acquire_image") as usize;
        sc_xr
            .handle
            .wait_image(openxr::Duration::INFINITE)
            .expect("wait_image");

        let views = xr.locate_views(&frame).expect("locate_views");

        // --- Locomotion: fly wherever the headset is looking. ---
        let now = Instant::now();
        let dt = (now - last_frame).as_secs_f32().clamp(0.0, 0.1);
        last_frame = now;
        if let Some(head) = views.first() {
            let head_rot = Quat::from_xyzw(
                head.pose.orientation.x,
                head.pose.orientation.y,
                head.pose.orientation.z,
                head.pose.orientation.w,
            );
            let forward = to_engine(head_rot * Vec3::NEG_Z);
            let right = to_engine(head_rot * Vec3::X);

            let stick = actions.stick(xr.session(), Hand::Left) + actions.stick(xr.session(), Hand::Right);
            let stick: Vec2 = Vec2::new(stick.x.clamp(-1.0, 1.0), stick.y.clamp(-1.0, 1.0));
            if stick.length_squared() > 1e-4 {
                let dir = (forward * stick.y + right * stick.x).normalize_or_zero();
                player_pos += dir * FLY_SPEED * stick.length().min(1.0) * dt;
            }
        }

        visualizer.ensure_target(sc_xr.resolution.width, sc_xr.resolution.height);
        let aspect = sc_xr.resolution.width as f32 / sc_xr.resolution.height.max(1) as f32;

        for eye in 0..mega_render::xr::VIEW_COUNT as usize {
            let Some(view) = views.get(eye) else { continue };
            let eye_pos = to_engine(Vec3::new(
                view.pose.position.x,
                view.pose.position.y,
                view.pose.position.z,
            )) + player_pos;
            let eye_rot = Quat::from_xyzw(
                view.pose.orientation.x,
                view.pose.orientation.y,
                view.pose.orientation.z,
                view.pose.orientation.w,
            );
            let forward = to_engine(eye_rot * Vec3::NEG_Z).normalize_or_zero();
            let up = to_engine(eye_rot * Vec3::Y).normalize_or_zero();

            let near = 0.05;
            let far = 100.0;
            scene.camera.eye = eye_pos;
            scene.camera.near = near;
            scene.camera.far = far;
            scene.camera.xr_view = Some(glam::camera::lh::view::look_to_mat4(eye_pos, forward, up));
            scene.camera.xr_proj = Some(Camera::asymmetric_perspective(
                view.fov.angle_left,
                view.fov.angle_right,
                view.fov.angle_up,
                view.fov.angle_down,
                near,
                far,
            ));
            visualizer.render_to(&scene, aspect, &sc_images[image_index].eye_views[eye]);
        }

        sc_xr.handle.release_image().expect("release_image");
        xr.end_frame(frame, &views, sc_xr).expect("end_frame");
    }

    // Explicit teardown order: `xr_wgpu` owns destroying the Vulkan
    // instance/device it wraps (see `XrWgpu::from_xr`'s safety doc), but the
    // OpenXR session/instance were built *from* that same instance/device and
    // must be torn down first — otherwise `xrDestroySession` runs against an
    // already-destroyed `VkDevice` and crashes. Rust's default (reverse
    // declaration order) drop would do this backwards since `xr_wgpu` was
    // declared after `xr`, so tear everything down by hand instead.
    drop(swapchain);
    drop(visualizer);
    drop(actions);
    drop(xr);
    drop(xr_wgpu);

    println!("xr_scene: exited cleanly");
}
