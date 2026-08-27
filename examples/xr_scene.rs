//! VR scene: ground plane + a ring of cubes/spheres, rendered through the
//! same [`WgpuVisualizer`] as desktop via [`mega_render::xr::XrWgpu`].
//!
//! Left stick flies along the headset look direction; right stick yaw-turns
//! in place. Ctrl-C requests a clean exit.
//!
//! ```text
//! cargo run --example xr_scene --features xr
//! ```

use glam::{Quat, Vec2, Vec3};
use mega_render::xr::{
    wrap_swapchain_images, Hand, HandPose, XrActions, XrContext, XrPollResult, XrWgpu,
    XR_COLOR_FORMAT_VK,
};
use mega_render::{
    cube, plane, sphere, Camera, DirectionalLight, Light, LineOpts, Material, Node, PointLight,
    Scene, Transform, Visualizer, WgpuVisualizer,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Flight speed in meters/second at full stick deflection.
const FLY_SPEED: f32 = 3.0;

/// Smooth-turn speed in radians/second at full stick deflection (right hand).
const TURN_SPEED: f32 = 1.8;

/// Multiplies real head/eye *translation* (never mesh geometry). `1.0` = a
/// `cube(1.0)` is a literal 1‑meter cube.
const WORLD_SCALE: f32 = 1.0;

/// Length in meters of the debug aiming ray drawn from each controller.
const HAND_RAY_LENGTH: f32 = 5.0;

fn build_scene() -> (Scene, usize) {
    let mut scene = Scene::new();
    scene.ambient = [0.05, 0.05, 0.06];
    scene.clear_color = [0.0, 0.0, 0.0, 1.0];
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        d.intensity = 0.1;
        d.direction = Vec3::new(0.3, -0.6, 0.5);
    }
    scene.lights.push(Light::Directional(DirectionalLight {
        direction: Vec3::new(-0.15, 0.55, -0.8).normalize(),
        color: [0.65, 0.72, 0.85],
        intensity: 0.2,
        enabled: true,
        cast_shadows: false,
    }));

    let player_light_idx = scene.lights.len();
    scene.lights.push(Light::Point(PointLight {
        position: Vec3::ZERO,
        color: [1.0, 0.92, 0.8],
        intensity: 2.5,
        range: 8.0,
        enabled: true,
    }));

    let mesh_plane = scene.meshes.insert(plane(30.0, 30.0));
    let mesh_cube = scene.meshes.insert(cube(0.6));
    let mesh_sphere = scene.meshes.insert(sphere(0.35, 32, 20));

    let mat_ground = scene
        .materials
        .insert(Material::new([0.5, 0.55, 0.6, 1.0], 0.0, 0.8));

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
        name: "ground".into(),
        parent: Some(root),
        local: Transform::default(),
        mesh: Some(mesh_plane),
        material: Some(mat_ground),
        skin: None,
        visible: true,
    });

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
                id: Node::new_id(),
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
                id: Node::new_id(),
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

    let mat_marker = scene
        .materials
        .insert(Material::new([0.9, 0.85, 0.2, 1.0], 0.0, 0.5));
    scene.nodes.insert(Node {
        id: Node::new_id(),
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

    (scene, player_light_idx)
}

/// OpenXR RH `-Z`-forward → engine LH `+Z`-forward (mirror across XY).
fn to_engine(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, -v.z)
}

fn aim_ray_world(pose: HandPose, rig_rot: Quat, rig_pos: Vec3) -> (Vec3, Vec3) {
    let world_pos = rig_rot * (to_engine(pose.position) * WORLD_SCALE) + rig_pos;
    let dir = (rig_rot * to_engine(pose.orientation * Vec3::NEG_Z)).normalize_or_zero();
    (world_pos, dir)
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
        4 => [t, p, q],
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

    let mut visualizer = WgpuVisualizer::new(
        &xr_wgpu.device,
        &xr_wgpu.queue,
        mega_render::xr::XR_COLOR_FORMAT_WGPU,
    );
    {
        let post = visualizer.post_process();
        post.env.enabled = false;
        post.ao.enabled = true;
        post.contact_shadow.enabled = true;
        post.ssgi.enabled = true;
    }
    let (mut scene, player_light_idx) = build_scene();
    visualizer.sync(&scene);

    let mut swapchain: Option<(mega_render::xr::XrSwapchain, Vec<mega_render::xr::XrSwapchainImage>)> =
        None;

    let mut rig_yaw = 0.0f32;
    let mut rig_rot = Quat::IDENTITY;
    let mut rig_pos = Vec3::ZERO;
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

        visualizer.sync(&scene);

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
            let head_pos = to_engine(Vec3::new(
                head.pose.position.x,
                head.pose.position.y,
                head.pose.position.z,
            )) * WORLD_SCALE;

            let turn = actions.stick(xr.session(), Hand::Right);
            let d_yaw = turn.x.clamp(-1.0, 1.0) * TURN_SPEED * dt;
            if d_yaw.abs() > 1e-6 {
                let rig_rot_before = rig_rot;
                rig_yaw += d_yaw;
                rig_rot = Quat::from_axis_angle(Vec3::Y, rig_yaw);
                let turn_delta = rig_rot * rig_rot_before.inverse();
                let head_world_before = rig_rot_before * head_pos + rig_pos;
                rig_pos = head_world_before - turn_delta * (head_world_before - rig_pos);
            }

            let forward = rig_rot * to_engine(head_rot * Vec3::NEG_Z);
            let right = rig_rot * to_engine(head_rot * Vec3::X);

            let fly = actions.stick(xr.session(), Hand::Left);
            let fly: Vec2 = Vec2::new(fly.x.clamp(-1.0, 1.0), fly.y.clamp(-1.0, 1.0));
            if fly.length_squared() > 1e-4 {
                let dir = (forward * fly.y + right * fly.x).normalize_or_zero();
                rig_pos += dir * FLY_SPEED * fly.length().min(1.0) * dt;
            }

            let head_world = rig_rot * head_pos + rig_pos;
            if let Some(Light::Point(p)) = scene.lights.get_mut(player_light_idx) {
                p.position = head_world;
            }
        }

        let hand_colors = [[0.2, 0.9, 1.0, 0.9], [1.0, 0.35, 0.85, 0.9]];
        scene.debug.clear();
        for (slot, hand) in [Hand::Left, Hand::Right].into_iter().enumerate() {
            let Some(aim_pose) =
                actions.aim_pose(xr.stage(), hand, frame.state.predicted_display_time)
            else {
                continue;
            };
            let (origin, dir) = aim_ray_world(aim_pose, rig_rot, rig_pos);
            let color = hand_colors[slot];
            scene
                .debug
                .line(origin, origin + dir * HAND_RAY_LENGTH, LineOpts::color(color).width(3.0));
            scene.debug.point_sized(origin, color, 10.0);
        }

        visualizer.ensure_target(sc_xr.resolution.width, sc_xr.resolution.height);
        let aspect = sc_xr.resolution.width as f32 / sc_xr.resolution.height.max(1) as f32;

        for eye in 0..mega_render::xr::VIEW_COUNT as usize {
            let Some(view) = views.get(eye) else { continue };
            let raw_eye_pos = to_engine(Vec3::new(
                view.pose.position.x,
                view.pose.position.y,
                view.pose.position.z,
            )) * WORLD_SCALE;
            let eye_rot = Quat::from_xyzw(
                view.pose.orientation.x,
                view.pose.orientation.y,
                view.pose.orientation.z,
                view.pose.orientation.w,
            );
            let eye_pos = rig_rot * raw_eye_pos + rig_pos;
            let forward = (rig_rot * to_engine(eye_rot * Vec3::NEG_Z)).normalize_or_zero();
            let up = (rig_rot * to_engine(eye_rot * Vec3::Y)).normalize_or_zero();

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

    // Tear down OpenXR before `XrWgpu` destroys the Vulkan device.
    drop(swapchain);
    drop(visualizer);
    drop(actions);
    drop(xr);
    drop(xr_wgpu);

    println!("xr_scene: exited cleanly");
}
