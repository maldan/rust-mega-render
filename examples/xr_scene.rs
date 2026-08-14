//! Stage 2: a real scene (ground plane + cubes + spheres) rendered in VR via
//! `WgpuVisualizer`, with two-stick flight controls: the left thumbstick
//! flies wherever the headset is looking (push forward to go, pull back to
//! reverse, strafe left/right), and the right thumbstick smooth-turns the
//! view (yaw left/right, pitch up/down) beyond what the physical headset
//! alone can look.
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
use mega_render::{
    cube, plane, sphere, Camera, DirectionalLight, Handle, Light, Material, Node, Scene,
    Transform, Visualizer, WgpuVisualizer,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Flight speed in meters/second at full stick deflection.
const FLY_SPEED: f32 = 3.0;

/// Smooth-turn speed in radians/second at full stick deflection (right hand).
const TURN_SPEED: f32 = 1.8;

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.ambient = [0.05, 0.05, 0.06];
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        d.intensity = 3.0;
        d.direction = Vec3::new(0.3, -0.6, 0.5);
    }
    // Shadowless fill light from behind and above (physical "forward" maps
    // to engine `+Z` here, so "behind" is `-Z`; the direction below is where
    // the light *travels*, i.e. positive Z/negative Y = from behind-above
    // toward the scene) — softens the backs of objects the main shadow-caster
    // leaves dark, without a second shadow map.
    scene.lights.push(Light::Directional(DirectionalLight {
        direction: Vec3::new(-0.15, 0.55, -0.8).normalize(),
        color: [0.65, 0.72, 0.85],
        intensity: 1.4,
        enabled: true,
        cast_shadows: false,
    }));

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

    // A row of glTF models beyond the cube/sphere ring, so they're right in
    // front of you when you first put the headset on — physical "forward"
    // maps to engine `+Z` here (see `to_engine`'s doc comment below), so
    // these sit at positive Z, not negative.
    let models = [
        (r"F:\3d\tripo_ai\boa.glb", Vec3::new(-4.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\bulma_full.glb", Vec3::new(-3.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\cammy_full.glb", Vec3::new(-1.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\chunli_full.glb", Vec3::new(0.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\zangya_full.glb", Vec3::new(1.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\rangiku_full.glb", Vec3::new(3.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\2b.glb", Vec3::new(4.5, 0.0, 8.0)),
    ];
    for &(path, offset) in &models {
        // `load_gltf_async` spawns a worker thread and only ever reports
        // failure through `Scene::poll_loads`'s `eprintln` (never panics),
        // but checking up front avoids spawning a thread and logging an
        // error for models that simply aren't present on this machine
        // (these are local test assets, not shipped with the repo).
        if !std::path::Path::new(path).is_file() {
            eprintln!("xr_scene: skipping missing model {path}");
            continue;
        }
        scene.load_gltf_async(path, Some(root), move |scene, h| {
            if let Some(n) = scene.nodes.get_mut(h) {
                // These positions are in the desktop/glTF `-Z`-away
                // convention already shared by the rest of this scene, so no
                // `to_engine` conversion is needed here (that only applies
                // to live OpenXR poses, not authored scene content).
                n.local.translation += offset;
            }
            enable_sss_on_subtree(scene, h);
        });
    }

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

    // The "rig" transform: everything the right-hand stick (look) and
    // left-hand stick (fly) control, applied on top of the raw headset
    // tracking. `rig_rot` composes with the headset's own orientation so you
    // can look further left/right/up/down than the physical headset alone
    // allows; `rig_pos` is the flight offset, expressed in the *rig's*
    // rotated frame so "forward" always means "wherever the headset is
    // currently looking, including the rig's own turn".
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

        // Apply any glTF loads kicked off by `build_scene`'s
        // `load_gltf_async` calls once they finish on their background
        // thread, then re-sync the GPU-side scene so the visualizer picks up
        // the newly-added nodes/meshes/materials.
        scene.poll_loads();
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

        // --- Right stick: smooth-turn (yaw + pitch), pivoting around the
        // head so turning in place doesn't fling you around the room. ---
        // --- Left stick: fly wherever the (rig-turned) headset is looking. ---
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
            ));

            let turn = actions.stick(xr.session(), Hand::Right);
            let d_yaw = turn.x.clamp(-1.0, 1.0) * TURN_SPEED * dt;
            let d_pitch = -turn.y.clamp(-1.0, 1.0) * TURN_SPEED * dt;
            if d_yaw.abs() > 1e-6 || d_pitch.abs() > 1e-6 {
                // World-space rotation delta this frame: yaw around the
                // global up axis, then pitch around the rig's (post-yaw)
                // right axis, so "look up/down" always tilts relative to
                // where you're currently facing.
                let yaw_delta = Quat::from_axis_angle(Vec3::Y, d_yaw);
                let right_axis = (yaw_delta * rig_rot) * Vec3::X;
                let pitch_delta = Quat::from_axis_angle(right_axis, d_pitch);
                let turn_delta = pitch_delta * yaw_delta;

                // Rotate in place around the head's current world position,
                // not the room's tracking origin — otherwise turning would
                // swing your position around the play space center.
                let head_world_before = rig_rot * head_pos + rig_pos;
                rig_pos = head_world_before - turn_delta * (head_world_before - rig_pos);
                rig_rot = (turn_delta * rig_rot).normalize();
            }

            let forward = rig_rot * to_engine(head_rot * Vec3::NEG_Z);
            let right = rig_rot * to_engine(head_rot * Vec3::X);

            let fly = actions.stick(xr.session(), Hand::Left);
            let fly: Vec2 = Vec2::new(fly.x.clamp(-1.0, 1.0), fly.y.clamp(-1.0, 1.0));
            if fly.length_squared() > 1e-4 {
                let dir = (forward * fly.y + right * fly.x).normalize_or_zero();
                rig_pos += dir * FLY_SPEED * fly.length().min(1.0) * dt;
            }
        }

        visualizer.ensure_target(sc_xr.resolution.width, sc_xr.resolution.height);
        let aspect = sc_xr.resolution.width as f32 / sc_xr.resolution.height.max(1) as f32;

        for eye in 0..mega_render::xr::VIEW_COUNT as usize {
            let Some(view) = views.get(eye) else { continue };
            let raw_eye_pos = to_engine(Vec3::new(
                view.pose.position.x,
                view.pose.position.y,
                view.pose.position.z,
            ));
            let eye_rot = Quat::from_xyzw(
                view.pose.orientation.x,
                view.pose.orientation.y,
                view.pose.orientation.z,
                view.pose.orientation.w,
            );
            // Apply the rig's turn/fly transform on top of the raw headset
            // pose, same as the locomotion step above.
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
