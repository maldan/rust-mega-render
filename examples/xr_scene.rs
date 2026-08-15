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
use mega_render::xr::{
    wrap_swapchain_images, Hand, HandPose, XrActions, XrContext, XrPollResult, XrWgpu,
    XR_COLOR_FORMAT_VK,
};
use mega_render::{
    cube, plane, sphere, Camera, DirectionalLight, Handle, Light, LineOpts, Material, Node,
    PointLight, Scene, Transform, Visualizer, WgpuVisualizer,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[path = "xr_scene/soft_chain.rs"]
mod soft_chain;
#[path = "xr_scene/root_grab.rs"]
mod root_grab;

/// Flight speed in meters/second at full stick deflection.
const FLY_SPEED: f32 = 3.0;

/// Smooth-turn speed in radians/second at full stick deflection (right hand).
const TURN_SPEED: f32 = 1.8;

/// World scale: multiplies real head/eye *translation* (never mesh geometry)
/// before it's combined with the rig transform. OpenXR always reports poses
/// in real meters, and nothing else in this pipeline scales position, so at
/// `1.0` a `cube(1.0)` is a literal 1‑meter cube — matching real life. This
/// is the standard "shrink ray / giant mode" VR trick: scaling head/eye
/// translation scales both the inter-eye stereo baseline (parallax) and
/// head-motion parallax together, which is what actually drives perceived
/// size — so it changes how big the world *feels* without touching a single
/// vertex. `> 1.0` = you feel smaller (world feels bigger, farther things
/// move less per real step); `< 1.0` = you feel bigger (world feels
/// smaller). Leave at `1.0` for a real-world-accurate scale.
const WORLD_SCALE: f32 = 1.0;

/// Length in meters of the debug aiming ray drawn from each controller.
const HAND_RAY_LENGTH: f32 = 5.0;

/// Extra reach-and-touch margin beyond a jiggle-chain's tip capsule surface
/// (see `soft_chain::pick_chain_tip`), in meters, for picking it up with the
/// controller's grip position.
const GRAB_TOUCH_MARGIN: f32 = 0.05;
/// Radius (meters) of the capsule grab collider wrapped around each
/// jiggle-chain's tip bone (e.g. `Nipple.L`/`Nipple.R`) — bigger than the
/// bone itself so both the touch and aim-ray grab tests are forgiving.
const GRAB_COLLIDER_RADIUS: f32 = 0.06;
/// Trigger pull (0..1) that starts a grab.
const GRAB_PRESS: f32 = 0.6;
/// Trigger release threshold (lower than `GRAB_PRESS` — hysteresis avoids flicker).
const GRAB_RELEASE: f32 = 0.4;

/// Size (meters) of the little cube marker drawn at each loaded model's
/// root node — grab this to move/rotate the whole model rigidly (see
/// `root_grab`), as opposed to grabbing a jiggle-chain tip which only
/// tugs on the soft-body simulation.
const HANDLE_CUBE_SIZE: f32 = 0.05;
/// Hit-test radius (meters) around a root handle cube — bigger than the
/// cube itself so both the touch and aim-ray grab tests are forgiving.
const HANDLE_GRAB_RADIUS: f32 = 0.09;

/// Returns the scene, the index into `scene.lights` of the point light that
/// follows the player's head each frame (see the `'main` loop, which
/// rewrites its `position` every frame instead of parenting it to a node —
/// lights aren't scene-graph nodes here, just a flat `Vec<Light>`), and a
/// shared queue of glTF model root handles: each `models` entry below loads
/// asynchronously, so this is how the `'main` loop finds out (once per
/// model, as each one finishes loading) which subtree to scan for jiggle
/// bones — see `create_breast_chains`'s doc comment for why this has to be
/// per-model instead of a single scene-wide, "stop after the first hit" scan.
fn build_scene() -> (Scene, usize, Arc<Mutex<Vec<Handle<Node>>>>) {
    let mut scene = Scene::new();
    scene.ambient = [0.05, 0.05, 0.06];
    scene.clear_color = [0.0, 0.0, 0.0, 1.0];
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        d.intensity = 0.1;
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
        intensity: 0.2,
        enabled: true,
        cast_shadows: false,
    }));

    // Personal "torch" light that follows the headset around (position is
    // overwritten every frame from the live head pose) — lights the
    // immediate surroundings wherever you look, independent of the fixed
    // directional lights above.
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
    let mesh_handle = scene.meshes.insert(cube(HANDLE_CUBE_SIZE));

    let mat_ground = scene
        .materials
        .insert(Material::new([0.5, 0.55, 0.6, 1.0], 0.0, 0.8));
    // Bright, unmistakable marker color for the whole-model root grab handle.
    let mat_handle = scene
        .materials
        .insert(Material::new([1.0, 0.85, 0.1, 1.0], 0.0, 0.4));

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
        /*(r"F:\3d\tripo_ai\boa.glb", Vec3::new(-4.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\bulma_full.glb", Vec3::new(-3.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\cammy_full.glb", Vec3::new(-1.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\chunli_full.glb", Vec3::new(0.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\zangya_full.glb", Vec3::new(1.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\rangiku_full.glb", Vec3::new(3.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\2b.glb", Vec3::new(4.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\satsuki_full.glb", Vec3::new(5.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\tatsumaki_full.glb", Vec3::new(5.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\tifa.glb", Vec3::new(6.0, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\lucoa_full.glb", Vec3::new(6.5, 0.0, 8.0)),
        (r"F:\3d\tripo_ai\robin_full.glb", Vec3::new(7.0, 0.0, 8.0)),*/
        //(r"F:\3d\tripo_ai\tier_full.glb", Vec3::new(7.5, 0.0, 8.0)),
        (r"F:/csharp/VR_Waifu/asset/model/Bulma/Bulma.glb", Vec3::new(2.5, 0.0, -4.5)),
        (r"F:/csharp/VR_Waifu/asset/model/Miruko/Miruko.gltf", Vec3::new(3.0, 0.0, -4.5)),
        (r"F:\csharp\VR_Waifu\asset\model\Tsunade\Tsunade.gltf", Vec3::new(3.5, 0.0, -4.5)),
        (r"F:\csharp\VR_Waifu\asset\model\Pony.glb", Vec3::new(4.0, 0.0, -4.5)),
        (r"F:\csharp\VR_Waifu\asset\model\Zangya\Zangya2.gltf", Vec3::new(4.5, 0.0, -4.5)),
    
    ];
    let model_roots: Arc<Mutex<Vec<Handle<Node>>>> = Arc::new(Mutex::new(Vec::new()));
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
        let model_roots = model_roots.clone();
        scene.load_gltf_async(path, Some(root), move |scene, h| {
            if let Some(n) = scene.nodes.get_mut(h) {
                // These positions are in the desktop/glTF `-Z`-away
                // convention already shared by the rest of this scene, so no
                // `to_engine` conversion is needed here (that only applies
                // to live OpenXR poses, not authored scene content).
                n.local.translation += offset;
            }
            enable_sss_on_subtree(scene, h);
            // Small marker cube, parented to the model root itself (local
            // transform stays identity), so grabbing it drags the whole
            // model rigidly — see `root_grab`.
            scene.nodes.insert(Node {
                name: "RootGrabHandle".into(),
                parent: Some(h),
                local: Transform::default(),
                mesh: Some(mesh_handle),
                material: Some(mat_handle),
                skin: None,
                visible: true,
            });
            // `absorb_gltf` (called just before this closure runs, from
            // `Scene::poll_loads`) has already merged this model's full
            // node/skeleton subtree into `scene`, so it's safe for the
            // `'main` loop to scan under `h` for jiggle bones immediately.
            model_roots.lock().unwrap().push(h);
        });
    }

    (scene, player_light_idx, model_roots)
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
            mat.sss_strength = 0.75;
            mat.sss_color = [1.0, 0.32, 0.18];
            mat.sss_curvature = 0.3;
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

/// Turns a controller's aim pose into a world-space ray origin/direction,
/// run through the same `to_engine` + rig (turn/fly) transform chain as the
/// head/eyes each frame. Uses the *aim* pose (not grip pose) — aim's local
/// `-Z` is the OpenXR-defined "laser pointer" direction, whereas grip's `-Z`
/// points out of the palm and is meant for rendering a hand/controller model
/// instead — so the ray lines up with whatever the controller is actually
/// aimed at in the rendered world. Used both for the debug laser line and
/// for the aim-ray grab test (see `soft_chain::pick_chain_tip`).
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
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// True if `name` (already lowercased) contains `keyword` and ends in a
/// `.L`/`_L`/`L` (or `.R`/`_R`/`R`) side suffix, e.g. `"boob.l"`,
/// `"boob_r"` — matches the Blender-style naming convention used by the
/// Miruko rig (see `model-rig/scripts/miruko.lua`'s `bone("Boob", side)`).
fn bone_name_matches(name: &str, keyword: &str, side_lc: char) -> bool {
    if !name.contains(keyword) {
        return false;
    }
    let bytes = name.as_bytes();
    let Some(&last) = bytes.last() else { return false };
    if last as char != side_lc {
        return false;
    }
    bytes.len() == 1 || matches!(bytes[bytes.len() - 2] as char, '.' | '_' | '-' | ' ')
}

/// Finds a bone matching [`bone_name_matches`] within `root`'s subtree only
/// — scoped per model (see `create_jiggle_chains`'s doc comment for why a
/// scene-wide search doesn't work once more than one model is loaded).
fn find_side_bone_under(scene: &Scene, root: Handle<Node>, keyword: &str, side: char) -> Option<Handle<Node>> {
    let side_lc = side.to_ascii_lowercase();
    let mut stack = vec![root];
    let mut hit = None;
    while let Some(h) = stack.pop() {
        if let Some(n) = scene.nodes.get(h) {
            if bone_name_matches(&n.name.to_lowercase(), keyword, side_lc) {
                hit = Some(h);
                break;
            }
        }
        for (child, node) in scene.nodes.iter() {
            if node.parent.map(|p| p.key()) == Some(h.key()) {
                stack.push(child);
            }
        }
    }
    hit
}

/// Finds a bone whose (lowercased) name contains `keyword`, with no side
/// suffix requirement (unlike [`find_side_bone_under`]) — for bones that
/// don't come in `.L`/`.R` pairs, e.g. `Penis`.
fn find_bone_under(scene: &Scene, root: Handle<Node>, keyword: &str) -> Option<Handle<Node>> {
    let mut stack = vec![root];
    let mut hit = None;
    while let Some(h) = stack.pop() {
        if let Some(n) = scene.nodes.get(h) {
            if n.name.to_lowercase().contains(keyword) {
                hit = Some(h);
                break;
            }
        }
        for (child, node) in scene.nodes.iter() {
            if node.parent.map(|p| p.key()) == Some(h.key()) {
                stack.push(child);
            }
        }
    }
    hit
}

/// Builds jiggle chains for `model_root` (one loaded model's subtree, if
/// present in that loaded model) with the tuning used by the Miruko rig's
/// breast physics:
/// - `Boob.L`/`Boob.R` — grabbable, and pushes the other one apart
///   (`collide_with_others: true`).
/// - `Penis_1` (a 4-segment chain: `Penis_1` -> `Penis_2` -> `Penis_3` ->
///   `Penis_4`, auto-collected by `collect_descendant_chain`) — also
///   grabbable, but `collide_with_others: false` so it never
///   pushes/gets-pushed-by the breast capsules.
///
/// Scoped to one model's subtree — and called once per model root as each
/// glTF finishes loading (see `build_scene`'s `model_roots` queue) — rather
/// than a single scene-wide search, because a scene-wide search would find
/// whichever model's bones happen to be inserted first and never look again
/// once `soft_chains` is non-empty, silently ignoring every other loaded
/// model's bones.
fn create_jiggle_chains(scene: &Scene, model_root: Handle<Node>) -> Vec<soft_chain::SoftChain> {
    let mut chains = Vec::new();
    for (side, label) in [('l', "Boob.L"), ('r', "Boob.R")] {
        let Some(bone) = find_side_bone_under(scene, model_root, "boob", side) else {
            continue;
        };
        if let Some(chain) = soft_chain::create_soft_chain(
            scene,
            label,
            bone,
            soft_chain::SoftChainConfig::default(),
            Some(GRAB_COLLIDER_RADIUS),
            true,
        ) {
            println!("xr_scene: jiggle chain '{label}' created");
            chains.push(chain);
        }
    }
    if let Some(bone) = find_bone_under(scene, model_root, "penis") {
        if let Some(chain) = soft_chain::create_soft_chain(
            scene,
            "Penis",
            bone,
            soft_chain::SoftChainConfig::default(),
            Some(GRAB_COLLIDER_RADIUS),
            false,
        ) {
            println!("xr_scene: jiggle chain 'Penis' created (grabbable, no collision with other chains)");
            chains.push(chain);
        }
    }
    chains
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
    {
        // No env map is loaded for this scene, so keep the sky off (falls
        // back to `clear_color`, i.e. black). Only GTAO + contact shadows +
        // SSGI are under test right now — the rest of the post stack stays off.
        let post = visualizer.post_process();
        post.env.enabled = false;
        post.ao.enabled = true;
        post.contact_shadow.enabled = true;
        post.ssgi.enabled = true;
    }
    let (mut scene, player_light_idx, model_roots) = build_scene();
    visualizer.sync(&scene);

    // Breast jiggle chains (`Boob.L`/`Boob.R`), created per model as each
    // one's async glTF load lands in `scene.nodes` (see the `model_roots`
    // drain in the `'main` loop below). A model that has no matching bones
    // just contributes nothing here — every *other* loaded model is still
    // scanned independently.
    let mut soft_chains: Vec<soft_chain::SoftChain> = Vec::new();
    // Per-hand active grab: (chain index, particle index, how to keep
    // tracking it — see `soft_chain::GrabAnchor`), `None` when that hand
    // isn't currently holding anything.
    let mut grab_state: [Option<(usize, usize, soft_chain::GrabAnchor)>; 2] = [None, None];

    // Every loaded model's root node — grabbing its `RootGrabHandle` child
    // cube (see `build_scene`) rigidly drags/rotates the whole model (see
    // `root_grab`), as opposed to `soft_chains`' per-bone jiggle grab.
    let mut grab_roots: Vec<Handle<Node>> = Vec::new();
    // Per-hand active whole-model grab, `None` when that hand isn't
    // currently holding a root handle.
    let mut root_grab_state: [Option<root_grab::RootGrab>; 2] = [None, None];

    let mut swapchain: Option<(mega_render::xr::XrSwapchain, Vec<mega_render::xr::XrSwapchainImage>)> = None;

    // The "rig" transform: everything the right-hand stick (look) and
    // left-hand stick (fly) control, applied on top of the raw headset
    // tracking. `rig_rot` composes with the headset's own orientation so you
    // can look further left/right/up/down than the physical headset alone
    // allows; `rig_pos` is the flight offset, expressed in the *rig's*
    // rotated frame so "forward" always means "wherever the headset is
    // currently looking, including the rig's own turn". Yaw-only: the right
    // stick used to also smooth-pitch (up/down), but that's been dropped —
    // pitch now comes exclusively from the physical headset. Yaw is stored
    // as a plain angle (not accumulated as a rolling quaternion) so the
    // rig's horizon never drifts: composing incremental yaw quaternions
    // frame-by-frame (as this used to, back when pitch was also involved)
    // could silently accumulate roll since yaw/pitch rotations don't
    // commute. Rebuilding `rig_rot` from scratch each frame guarantees zero
    // roll and fully reversible turning.
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

        // Apply any glTF loads kicked off by `build_scene`'s
        // `load_gltf_async` calls once they finish on their background
        // thread, then re-sync the GPU-side scene so the visualizer picks up
        // the newly-added nodes/meshes/materials.
        scene.poll_loads();
        visualizer.sync(&scene);

        // Scan each model's subtree for `Boob.L`/`Boob.R` bones exactly once,
        // right after it finishes loading (`model_roots` is only ever
        // pushed to from `build_scene`'s `on_ready` callbacks — see there
        // for why this has to be per-model instead of a single scene-wide,
        // "stop after the first hit" scan).
        let new_roots: Vec<Handle<Node>> = std::mem::take(&mut *model_roots.lock().unwrap());
        for model_root in new_roots {
            soft_chains.extend(create_jiggle_chains(&scene, model_root));
            grab_roots.push(model_root);
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
            )) * WORLD_SCALE;

            let turn = actions.stick(xr.session(), Hand::Right);
            let d_yaw = turn.x.clamp(-1.0, 1.0) * TURN_SPEED * dt;
            if d_yaw.abs() > 1e-6 {
                let rig_rot_before = rig_rot;

                // Plain angle accumulation (no quaternion composed onto
                // itself frame after frame) — this is what keeps the rig
                // roll-free and makes the stick fully reversible. Yaw-only:
                // pitch is left at its initial identity value, so up/down on
                // the right stick no longer does anything.
                rig_yaw += d_yaw;
                rig_rot = Quat::from_axis_angle(Vec3::Y, rig_yaw);

                // Rotate in place around the head's current world position,
                // not the room's tracking origin — otherwise turning would
                // swing your position around the play space center.
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

            // Keep the player's torch light glued to the head, in world
            // space, after this frame's turn/fly update.
            let head_world = rig_rot * head_pos + rig_pos;
            if let Some(Light::Point(p)) = scene.lights.get_mut(player_light_idx) {
                p.position = head_world;
            }
        }

        // Controller grab: a jiggle-chain's tip capsule (see
        // `soft_chain::pick_chain_tip`) can be grabbed either by touching it
        // with the grip position, or by aiming the laser pointer (aim pose)
        // through it from a distance — so you don't have to fly the
        // controller right up against it. Hold the trigger to keep pulling.
        let hand_colors = [[0.2, 0.9, 1.0, 0.9], [1.0, 0.35, 0.85, 0.9]];
        let mut grab_pulls: Vec<soft_chain::GrabPull> = Vec::new();
        scene.debug.clear();
        for (slot, hand) in [Hand::Left, Hand::Right].into_iter().enumerate() {
            let Some(grip_pose) = actions.hand_pose(xr.stage(), hand, frame.state.predicted_display_time)
            else {
                continue;
            };
            let hand_world = rig_rot * (to_engine(grip_pose.position) * WORLD_SCALE) + rig_pos;
            let hand_rot = root_grab::hand_rot_world(to_engine, rig_rot, grip_pose.orientation);

            let aim_pose = actions.aim_pose(xr.stage(), hand, frame.state.predicted_display_time);
            let ray = aim_pose.map(|p| aim_ray_world(p, rig_rot, rig_pos));
            if let Some((origin, dir)) = ray {
                let end = origin + dir * HAND_RAY_LENGTH;
                let color = hand_colors[slot];
                scene
                    .debug
                    .line(origin, end, LineOpts::color(color).width(3.0));
                scene.debug.point_sized(origin, color, 10.0);
            }

            let trigger = actions.select(xr.session(), hand);
            let probe = soft_chain::GrabProbe {
                hand_pos: hand_world,
                ray,
                ray_max_dist: HAND_RAY_LENGTH,
            };

            if let Some((chain_idx, ..)) = grab_state[slot] {
                if trigger < GRAB_RELEASE {
                    soft_chain::release_grab(&mut soft_chains, chain_idx);
                    grab_state[slot] = None;
                }
            } else if trigger > GRAB_PRESS {
                if let Some(hit) = soft_chain::pick_chain_tip(&scene, &soft_chains, &probe, GRAB_TOUCH_MARGIN) {
                    grab_state[slot] = Some((hit.chain_idx, hit.particle, hit.anchor));
                }
            }

            // Whole-model root-handle grab (see `root_grab`) — only tried
            // when this hand isn't already holding a jiggle-chain tip, so
            // one trigger pull can't grab both at once.
            if root_grab_state[slot].is_some() {
                if trigger < GRAB_RELEASE {
                    root_grab_state[slot] = None;
                }
            } else if trigger > GRAB_PRESS && grab_state[slot].is_none() {
                let root_probe = root_grab::RootGrabProbe {
                    hand_pos: hand_world,
                    ray,
                    ray_max_dist: HAND_RAY_LENGTH,
                };
                if let Some(hit) =
                    root_grab::pick_root(&scene, &grab_roots, &root_probe, HANDLE_GRAB_RADIUS, GRAB_TOUCH_MARGIN)
                {
                    let anchor_point = match hit.anchor {
                        root_grab::RootAnchor::Touch => hand_world,
                        root_grab::RootAnchor::Ray { distance } => match ray {
                            Some((origin, dir)) => origin + dir * distance,
                            None => hand_world,
                        },
                    };
                    let root = grab_roots[hit.root_idx];
                    root_grab_state[slot] = Some(root_grab::begin_grab(&scene, root, hit.anchor, hand_rot, anchor_point));
                }
            }

            if let Some(grab) = &root_grab_state[slot] {
                // Same touch-vs-ray tracking rule as jiggle-chain grabs (see
                // `soft_chain::GrabAnchor`'s doc comment) — `Ray` grabs
                // track a point out along the *current* aim ray instead of
                // the hand itself, so distance-grabbed models actually
                // follow the controller instead of barely budging.
                let target = match grab.anchor {
                    root_grab::RootAnchor::Touch => hand_world,
                    root_grab::RootAnchor::Ray { distance } => match ray {
                        Some((origin, dir)) => origin + dir * distance,
                        None => hand_world,
                    },
                };
                root_grab::apply_grab(&mut scene, grab, hand_rot, target);
            }

            if let Some((chain_idx, particle, anchor)) = grab_state[slot] {
                // `Touch` grabs drag with the hand directly; `Ray` grabs
                // (grabbed by aiming from a distance) instead track a point
                // that far out along *this frame's* aim ray, so moving or
                // rotating the controller drags the object even though it's
                // nowhere near the hand — see `soft_chain::GrabAnchor`'s doc
                // comment for why `hand_world` alone doesn't work for those.
                let target = match anchor {
                    soft_chain::GrabAnchor::Touch => hand_world,
                    soft_chain::GrabAnchor::Ray { distance } => match ray {
                        Some((origin, dir)) => origin + dir * distance,
                        None => hand_world,
                    },
                };
                grab_pulls.push(soft_chain::GrabPull {
                    chain_idx,
                    particle,
                    target,
                });
            }
        }
        soft_chain::evaluate_soft_chains(&mut scene, &mut soft_chains, dt, &grab_pulls);

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
