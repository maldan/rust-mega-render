//! Material showcase demo scene.
//!
//! Move: WASD · Up/Down: E/Q · Sprint: Shift · Look: RMB · Quit: Esc
//!
//! ```text
//! cargo run --example demo
//! ```

#[path = "framework.rs"]
mod framework;

use framework::{Demo, Host};
use glam::Vec3;
use mega_render::{
    cube, plane, sphere, Light, Material, Node, PointLight, Scene, Transform,
};

fn build_demo_scene() -> Scene {
    let mut scene = Scene::new();
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        d.intensity = 2.5;
        d.color = [1.0, 0.98, 0.92];
    }
    scene.lights.push(Light::Point(PointLight {
        position: Vec3::new(-2.0, 3.0, 1.5),
        color: [0.5, 0.7, 1.0],
        intensity: 6.0,
        range: 10.0,
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

    // dielectric spheres: roughness left → right
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

    // metal spheres: roughness left → right
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

    // cubes with mixed materials
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

    scene
}

struct MaterialDemo;

impl Demo for MaterialDemo {
    fn title() -> &'static str {
        "mega-render demo"
    }

    fn build_scene() -> Scene {
        build_demo_scene()
    }

    fn update(scene: &mut Scene, _dt: f32) -> bool {
        scene.debug.clear();
        scene.debug.axes(Vec3::ZERO, 1.5);
        false
    }
}

fn main() {
    Host::<MaterialDemo>::run();
}
