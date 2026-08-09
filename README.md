# mega-render

Rust-библиотека 3D-рендеринга на **wgpu** (WebGPU). Сцена — CPU-данные (`Scene` + `Store`/`Handle`), GPU-бэкенд — `WgpuVisualizer`. Окно/ввод/UI не входят в крейт: хост сам создаёт device/surface и вызывает `sync` → `render` / `render_to`.

```text
cargo run --example demo      # материалы + mega-ui dock
cargo run --example hud_demo  # минимальный хост + in-engine HUD
```

## Возможности

- PBR (metallic/roughness), normal maps, pre-integrated SSS
- Directional + point lights (до 8), ambient, shadow map (PCF / PCSS)
- Env map (HDR/EXR equirect) + skybox
- Пост: GTAO/SSAO, contact shadows, SSGI, SSR, DOF, motion blur, bloom, fog, tonemap, FXAA, vignette, grain
- glTF/GLB (меши, материалы, скины, анимации), async-загрузка
- Immediate HUD (текст, кнопки), debug draw (линии/точки/скелет)
- Debug views внутренних буферов

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│  Host (winit / свой loop)                               │
│    Scene  ──sync──►  WgpuVisualizer  ──render_to──► RT │
└─────────────────────────────────────────────────────────┘
         │                      │
         │ CPU data             │ GPU caches + passes
         ▼                      ▼
   meshes, textures,      shadow → G-buffer → post → HUD
   materials, nodes,
   lights, camera, hud
```

| Слой | Роль |
|------|------|
| **Scene** | Источник правды: ассеты, граф нод, камера, свет, HUD, debug |
| **Visualizer** | Трейт бэкенда: `sync`, `render`, настройки пост/теней |
| **WgpuVisualizer** | Единственная реализация: пайплайны, G-buffer, пост |

Библиотека **не** зависит от `winit` / `mega-ui`. UI в `demo` — только `dev-dependency`.

### Координаты

Левосторонняя система: **X+ вправо, Y+ вверх, Z+ вперёд**. Проекция — DirectX-style (`glam::camera::lh`). glTF импортируется с flip Z.

---

## Структура модулей

```
src/
  lib.rs            публичный API
  scene.rs          Scene, анимации, world matrices, poll_loads
  store.rs          Handle + generational Store
  node.rs           Node, Transform
  mesh.rs / material.rs / texture.rs / skin.rs
  camera.rs / light.rs / shadow.rs
  post_process.rs   все PostProcessSettings
  gltf_load.rs      sync + async glTF
  animation.rs      clips + Animator
  primitives.rs     plane / cube / sphere
  debug_draw.rs / debug_view.rs
  hud.rs / hud_font.rs / input.rs
  ibl.rs
  visualizer/
    mod.rs          trait Visualizer
    wgpu/           бэкенд + WGSL
```

---

## Scene и ресурсы

`Scene` держит generational stores. `Handle<T>` инвалидируется при `remove`.

```rust
pub struct Scene {
    pub meshes: Store<Mesh>,
    pub textures: Store<Texture>,
    pub materials: Store<Material>,
    pub skins: Store<Skin>,
    pub animations: Store<AnimationClip>,
    pub animators: Vec<Animator>,
    pub nodes: Store<Node>,
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub ambient: [f32; 3],
    pub debug: DebugDraw,
    pub hud: Hud,
}
```

### Node

```rust
pub struct Node {
    pub name: String,
    pub parent: Option<Handle<Node>>,
    pub local: Transform,              // T * R * S
    pub mesh: Option<Handle<Mesh>>,
    pub material: Option<Handle<Material>>,
    pub skin: Option<Handle<Skin>>,
    pub visible: bool,
}
```

Рисуются только видимые ноды с `mesh`. World matrix = произведение локальных от корня.

### Material

```rust
Material::new(albedo_rgba, metallic, roughness)
    .with_map(albedo_tex)                          // optional
    .with_sss(strength, color_rgb, curvature);     // optional
```

Карты: `albedo_map` (sRGB), `normal_map`, `metallic_roughness_map` (linear). После правки пикселей — `texture.mark_changed()` / `mesh.mark_changed()`.

### Свет

- `Light::Directional` — направление «куда светит», `cast_shadows`
- `Light::Point` — позиция, intensity, range
- Первая включённая directional с `cast_shadows` пишет shadow map
- Максимум **8** ламп на GPU

### Camera

`look_at` / `orbit`, FOV, near/far. Для DOF: `focus_distance`, `focus_target`, `f_stop`, `tick_focus(dt)`, `autofocus_ground` / `autofocus_point`.

---

## Типичный кадр

```rust
// 1. Логика
scene.poll_loads();                 // async glTF
scene.update_animations(dt);
scene.camera.tick_focus(dt);
scene.debug.clear();
// … HUD begin/widgets/end при необходимости

// 2. GPU
visualizer.ensure_target(w, h);
visualizer.sync(&scene);            // upload meshes/textures by version
visualizer.render_to(&scene, aspect, &swapchain_view);
// или visualizer.render(&scene, aspect) → внутренний present
```

`sync` заливает только изменившиеся меши/текстуры (`version`). Ноды, материалы, кости обновляются каждый кадр в `render`.

### Env map

```rust
visualizer.set_env_map_async("studio.exr");  // decode в фоне
// poll_env_map вызывается из render автоматически
visualizer.post_process().env.enabled = true;
```

Также: `set_env_map` (блокирующий), `clear_env_map`, `env_map_loading`.

---

## GPU-пайплайн

Требование GPU: `max_color_attachment_bytes_per_sample ≥ 36` (5 color targets в G-buffer).

### G-buffer

| Target | Format | Содержимое |
|--------|--------|------------|
| color | `Rgba16Float` | линейный HDR lit |
| normal | `Rgba16Float` | world normals |
| velocity | `Rg16Float` | motion в пикселях |
| orm | `Rgba8Unorm` | occlusion / roughness / metallic |
| albedo | `Rgba8Unorm` | base color (для SSGI) |
| depth | `Depth32Float` | |

### Порядок проходов

1. **Shadow** — depth-only, front cull, одна directional
2. **Scene** — lit mesh (+ skinning) → MRT; debug lines/points; skybox
3. **Post** (если `DebugView::Final` и есть включённые эффекты):
   - AO (GTAO или SSAO) → bilateral blur
   - Contact shadow
   - SSGI (half-res gather → temporal → denoise → upsample → optional 2nd bounce)
   - SSR (march + env fallback + temporal)
   - DOF (prelit composite → CoC gather → temporal)
   - Motion blur (velocity dilate + gather)
   - Bloom (downsample / upsample)
   - **Composite**: AO × contact × SSGI × SSR + fog + bloom + tonemap + grade + vignette + grain
   - FXAA
4. **Debug blit** — иначе (просмотр буферов)
5. **HUD** — поверх present

### Тени (настройки visualizer, не Scene)

```rust
let shadow = visualizer.shadow_settings();
shadow.filter = ShadowFilter::Pcss; // или Pcf
shadow.map_size = 4096;             // 1024 | 2048 | 4096
shadow.bias = 0.00001;
shadow.pcss_light_size = 0.35;
```

### DebugView

`Final`, `SceneColor`, `Normals`, `Roughness`, `Metallic`, `Depth`, `Ao`, `ContactShadow`, `Ssgi`, `Ssr`, `Albedo`, `DofCoc`, `Dof`, `Velocity`.

```rust
visualizer.set_debug_view(DebugView::Normals);
```

---

## Минимальный пример

```rust
use glam::Vec3;
use mega_render::{
    cube, plane, Material, Node, Scene, Transform, Visualizer, WgpuVisualizer,
};

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    let ground = scene.meshes.insert(plane(10.0, 10.0));
    let box_m = scene.meshes.insert(cube(1.0));
    let mat = scene.materials.insert(Material::new([0.8, 0.2, 0.15, 1.0], 0.0, 0.4));
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
        mesh: Some(ground),
        material: Some(mat),
        skin: None,
        visible: true,
    });
    scene.nodes.insert(Node {
        name: "cube".into(),
        parent: Some(root),
        local: Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
        mesh: Some(box_m),
        material: Some(mat),
        skin: None,
        visible: true,
    });
    scene
}

// После создания wgpu::Device / Queue / Surface:
let mut visualizer = WgpuVisualizer::new(&device, &queue);
visualizer.ensure_target(width, height);
visualizer.post_process().tonemap.exposure = 1.2;

// каждый кадр:
visualizer.sync(&scene);
visualizer.render_to(&scene, aspect, &surface_view);
```

Полный хост без UI — `examples/hud_demo.rs`. С dock-панелью эффектов — `examples/demo.rs` + `examples/framework.rs`.

---

## glTF

```rust
// sync
let root = mega_render::load_gltf(&mut scene, "model.glb", Some(parent))?;

// async — в фоне, merge на poll_loads
scene.load_gltf_async("model.glb", Some(parent), |scene, root| {
    if let Some(n) = scene.nodes.get_mut(root) {
        n.local.translation = Vec3::new(0.0, 0.0, -2.0);
    }
    // анимации уже в scene.animators (playing)
});
```

Импорт: иерархия, PBR-материалы/текстуры, skins, animation clips → `Animator`.

Отладка скелета:

```rust
scene.debug.clear();
scene.debug_skeletons([1.0, 0.9, 0.2, 1.0], true);
```

---

## HUD

Immediate-mode поверх 3D (ASCII + кириллица, 8×8 atlas).

```rust
use mega_render::{HudRect, InputFrame};
use glam::Vec2;

let input = InputFrame {
    cursor: mouse_px,
    mouse_down, mouse_pressed, mouse_released,
    scroll_delta: Vec2::ZERO,
    dt,
};
scene.hud.begin(&input, Vec2::new(w as f32, h as f32));
scene.hud.panel(HudRect::from_min_size(Vec2::new(12.0, 12.0), Vec2::new(220.0, 80.0)),
                [0.08, 0.09, 0.12, 0.9]);
scene.hud.text(Vec2::new(24.0, 24.0), "Hello", [1.0, 1.0, 1.0, 1.0], 2.0);
if scene.hud.button("spawn", HudRect::from_min_size(Vec2::new(24.0, 48.0), Vec2::new(100.0, 28.0)), "Spawn") {
    // …
}
let out = scene.hud.end();
// out.want_capture_mouse — не отдавать клик камере
```

Виджеты: `fill`, `panel`, `text`, `button`. Движок не читает OS-ввод — только `InputFrame` от хоста.

---

## Постпроцесс (кратко)

Настройки живут в visualizer:

```rust
let (post, shadow) = visualizer.effect_settings();
post.ao.enabled = true;
post.ao.method = AoMethod::Gtao;
post.ssgi.enabled = true;
SsgiQuality::Medium.apply(&mut post.ssgi);
post.ssr.enabled = true;
post.bloom.enabled = true;
post.dof.enabled = true;
post.fxaa.enabled = true;
```

| Эффект | Заметки |
|--------|---------|
| `env` | intensity, rotation_y; skybox + reflections |
| `ao` | GTAO (default) или SSAO |
| `contact_shadow` | screen-space вдоль primary directional |
| `ssgi` | `ambient_dim` гасит constant ambient; quality presets |
| `ssr` | при включении env specular идёт через SSR confidence blend |
| `dof` | фокус/f-stop на `Camera`; temporal + half-res |
| `motion_blur` | velocity из G-buffer + skin/camera history |
| `tonemap` | ACES / Reinhard + exposure |

Defaults: tonemap + FXAA включены; остальное выключено.

---

## Примеры

### `demo`

Витрина материалов (dielectric/metal spheres, cubes), point lights, async glTF, HDRI, полный пост-стек и dock UI (`mega-ui`) для тюнинга эффектов / debug view / теней.

```text
WASD · E/Q · Shift · RMB look · Esc
cargo run --example demo
```

### `hud_demo`

Тот же wgpu-хост без `mega-ui`: сцена + in-engine HUD (спавн объектов, toggle света).

```text
cargo run --example hud_demo
```

### Свой хост через `framework` (только examples)

```rust
impl Demo for MyDemo {
    fn title() -> &'static str { "my demo" }
    fn build_scene() -> Scene { /* … */ }
    fn configure(v: &mut WgpuVisualizer) {
        v.set_env_map_async("env.exr");
    }
    fn build_ui(ui: &mut Ui, ctx: &mut UiCtx<'_>) -> bool {
        // dock + viewport → ctx.viewport_size
        true
    }
}

fn main() {
    Host::<MyDemo>::run();
}
```

---

## Зависимости

| Crate | Зачем |
|-------|--------|
| `wgpu` 30 | GPU |
| `glam` | математика |
| `bytemuck` | uniforms |
| `image` | png/jpeg/webp/hdr/exr |
| `gltf` | импорт моделей |
| `half` | HDR half-float |

Dev: `winit`, `pollster`, `mega-ui` (path `../ui`), `env_logger`.

---

## Публичный API (ориентир)

```rust
pub use mega_render::{
    // core
    Scene, Node, Transform, Mesh, Material, Texture, Skin,
    Camera, Light, DirectionalLight, PointLight,
    Handle, Store,
    // anim / assets
    AnimationClip, Animator, load_gltf,
    // render
    Visualizer, WgpuVisualizer,
    PostProcessSettings, ShadowSettings, ShadowFilter,
    AoMethod, SsgiQuality, DebugView,
    // helpers
    cube, plane, sphere,
    DebugDraw, Hud, InputFrame,
};
```

Ключевые методы `WgpuVisualizer`:

- `new(device, queue)`
- `ensure_target(w, h)` / `target_view()`
- `sync(scene)` / `render` / `render_to`
- `post_process()` / `shadow_settings()` / `effect_settings()`
- `set_env_map` / `set_env_map_async` / `clear_env_map`
- `set_debug_view`

---

## Ограничения

- Один бэкенд: wgpu
- Одна directional shadow map (орто, фиксированный frustum вокруг сцены)
- До 8 lights
- Нет прозрачности / multi-pass materials
- Нет окна в библиотеке — хост обязан поднять device с достаточным `max_color_attachment_bytes_per_sample`
