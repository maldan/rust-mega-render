# Hair in mega-render

Причёска — это **карты-ленты (cards)** по гайдам, не физические нити. Рендер умеет:

1. спечь fiber-текстуры (albedo / roughness / normal);
2. построить меш по гайдам (ribbon или tube) с skin-весами;
3. создать rest-кости и `Skin`;
4. повесить это в `Scene`.

Симуляции здесь нет. Кости стоят в rest. Двигать суставы можно своей анимацией / IK / физикой.

Авторский тул — `hair-editor`. Он пишет те же структуры. В другой проект переносишь `HairDesc` (гайды + `HairParams`) и вызываешь API ниже.

Модуль: `mega_render::hair`.

---

## Быстрый старт

Один вызов собирает всё готовое к кадру:

```rust
use mega_render::hair::{
    spawn_hair, HairDesc, HairGuide, HairGuidePoint, HairLayerDesc, HairParams,
};
use glam::Vec3;

let mut params = HairParams::default();
params.shape = mega_render::hair::HairShape::Ribbon;

let desc = HairDesc {
    layers: vec![HairLayerDesc {
        name: "main".into(),
        guides: vec![HairGuide {
            points: vec![
                HairGuidePoint { pos: Vec3::new(0.0, 1.6, 0.0), normal: Vec3::Y },
                HairGuidePoint { pos: Vec3::new(0.0, 1.4, 0.05), normal: Vec3::Y },
                HairGuidePoint { pos: Vec3::new(0.0, 1.2, 0.08), normal: Vec3::Y },
            ],
            mirror_x: false,
            lift: 0.06,
            width: 0.025,
            fill_with: None,
            is_static: false,
        }],
        params,
        fiber_gap: 0.94,
        visible: true,
    }],
};

let hair = spawn_hair(&mut scene, &desc);
// Дальше обычный Visualizer::render. Hair-материалы сами идут в hair-pass.
```

`HairInstance` держит хендлы слоёв: меши, текстуры, материал, `Skin`, joint-ноды, rest-цепи (`chains`). Не теряй его, если потом хочешь двигать кости или снести причёску из сцены.

Несколько слоёв = несколько независимых мешей (чёлка / затылок / пушок) с общим fiber-bake (rough/normal одни, albedo на слой — свой цвет и `fiber_gap`).

---

## Из чего состоит причёска

```
HairDesc
  └─ HairLayerDesc[]          // слой редактора
        ├─ guides[]           // полилинии на скальпе
        ├─ params             // форма, стиль, fiber, shading
        └─ fiber_gap          // дырки в albedo этого слоя
```

**Гайд** — ломаная `HairGuidePoint { pos, normal }`. `pos` на поверхности (или рядом), `normal` — нормаль скальпа. Меш отъезжает от поверхности на `lift` вдоль нормали (ещё масштабируется `lift_curve` / `lift_mult`). Ширина карты — `width` × `width_curve` × `width_mult`.

Полезные флаги гайда:

| Поле | Что делает |
|---|---|
| `mirror_x` | Дублирует гайд через X=0, если он не на середине. Кости и веса тоже дублируются. |
| `fill_with` | Индекс другого гайда: между парой сеются extra-стренды (`params.density`). |
| `is_static` | **Отдельный меш без `Skin`.** Костей нет, карты стоят в rest. Fill между двумя static тоже rigid; fill к живому гайду остаётся на скиннутом меше (веса только с живой стороны). |

Порядок костей = порядок expanded-гайдов: исходный, потом зеркало. Static и короткие (< 2 точек) слот не занимают. Лимит суставов на слой: `MAX_HAIR_BONES` (255).

---

## Как генерируется

### 1. Карта волокна

`bake_hair_maps(&params, &[HairLayerBake { gap, color_stops }]) -> HairMaps`

CPU рисует тонкие «нити» в UV карты:

- **U (X)** — поперёк карты (волокна рядом);
- **V (Y)** — от корня к кончику (градиент цвета).

На выходе RGBA:

- `albedos[i]` — sRGB, alpha = покрытие + `gap` в пустотах;
- `roughness` — linear metallic-roughness (G = вариация, остальные каналы фиктивные);
- `normal` — linear tangent-space bump волокон.

Общие на bake: `card_strands`, `tex_width`/`tex_height`, `fiber_*`, `seed`, `rough_variance`, `normal_strength`.  
На слой: цвет (`color_stops`) и `gap`.

`spawn_hair` берёт fiber-поля из **первого** слоя `HairDesc`, albedo печёт на каждый слой.

### 2. Меш

`generate_hair_mesh(...) -> HairMeshes { skinned, rigid }`

Каждый кусок — `HairCardMeshes { front, back }`. Пустой кусок: `has_geo() == false`.

1. Гайды сэмплятся в стренды (`segments`, `smooth`, Catmull vs polyline).
2. Стиль деформирует путь: Straight / Roll / Curl / Wave / Crimp / Coil.
3. Density fill — lerp между парами (+ `fill_curve` = выпуклость).
4. `multiply` — копии с рандомом (`layer_rand[0]`, `seed`).
5. `layers` + `layer_gap` — auto-stack: дополнительные меши вдоль **корневой** нормали (`auto_stack_index`).
6. Ribbon = две карты (front + back, разные winding). Tube = призма по `section_curve`, без back.

Веса суставов (только `skinned`): вдоль стренда `t=0..1` блендятся соседние кости своей цепи; fill-стренды блендятся между двумя цепями. `rigid` пишется через `apply_hair_mesh_rigid` (без joint palette), нода **без** `Skin`. Нулевые веса на общем скине утаскивают вершины в origin — поэтому static и animated разнесены.

Вершинный `colors[0]`:

- `x` — локальная ширина (для шейдера);
- `y` — shade-множитель стренда;
- `z` — opacity слоя (`layer_alpha[0]`).

UV: `u=0..1` поперёк ленты, `v=0` корень, `v=1` кончик.

`apply_hair_mesh` — скиннутый буфер. `apply_hair_mesh_rigid` — тот же меш без joints/weights.

### 3. Риг

`generate_hair_rig(guides, lift) -> HairRig`  
`spawn_hair_joints(scene, prefix, &rig) -> (joints, Option<Skin>)`

Кость = сегмент между соседними точками гайда (после `pos + normal * lift`, без style-деформации). Inverse bind считается из rest-кадров. Joint-ноды невидимые, без родителя, `local` = rest.

`spawn_hair` вешает один `Skin` только на **skinned** front/back слоя. `HairStack.static_*` без скина.

---

## Как рисуется

Материал: `ShadingModel::Hair(HairShading)`, карты:

- `albedo_map` — fiber albedo (sRGB);
- `metallic_roughness_map` — baked roughness;
- `normal_map` — fiber bump;
- `alpha_cutoff` — для ribbon ≈ `params.cutout`, для tube `0`.

Визуализатор гонит Hair не в обычный G-buffer:

1. **Depth prepass** (если `soft_blend == false`) — непрозрачное ядро по cutoff;
2. **Hair blend** — dual-specular (Scheuermann / shifted Kajiya-Kay): два блика, сдвинутых вдоль нормали.

`tip_fade` — софт кончика по `UV.v` в шейдере, не в текстуре.  
`soft_blend` — без hair-prepass, карты полупрозрачно перекрываются (сквозь дырки может читаться череп).  
`cutout_fringe` — ширина мягкой кромки ниже cutoff.

Ribbon: back-нода рисуется раньше front (порядок спавна в `spawn_hair`). Для tube back скрыт.

Обычный `Visualizer` больше ничего настраивать не надо: увидел `ShadingModel::Hair` — отправил в hair-pass.

---

## Кости в рантайме

После `spawn_hair` суставы в rest. Чтобы волосы шевелились, каждый кадр пиши `node.local` joint-нод (world-space кадр сегмента, как в редакторском test-режиме). GPU скиннет меш по `Skin.joints` + `inverse_bind`.

Static-гайды в `joints` не попадают и рисуются отдельным мешем (`HairStack.static_node` / `static_back_node`). Rest-pose для чёлки / залома.

Физики в рендере нет. Редакторский test-sim (`hair-editor` + `mega-physics`) только двигает уже созданные суставы.

---

## Если не нужен spawn_hair

Тот же пайплайн по кускам:

```rust
use mega_render::hair::{
    apply_hair_mesh, apply_hair_mesh_rigid, bake_hair_maps, fill_pairs_of, generate_hair_mesh,
    generate_hair_rig, spawn_hair_joints, HairLayerBake,
};

let maps = bake_hair_maps(&params, &[HairLayerBake {
    gap: 0.94,
    color_stops: params.color_stops.clone(),
}]);
// сам кладёшь maps.* в Texture + Material { shading_model: Hair(..) }

let fills = fill_pairs_of(&guides);
let meshes = generate_hair_mesh(&guides, &fills, &params, 0);
apply_hair_mesh(skinned_mesh, meshes.skinned.front);
apply_hair_mesh_rigid(static_mesh, meshes.rigid.front);

let rig = generate_hair_rig(&guides, lift);
let (joints, skin) = spawn_hair_joints(scene, "hair_j0", &rig);
```

Так делает редактор: меши уже есть, он только перезаписывает буферы и печёт текстуры с debounce.

---

## Параметры, которые чаще крутят

**Форма:** `shape`, `segments`, `density`, `smooth`, `multiply`, `layers` / `layer_gap`, `tip_density`.  
**Стиль:** `style` + соответствующий блок (`roll_*`, `curl_*`, `wave_*`, `crimp_*`, `coil_*`).  
**Карта:** `card_strands`, `tex_width`/`tex_height`, `fiber_waviness`, `fiber_overlap`, `fiber_gap` / `fiber_gap_shade`, `seed`.  
**Шейдер:** `roughness`, `hair_shading`, `tip_fade`, `cutout`, `soft_blend`.  
**Рандом стрендов:** `layer_rand[0]` — length / width / roll / offset / rotate.

`HairParams::default()` — рабочий тёмный ribbon. Кривые (`lift_curve`, `width_curve`, `section_curve`) — те же Hermite, что в UI редактора, без зависимости на mega-ui.

---

## Редактор → другой проект

1. В `hair-editor` собираешь гайды (клики / Lua).
2. Забираешь данные слоя: точки гайдов, `mirror` / `fill_with` / `static`, overlay-параметры слоя, глобальный fiber.
3. Собираешь `HairDesc` и зовёшь `spawn_hair`.

Lua редактора (`export_lua`) — формат **редактора**, не рантайма. В игре удобнее держать свой сериализованный `HairDesc` (или просто захардкодить / сгенерировать гайды кодом). Главное: те же мировые точки и те же `HairParams`, иначе bake/меш разъедутся.

---

## Чего в рендере нет

- Физика, ветер, капсулы головы, grab костей.
- Привязка корня к треугольнику скальпа (`node`/`tri`/`bary` живут только в редакторе).
- Авто-экспорт в glTF. Меш+скин уже в `Scene` — можно сохранить своим пайплайном, если он есть.
- GPU-bake текстур: всё CPU.

Если картинка «не как в редакторе»: проверь, что в `HairParams` попали overlay слоя (не только глобальные слайдеры), что `fiber_gap` слоя не затёрт дефолтом, и что auto-stack (`params.layers`) совпадает.
