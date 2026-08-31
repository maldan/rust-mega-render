# Material binary format (`MAT `)

Контейнер для рецепта `mega_render::Material`: один файл — один материал. Хранит PBR-параметры, SSS, alpha cutout, hair-шейдинг и **ссылки на текстуры по строковому id**, а не `Handle`. Все чанки, кроме `PBR `, **опциональны** и идут **в любом порядке**.

Endian: **little-endian**. FourCC и id текстур: ASCII / UTF-8.

Пока всё пишется «как есть» (`f32` / `u32` / `u16`). Квантование заложено во флагах, но в version 1 **запрещено**.

Код: `src/io/material.rs` (`MaterialFile`, `to_bytes` / `from_bytes`). Обёртка `Material::to_bytes(&self, &TextureStore)` в `src/material.rs`.

---

## Файл

```
offset  size  field
0       4     magic    b"MAT "
4       2     version  u16     // 1
6       2     flags    u16
8       …     chunks   до EOF
```

Отдельного `chunk_count` нет: конец файла = конец списка. Обрезанный последний чанк — ошибка (`Truncated`).

### `flags`

| бит | имя         | version 1                      |
|-----|-------------|---------------------------------|
| 0   | `QUANTIZED` | должен быть 0; иначе не читать (`Quantized`) |
| 1–15 | reserved   | писать 0; читатель игнорирует  |

---

## Чанк

Общий заголовок как в `MESH` / `SKIN` — неизвестный FourCC можно пропустить.

```
0    4    id      [u8; 4]   // FourCC, верхний регистр, pad пробелами
4    4    size    u32       // байт payload сразу после этого поля
8    size payload
      …   pad     0–3 нуля  // следующий чанк с 4-байтовой границы
```

`pad` **не входит** в `size`:

```
aligned = (8 + size + 3) & !3
```

`size == 0` лучше не писать: нет данных — нет чанка.

Дубликаты: каждый FourCC из таблицы ниже — не больше одного; повтор — ошибка `Duplicate(id)`.

Смешивать одиночные карты (`ALB `/`NRM `/`MR  `) и UDIM-карты (`UALB`/`UNRM`/`UMR `) в одном файле нельзя — `MixedMaps` (в рантайме `Material.maps` и так один из двух вариантов, так что писатель это смешение никогда не производит; ошибка ловит битые/склеенные вручную файлы).

Неизвестный `id`: пропустить `size` (+ pad) и продолжить.

---

## FourCC

| id     | элемент                          | поле `Material` / рантайм                     |
|--------|----------------------------------|-------------------------------------------------|
| `PBR ` | 4×`f32` + 2×`f32`                | `albedo`, `metallic`, `roughness`               |
| `SSS ` | `f32` + 3×`f32` + `f32`          | `sss_strength`, `sss_color`, `sss_curvature`    |
| `CUT `  | `f32`                            | `alpha_cutoff`                                  |
| `HAIR` | см. ниже                         | `shading_model = ShadingModel::Hair(_)`         |
| `ALB ` | id-строка                        | `maps: Single { albedo }`                       |
| `NRM ` | id-строка                        | `maps: Single { normal }`                       |
| `MR  ` | id-строка                        | `maps: Single { metallic_roughness }`           |
| `UALB` | список `(udim, id-строка)`       | `maps: Udim { albedo }`                         |
| `UNRM` | список `(udim, id-строка)`       | `maps: Udim { normal }`                         |
| `UMR ` | список `(udim, id-строка)`       | `maps: Udim { metallic_roughness }`             |
| `TEXG` | IR процедурного графа           | `graph: Option<TexGraphFile>`                    |

`ShadingModel::Standard` в файл не пишется никак — отсутствие `HAIR` и означает `Standard`.

---

## Payload по типам

### `PBR ` — база (пишется всегда)

```
f32 albedo[4]    // rgba
f32 metallic
f32 roughness
```

`size = 24`. Единственный обязательный чанк: если его нет — ошибка? Нет: читатель не требует `PBR`, просто оставляет `MaterialFile::default()` для этих полей (albedo/metallic/roughness дефолтного `Material`). Писатель (`to_bytes`) всегда его пишет.

### `SSS ` — subsurface scattering

Пишется, только если `sss_strength > 0.0`.

```
f32 sss_strength
f32 sss_color[3]
f32 sss_curvature
```

`size = 20`.

### `CUT ` — alpha cutout

Пишется, только если `alpha_cutoff > 0.0` (0 = материал непрозрачный, чанк не нужен).

```
f32 alpha_cutoff
```

`size = 4`.

### `HAIR` — Kajiya-Kay dual-specular параметры

Пишется, только если `shading_model` — `ShadingModel::Hair(_)`.

```
f32 primary_shift
f32 secondary_shift
f32 primary_exponent
f32 secondary_exponent
f32 secondary_tint[3]
f32 secondary_strength
f32 tip_fade
f32 cutout_fringe
u32 flags           // бит 0 = soft_blend
```

`size = 44` **ровно** (в отличие от остальных чанков, здесь при чтении проверяется точное совпадение длины, а не «не меньше»; несовпадение — `SizeMismatch`).

### `ALB ` / `NRM ` / `MR  ` — одиночная карта (id текстуры)

```
u16 id_len
u8  id[id_len]     // UTF-8, без NUL
```

`size = 2 + id_len`. Пустая/отсутствующая карта — чанк просто не пишется (`None`, а не пустая строка).

Внутри чанка паддинга нет — выравнивание чанка целиком делает общий `pad` в чанк-хедере.

### `UALB` / `UNRM` / `UMR ` — UDIM-набор

```
u32 count
для каждого тайла:
  u32 udim          // напр. 1001
  u16 id_len
  u8  id[id_len]    // UTF-8, без NUL
  u8  pad[0..3]     // до 4-байтовой границы ОТНОСИТЕЛЬНО НАЧАЛА PAYLOAD
```

Паддинг стоит **после каждого тайла** (а не только в конце payload), чтобы следующий `udim: u32` всегда лежал на 4-байтовой границе:

```
tile_section_bytes = 4 + 2 + id_len
pad = (4 - (tile_section_bytes % 4)) % 4
```

Этот pad **входит** в `size` (в отличие от pad чанка на верхнем уровне, который в `size` не входит). `size = 4 + Σ (tile_section_bytes + pad)`. Пустой список — чанк не пишется.

### `TEXG` — процедурный граф

Пишется, только если `MaterialFile.graph` — `Some`. Старый читатель чанк пропускает (неизвестный FourCC). Версия контейнера `MAT ` остаётся 1.

Детект: `file.is_procedural()` / `file.graph.is_some()`.

Код payload: `src/texgen/ser.rs` (`TexGraphFile`). Bake в сцену: `GpuEval::instantiate` — печёт карты из графа, затем копирует PBR/SSS/CUT/HAIR с рецепта на получившийся `Material`.

```
u32 ir_version     // 1
u32 resolution     // bake, clamp 64..=2048 при чтении
u64 next_serial
u16 output_id_len + utf8
u32 node_count
  повторить node_count раз: фиксированный набор полей GraphNode
  (без preview-слотов), kind как u8, строки как u16 len + utf8,
  градиент: u16 count + стопы
u32 link_count
  для каждой связи: 4 строки (from_node, from_port, to_node, to_port)
```

Неизвестный `ir_version` — `MaterialBytesError::Graph(UnsupportedVersion)`. Нет Output — `NoOutput`. Карты `ALB `/`NRM `/`MR  ` при процедурном экспорте из генератора не пишутся: пиксели рождаются на bake.

---

## На load

`MaterialFile::from_bytes` возвращает **рецепт**, а не готовый `Material`: слоты карт — это `String` (id текстуры), а не `Handle<Texture>`, потому что хендлы из одного процесса/сцены не переносимы в файл.

Шаги приложения:

1. Распарсить байты в `MaterialFile` (`from_bytes`).
2. Если `file.is_procedural()`: `GpuEval::instantiate(device, queue, scene, &file)` — карты из графа, скаляры с рецепта. Иначе для каждого id из `file.texture_ids()` найти текстуру в `TextureStore`.
3. Собрать `Material { albedo, metallic, roughness, sss_*, alpha_cutoff, shading_model, maps }` вручную, подставив в `MaterialMaps::Single`/`Udim` уже полученные `Handle<Texture>` (или `None`, если id не нашёлся).

Обратный путь: `Material::to_bytes(&self, &TextureStore)` берёт хендлы из `self.maps`, резолвит их в `TextureStore` через `Texture.id` и пишет `MaterialFile` → байты. Текстура без `id` (`""`) в файл не попадает — слот пишется как отсутствующий.

Не сериализуется: `Handle<Texture>` (только строковый id), сами байты текстуры (отдельный ассет), `ShadingModel::Standard` (кодируется отсутствием `HAIR`).

---

## Примеры кода

### Сохранение

`Material::to_bytes` сам резолвит хендлы в `self.maps` через `TextureStore` и отдаёт готовый `MAT `-блоб — руками `MaterialFile` собирать не нужно.

```rust
use mega_render::{Material, MaterialMaps, Texture, TextureStore};

let mut textures = TextureStore::default();

let mut albedo = Texture::solid(200, 160, 140, 255);
albedo.id = "char/skin/albedo".into(); // без id текстура не попадёт в файл
let albedo_h = textures.insert(albedo);

let mut mat = Material::new([1.0, 1.0, 1.0, 1.0], 0.0, 0.5);
mat.maps = MaterialMaps::Single {
    albedo: Some(albedo_h),
    normal: None,
    metallic_roughness: None,
};
mat.with_sss(0.6, [1.0, 0.35, 0.2], 0.3); // включит чанк SSS

let bytes: Vec<u8> = mat.to_bytes(&textures);
std::fs::write("char_skin.mat", &bytes)?;
```

### Загрузка: `MaterialFile` — это рецепт, не `Material`

`MaterialFile::from_bytes` возвращает промежуточный рецепт с id-строками вместо хендлов. Чтобы получить `Material`, нужно самому подставить хендлы:

```rust
use mega_render::{Material, MaterialFile, MaterialFileMaps, MaterialMaps, TextureStore};

let bytes = std::fs::read("char_skin.mat")?;
let file: MaterialFile = MaterialFile::from_bytes(&bytes)?; // рецепт, Handle ещё нет

let mut textures = TextureStore::default(); // сюда уже могли быть загружены другие текстуры

// подтягиваем текстуры, которых ещё нет в TextureStore, с диска по id
let resolve = |textures: &mut TextureStore, id: &str| -> Option<mega_render::Handle<mega_render::Texture>> {
    if let Some(h) = textures.get_by_id(id) {
        return Some(h); // уже загружена — просто переиспользуем хендл
    }
    // id из ALB/NRM/MR/UALB/UNRM/UMR — это то же самое, что писатель положил
    // в Texture.id при сохранении (например, путь к файлу текстуры).
    let tex = load_texture_from_disk(id)?; // свой загрузчик PNG/KTX/т.п., формат MAT картинки не хранит
    Some(textures.insert(tex))
};

let maps = match file.maps {
    MaterialFileMaps::Single { albedo, normal, metallic_roughness } => MaterialMaps::Single {
        albedo: albedo.and_then(|id| resolve(&mut textures, &id)),
        normal: normal.and_then(|id| resolve(&mut textures, &id)),
        metallic_roughness: metallic_roughness.and_then(|id| resolve(&mut textures, &id)),
    },
    MaterialFileMaps::Udim { albedo, normal, metallic_roughness } => MaterialMaps::Udim {
        albedo: albedo.into_iter().filter_map(|(udim, id)| Some((udim, resolve(&mut textures, &id)?))).collect(),
        normal: normal.into_iter().filter_map(|(udim, id)| Some((udim, resolve(&mut textures, &id)?))).collect(),
        metallic_roughness: metallic_roughness.into_iter().filter_map(|(udim, id)| Some((udim, resolve(&mut textures, &id)?))).collect(),
    },
};

let material = Material {
    albedo: file.albedo,
    metallic: file.metallic,
    roughness: file.roughness,
    sss_strength: file.sss_strength,
    sss_color: file.sss_color,
    sss_curvature: file.sss_curvature,
    maps,
    alpha_cutoff: file.alpha_cutoff,
    shading_model: file.shading_model,
};

let mat_handle = scene.materials.insert(material);
```

Если текстура не нашлась и не загрузилась (`resolve` вернул `None`), слот просто станет `None` — материал всё равно соберётся, но без этой карты (albedo tint / чёрная MR и т.п.), без паники.

### Предзагрузка всех текстур одним списком

Если нужно заранее пройтись по всем id и подгрузить недостающие текстуры одним пакетом (например, до вставки материала в сцену):

```rust
let file = MaterialFile::from_bytes(&bytes)?;

for id in file.texture_ids() { // albedo, normal, MR — без дублей, в этом порядке
    if textures.get_by_id(id).is_none() {
        if let Some(tex) = load_texture_from_disk(id) {
            textures.insert(tex);
        }
    }
}
// после этого resolve() выше всегда найдёт их через get_by_id и на диск не полезет
```

### Обработка ошибок формата

```rust
match MaterialFile::from_bytes(&bytes) {
    Ok(file) => { /* ... */ }
    Err(mega_render::MaterialBytesError::UnsupportedVersion(v)) => {
        eprintln!("файл материала новее, чем понимает эта версия движка: {v}");
    }
    Err(e) => eprintln!("битый .mat файл: {e}"),
}
```

---

## Ошибки (`MaterialBytesError`)

| вариант | когда |
|---|---|
| `Truncated` | меньше 8 байт заголовка, или чанк обрезан |
| `BadMagic` | первые 4 байта не `b"MAT "` |
| `UnsupportedVersion(v)` | `version != 1` |
| `Quantized` | бит `QUANTIZED` установлен |
| `Duplicate(id)` | повтор уникального FourCC |
| `MixedMaps` | в файле одновременно есть одиночные (`ALB`/`NRM`/`MR`) и UDIM (`UALB`/`UNRM`/`UMR`) чанки |
| `SizeMismatch` | длина `HAIR`/строки/UDIM-списка не сходится с ожидаемой |
| `BadUtf8` | id текстуры — не валидный UTF-8 |
| `NotProcedural` | `instantiate` вызван без чанка `TEXG` |
| `Graph(_)` | битый / неизвестный IR внутри `TEXG` |

---

## Размеры — сводка

```
chunk_bytes = 8 + size + pad4(size)

PBR:   size = 24
SSS:   size = 20
CUT:   size = 4
HAIR:  size = 44
ALB/NRM/MR:  size = 2 + id_len
UALB/UNRM/UMR: size = 4 + Σ (6 + id_len + pad_to_4)
TEXG:  size = длина IR (`TexGraphFile::to_bytes`)
```

---

## Пример

Материал по умолчанию (`Material::default()`, без карт, без SSS/cutout/hair) — единственный чанк `PBR `:

```
4d 41 54 20     "MAT "
01 00           version = 1
00 00           flags = 0

50 42 52 20     PBR
18 00 00 00     size = 24
00 00 80 3f     albedo.r = 1.0
00 00 80 3f     albedo.g = 1.0
00 00 80 3f     albedo.b = 1.0
00 00 80 3f     albedo.a = 1.0
00 00 00 00     metallic = 0.0
00 00 00 3f     roughness = 0.5
```

24 байта payload уже кратны 4 — pad нет.

Материал с albedo-картой `"red"` добавил бы после `PBR` ещё один чанк:

```
41 4c 42 20     ALB
05 00 00 00     size = 5
03 00           id_len = 3
72 65 64        "red"
00              pad (1 байт до границы 4)
```

---

## Контракт писателя (version 1)

- Magic `MAT `, `version = 1`, `flags = 0`.
- `PBR ` пишется всегда; `SSS `/`CUT `/`HAIR` — только если значение отличается от «выключено» (`sss_strength > 0`, `alpha_cutoff > 0`, `shading_model == Hair`).
- Карту не писать, если у текстуры пустой `id` или слот `None`.
- FourCC ровно из таблицы, с пробелами.
- Не смешивать `Single` и `Udim` карты в одном файле (и так невозможно из `Material.maps`, но явный запрет фиксирует инвариант).
- Не ставить `QUANTIZED`.
- Не сериализовать `Handle<Texture>` — только `Texture.id`.

Читатель отвергает файл, если: неверный magic, `version != 1`, `QUANTIZED`, обрезанный чанк, несовпадение `size` (для `HAIR` — точное, для строк/UDIM — по вычисленной формуле), дубликат FourCC, одновременно одиночные и UDIM чанки карт, id текстуры не UTF-8.
