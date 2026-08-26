use crate::store::{Handle, Store};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEXTURE_ID_SEQ: AtomicU64 = AtomicU64::new(1);

pub struct Texture {
    /// Stable catalog key. Required; never empty after insert.
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub version: u64,
    /// Color maps true; normal / metallicRoughness false.
    pub srgb: bool,
    /// Optional dirty region `(x, y, w, h)` for partial GPU upload.
    /// Cleared by the visualizer after sync. `None` = upload full texture.
    /// Ignored when [`Self::gpu_resident`] is set (GPU is source of truth).
    pub dirty: Option<(u32, u32, u32, u32)>,
    /// When true, the visualizer keeps a GPU texture with `STORAGE | COPY_SRC`
    /// and does not overwrite it from [`Self::rgba`] after the initial create.
    /// Host compute / render passes may write via [`crate::WgpuVisualizer::texture_gpu`].
    ///
    /// STORAGE textures cannot use `*-srgb` formats or sRGB views (WebGPU). The GPU
    /// resource is always `rgba8unorm`; [`Self::srgb`] is kept as metadata only.
    pub gpu_resident: bool,
}

impl Texture {
    /// Unique id for runtime-created textures (not loaded from a file).
    pub fn new_id() -> String {
        let n = TEXTURE_ID_SEQ.fetch_add(1, Ordering::Relaxed);
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("tex/{t:x}-{n:x}")
    }

    pub fn solid(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            id: Self::new_id(),
            width: 1,
            height: 1,
            rgba: vec![r, g, b, a],
            version: 1,
            srgb: true,
            dirty: None,
            gpu_resident: false,
        }
    }

    pub fn solid_linear(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            srgb: false,
            ..Self::solid(r, g, b, a)
        }
    }

    /// GPU-backed map after the first `sync`. CPU `rgba` stays empty until
    /// [`Self::ensure_rgba`] — paint layers must not allocate 2k×2k host pixels.
    pub fn gpu_resident(width: u32, height: u32, srgb: bool) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        Self {
            id: Self::new_id(),
            width: w,
            height: h,
            rgba: Vec::new(),
            version: 1,
            srgb,
            dirty: None,
            gpu_resident: true,
        }
    }

    /// Allocate a CPU mirror (`width * height * 4`). Used to seed material maps.
    pub fn ensure_rgba(&mut self) {
        let n = (self.width.max(1) * self.height.max(1) * 4) as usize;
        if self.rgba.len() != n {
            self.rgba.resize(n, 0);
        }
    }

    pub fn mark_changed(&mut self) {
        self.version += 1;
        self.dirty = None;
    }

    /// Mark changed with a dirty rect for partial upload.
    /// No-op for upload purposes when [`Self::gpu_resident`] (still bumps version).
    pub fn mark_changed_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.version += 1;
        if self.gpu_resident {
            self.dirty = None;
            return;
        }
        let x1 = (x + w).min(self.width);
        let y1 = (y + h).min(self.height);
        let x = x.min(self.width);
        let y = y.min(self.height);
        if x1 <= x || y1 <= y {
            self.dirty = None;
            return;
        }
        let nw = x1 - x;
        let nh = y1 - y;
        self.dirty = Some(match self.dirty {
            Some((ox, oy, ow, oh)) => {
                let nx0 = ox.min(x);
                let ny0 = oy.min(y);
                let nx1 = (ox + ow).max(x1);
                let ny1 = (oy + oh).max(y1);
                (nx0, ny0, nx1 - nx0, ny1 - ny0)
            }
            None => (x, y, nw, nh),
        });
    }
}

/// `Store<Texture>` plus id → handle catalog. Duplicate ids intern to the first handle.
pub struct TextureStore {
    inner: Store<Texture>,
    by_id: HashMap<String, Handle<Texture>>,
}

impl Default for TextureStore {
    fn default() -> Self {
        Self {
            inner: Store::default(),
            by_id: HashMap::new(),
        }
    }
}

impl TextureStore {
    pub fn insert(&mut self, mut tex: Texture) -> Handle<Texture> {
        if tex.id.is_empty() {
            tex.id = Texture::new_id();
        }
        if let Some(&h) = self.by_id.get(&tex.id) {
            if self.inner.get(h).is_some() {
                return h;
            }
        }
        let id = tex.id.clone();
        let h = self.inner.insert(tex);
        self.by_id.insert(id, h);
        h
    }

    pub fn get(&self, h: Handle<Texture>) -> Option<&Texture> {
        self.inner.get(h)
    }

    pub fn get_mut(&mut self, h: Handle<Texture>) -> Option<&mut Texture> {
        self.inner.get_mut(h)
    }

    pub fn get_by_id(&self, id: &str) -> Option<Handle<Texture>> {
        let h = *self.by_id.get(id)?;
        self.inner.get(h).map(|_| h)
    }

    pub fn remove(&mut self, h: Handle<Texture>) -> Option<Texture> {
        let tex = self.inner.remove(h)?;
        self.by_id.remove(&tex.id);
        Some(tex)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Handle<Texture>, &Texture)> {
        self.inner.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_resident_skips_cpu_pixels() {
        let t = Texture::gpu_resident(2048, 2048, true);
        assert!(t.rgba.is_empty());
        assert_eq!(t.width, 2048);
        assert!(!t.id.is_empty());
        let mut seeded = Texture::gpu_resident(4, 4, true);
        seeded.ensure_rgba();
        assert_eq!(seeded.rgba.len(), 64);
    }

    #[test]
    fn catalog_interns_and_remove() {
        let mut store = TextureStore::default();
        let mut a = Texture::solid(1, 2, 3, 255);
        a.id = "char/albedo".into();
        let h1 = store.insert(a);
        let mut b = Texture::solid(9, 9, 9, 255);
        b.id = "char/albedo".into();
        let h2 = store.insert(b);
        assert_eq!(h1.key(), h2.key());
        assert_eq!(store.get_by_id("char/albedo").map(|h| h.key()), Some(h1.key()));
        assert_eq!(store.get(h1).unwrap().rgba[0], 1);
        store.remove(h1);
        assert!(store.get_by_id("char/albedo").is_none());
    }
}
