pub struct Texture {
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
    pub fn solid(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
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

    /// Empty (zeroed) map that lives on the GPU after the first `sync`.
    /// Any material slot (`albedo_map`, `normal_map`, …) can use this.
    pub fn gpu_resident(width: u32, height: u32, srgb: bool) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        Self {
            width: w,
            height: h,
            rgba: vec![0; (w * h * 4) as usize],
            version: 1,
            srgb,
            dirty: None,
            gpu_resident: true,
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
