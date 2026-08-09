pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub version: u64,
    /// Color maps true; normal / metallicRoughness false.
    pub srgb: bool,
    /// Optional dirty region `(x, y, w, h)` for partial GPU upload.
    /// Cleared by the visualizer after sync. `None` = upload full texture.
    pub dirty: Option<(u32, u32, u32, u32)>,
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
        }
    }

    pub fn solid_linear(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            srgb: false,
            ..Self::solid(r, g, b, a)
        }
    }

    pub fn mark_changed(&mut self) {
        self.version += 1;
        self.dirty = None;
    }

    /// Mark changed with a dirty rect for partial upload.
    pub fn mark_changed_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.version += 1;
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
