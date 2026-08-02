pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub version: u64,
    /// Color maps true; normal / metallicRoughness false.
    pub srgb: bool,
}

impl Texture {
    pub fn solid(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            width: 1,
            height: 1,
            rgba: vec![r, g, b, a],
            version: 1,
            srgb: true,
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
    }
}
