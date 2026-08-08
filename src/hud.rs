//! Immediate-mode screen-space HUD (imgui-style).
//!
//! Each frame: [`Hud::begin`] → widgets → [`Hud::end`] → visualizer draws
//! [`Hud::primitives`] on top of the present target.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use glam::Vec2;

use crate::hud_font::{self, GLYPH_H, GLYPH_W};
use crate::input::InputFrame;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub fn from_min_size(min: Vec2, size: Vec2) -> Self {
        Self {
            min,
            max: min + size,
        }
    }

    pub fn width(self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn height(self) -> f32 {
        self.max.y - self.min.y
    }

    pub fn contains(self, p: Vec2) -> bool {
        p.x >= self.min.x && p.y >= self.min.y && p.x < self.max.x && p.y < self.max.y
    }

    pub fn inset(self, v: f32) -> Self {
        Self {
            min: self.min + Vec2::splat(v),
            max: self.max - Vec2::splat(v),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HudId(u64);

impl HudId {
    pub fn new(v: impl Hash) -> Self {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        Self(h.finish())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HudQuad {
    pub rect: Rect,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct HudOutput {
    pub want_capture_mouse: bool,
    pub want_capture_keyboard: bool,
}

/// Immediate HUD built each frame and drawn as a 2D overlay.
pub struct Hud {
    quads: Vec<HudQuad>,
    size: Vec2,
    input: InputFrame,
    active: Option<HudId>,
    want_capture_mouse: bool,
    white_uv: [f32; 2],
    atlas_w: u32,
    atlas_h: u32,
    /// CPU font atlas (R8) — uploaded once by the visualizer.
    pub(crate) atlas_pixels: Vec<u8>,
}

impl Default for Hud {
    fn default() -> Self {
        Self::new()
    }
}

impl Hud {
    pub fn new() -> Self {
        let (atlas_pixels, atlas_w, atlas_h) = hud_font::bake_atlas();
        let white_uv = hud_font::white_uv(atlas_w, atlas_h);
        Self {
            quads: Vec::new(),
            size: Vec2::ZERO,
            input: InputFrame::default(),
            active: None,
            want_capture_mouse: false,
            white_uv,
            atlas_w,
            atlas_h,
            atlas_pixels,
        }
    }

    pub fn begin(&mut self, input: &InputFrame, size: Vec2) {
        self.quads.clear();
        self.input = input.clone();
        self.size = size;
        self.want_capture_mouse = false;
    }

    pub fn end(&mut self) -> HudOutput {
        if self.active.is_some() {
            self.want_capture_mouse = true;
        }
        let out = HudOutput {
            want_capture_mouse: self.want_capture_mouse,
            want_capture_keyboard: false,
        };
        // Clear after widgets saw mouse_released this frame.
        if !self.input.mouse_down {
            self.active = None;
        }
        out
    }

    pub fn size(&self) -> Vec2 {
        self.size
    }

    pub fn primitives(&self) -> &[HudQuad] {
        &self.quads
    }

    pub(crate) fn atlas_size(&self) -> (u32, u32) {
        (self.atlas_w, self.atlas_h)
    }

    /// Solid rectangle (decorative — does not capture mouse).
    pub fn fill(&mut self, rect: Rect, color: [f32; 4]) {
        self.push_solid(rect, color);
    }

    /// Background panel that captures mouse while hovered.
    pub fn panel(&mut self, rect: Rect, color: [f32; 4]) {
        self.push_solid(rect, color);
        if rect.contains(self.input.cursor) {
            self.want_capture_mouse = true;
        }
    }

    /// Bitmap text (8×8 glyphs, `scale` multiplies pixel size).
    /// Supports ASCII and Cyrillic (А-я, Ёё).
    pub fn text(&mut self, pos: Vec2, s: &str, color: [f32; 4], scale: f32) {
        let scale = scale.max(1.0);
        let mut x = pos.x;
        let y = pos.y;
        let gw = GLYPH_W as f32 * scale;
        let gh = GLYPH_H as f32 * scale;
        for ch in s.chars() {
            if ch == '\n' {
                continue;
            }
            let idx = hud_font::char_index(ch);
            let (uv0, uv1) = hud_font::glyph_uv(idx, self.atlas_w, self.atlas_h);
            self.quads.push(HudQuad {
                rect: Rect::from_min_size(Vec2::new(x, y), Vec2::new(gw, gh)),
                uv_min: uv0,
                uv_max: uv1,
                color,
            });
            x += gw;
        }
    }

    pub fn text_width(&self, s: &str, scale: f32) -> f32 {
        s.chars().count() as f32 * GLYPH_W as f32 * scale.max(1.0)
    }

    /// Clickable button. Returns `true` on release while still hovered.
    pub fn button(&mut self, id: impl Hash, rect: Rect, label: &str) -> bool {
        let id = HudId::new(id);
        let hovered = rect.contains(self.input.cursor);
        if hovered {
            self.want_capture_mouse = true;
        }
        if hovered && self.input.mouse_pressed {
            self.active = Some(id);
        }
        let held = self.active == Some(id);
        if held {
            self.want_capture_mouse = true;
        }

        let bg = if held {
            [0.22, 0.45, 0.75, 0.95]
        } else if hovered {
            [0.30, 0.55, 0.85, 0.95]
        } else {
            [0.18, 0.22, 0.30, 0.92]
        };
        self.push_solid(rect, [0.05, 0.06, 0.08, 0.95]);
        self.push_solid(rect.inset(1.0), bg);

        let scale = 2.0;
        let tw = self.text_width(label, scale);
        let th = GLYPH_H as f32 * scale;
        let tp = Vec2::new(
            rect.min.x + (rect.width() - tw) * 0.5,
            rect.min.y + (rect.height() - th) * 0.5,
        );
        self.text(tp, label, [1.0, 1.0, 1.0, 1.0], scale);

        held && self.input.mouse_released && hovered
    }

    fn push_solid(&mut self, rect: Rect, color: [f32; 4]) {
        let u = self.white_uv;
        self.quads.push(HudQuad {
            rect,
            uv_min: u,
            uv_max: u,
            color,
        });
    }
}
