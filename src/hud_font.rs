//! 8×8 bitmap font (KeyRus / CP866) — ASCII + Cyrillic.
//!
//! Embedded `hud_08x08.fnt` is freeware (c) Dmitry Gurtyak, via rusq/fontpic.

pub const GLYPH_W: u32 = 8;
pub const GLYPH_H: u32 = 8;
const COLS: u32 = 16;
const ROWS: u32 = 16;
const GLYPH_COUNT: usize = 256;

static FONT: &[u8; 2048] = include_bytes!("hud_08x08.fnt");

/// Map Unicode → CP866 codepoint used by the bitmap font.
pub fn char_index(c: char) -> u8 {
    match c {
        '\u{0000}'..='\u{007f}' => c as u8,
        // А..Я
        '\u{0410}'..='\u{042F}' => (c as u32 - 0x0410 + 0x80) as u8,
        // а..п
        '\u{0430}'..='\u{043F}' => (c as u32 - 0x0430 + 0xA0) as u8,
        // р..я
        '\u{0440}'..='\u{044F}' => (c as u32 - 0x0440 + 0xE0) as u8,
        'Ё' => 0xF0,
        'ё' => 0xF1,
        _ => b'?',
    }
}

fn glyph_rows(idx: u8) -> [u8; 8] {
    let off = idx as usize * 8;
    let mut rows = [0u8; 8];
    rows.copy_from_slice(&FONT[off..off + 8]);
    rows
}

/// Bake an R8 atlas: 16×16 glyphs (full CP866 page).
pub fn bake_atlas() -> (Vec<u8>, u32, u32) {
    let w = COLS * GLYPH_W;
    let h = ROWS * GLYPH_H;
    let mut px = vec![0u8; (w * h) as usize];
    for i in 0..GLYPH_COUNT {
        let gx = (i as u32) % COLS;
        let gy = (i as u32) / COLS;
        let rows8 = glyph_rows(i as u8);
        for (row, bits) in rows8.iter().enumerate() {
            for col in 0..8u32 {
                // VGA / KeyRus: bit 7 = leftmost pixel
                if bits & (0x80 >> col) != 0 {
                    let x = gx * GLYPH_W + col;
                    let y = gy * GLYPH_H + row as u32;
                    px[(y * w + x) as usize] = 255;
                }
            }
        }
    }
    // Dedicated white texel for solid fills (bottom-right).
    px[((h - 1) * w + (w - 1)) as usize] = 255;
    (px, w, h)
}

pub fn glyph_uv(idx: u8, atlas_w: u32, atlas_h: u32) -> ([f32; 2], [f32; 2]) {
    let i = idx as u32;
    let gx = i % COLS;
    let gy = i / COLS;
    let u0 = (gx * GLYPH_W) as f32 / atlas_w as f32;
    let v0 = (gy * GLYPH_H) as f32 / atlas_h as f32;
    let u1 = ((gx + 1) * GLYPH_W) as f32 / atlas_w as f32;
    let v1 = ((gy + 1) * GLYPH_H) as f32 / atlas_h as f32;
    ([u0, v0], [u1, v1])
}

pub fn white_uv(atlas_w: u32, atlas_h: u32) -> [f32; 2] {
    [
        (atlas_w - 1) as f32 / atlas_w as f32 + 0.5 / atlas_w as f32,
        (atlas_h - 1) as f32 / atlas_h as f32 + 0.5 / atlas_h as f32,
    ]
}
