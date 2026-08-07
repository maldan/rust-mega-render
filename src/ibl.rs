//! Equirect HDR/EXR: full-res sharp map + smaller pre-blurred copies for roughness.

use std::path::Path;

/// Max width of pre-blurred roughness maps (height follows aspect).
pub const BLUR_MAX_WIDTH: u32 = 1024;
/// Number of progressively blurred maps (all ≤ [`BLUR_MAX_WIDTH`]).
pub const BLUR_LEVELS: u32 = 4;

#[derive(Clone)]
pub struct EquirectHdr {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<[f32; 3]>,
}

pub struct EnvMaps {
    /// Full-resolution source (skybox + sharp reflections).
    pub sharp: EquirectHdr,
    /// Progressively blurred copies, same size, width ≤ [`BLUR_MAX_WIDTH`].
    pub blurred: Vec<EquirectHdr>,
}

pub fn load_equirect_hdr(path: impl AsRef<Path>) -> Result<EquirectHdr, String> {
    let path = path.as_ref();
    let img = image::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let rgba = img.to_rgba32f();
    let width = rgba.width();
    let height = rgba.height();
    if width < 2 || height < 2 {
        return Err("HDRI too small".into());
    }
    let rgb = rgba
        .pixels()
        .map(|p| [p.0[0], p.0[1], p.0[2]])
        .collect();
    Ok(EquirectHdr { width, height, rgb })
}

/// Keep original sharp; build [`BLUR_LEVELS`] blurred maps at ≤1024.
pub fn prepare_env_maps(sharp: EquirectHdr) -> EnvMaps {
    let base = if sharp.width > BLUR_MAX_WIDTH {
        resize_equirect(&sharp, BLUR_MAX_WIDTH)
    } else {
        EquirectHdr {
            width: sharp.width,
            height: sharp.height,
            rgb: sharp.rgb.clone(),
        }
    };

    // Pixel radii on the ≤1024 map — real blur, not mip shrink.
    let radii: [u32; BLUR_LEVELS as usize] = [4, 14, 36, 80];
    let mut blurred = Vec::with_capacity(BLUR_LEVELS as usize);
    for &radius in &radii {
        blurred.push(blur_equirect(&base, radius));
    }

    EnvMaps { sharp, blurred }
}

fn resize_equirect(src: &EquirectHdr, new_w: u32) -> EquirectHdr {
    let new_h = ((src.height as f32) * (new_w as f32 / src.width as f32))
        .round()
        .max(1.0) as u32;
    let mut rgb = vec![[0.0; 3]; (new_w * new_h) as usize];
    for y in 0..new_h {
        for x in 0..new_w {
            let u = (x as f32 + 0.5) / new_w as f32;
            let v = (y as f32 + 0.5) / new_h as f32;
            rgb[(y * new_w + x) as usize] = sample_equirect(src, u, v);
        }
    }
    EquirectHdr {
        width: new_w,
        height: new_h,
        rgb,
    }
}

/// Separable box blur ≈ soft gaussian (3 passes). U wraps, V clamps.
fn blur_equirect(src: &EquirectHdr, radius: u32) -> EquirectHdr {
    let r = radius.max(1);
    let mut img = src.rgb.clone();
    for _ in 0..3 {
        img = box_blur_horizontal(&img, src.width, src.height, r);
        img = box_blur_vertical(&img, src.width, src.height, r);
    }
    EquirectHdr {
        width: src.width,
        height: src.height,
        rgb: img,
    }
}

fn box_blur_horizontal(src: &[[f32; 3]], w: u32, h: u32, radius: u32) -> Vec<[f32; 3]> {
    let w = w as usize;
    let h = h as usize;
    let r = radius as usize;
    let inv = 1.0 / (2 * r + 1) as f32;
    let mut out = vec![[0.0; 3]; w * h];
    for y in 0..h {
        let row = y * w;
        // Running sum seed
        let mut sum = [0.0f32; 3];
        for k in 0..=(2 * r) {
            let x = (k as isize - r as isize).rem_euclid(w as isize) as usize;
            let c = src[row + x];
            sum[0] += c[0];
            sum[1] += c[1];
            sum[2] += c[2];
        }
        for x in 0..w {
            out[row + x] = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
            let x_out = (x as isize - r as isize).rem_euclid(w as isize) as usize;
            let x_in = (x as isize + r as isize + 1).rem_euclid(w as isize) as usize;
            let c_out = src[row + x_out];
            let c_in = src[row + x_in];
            sum[0] += c_in[0] - c_out[0];
            sum[1] += c_in[1] - c_out[1];
            sum[2] += c_in[2] - c_out[2];
        }
    }
    out
}

fn box_blur_vertical(src: &[[f32; 3]], w: u32, h: u32, radius: u32) -> Vec<[f32; 3]> {
    let w = w as usize;
    let h = h as usize;
    let r = radius as usize;
    let inv = 1.0 / (2 * r + 1) as f32;
    let mut out = vec![[0.0; 3]; w * h];
    for x in 0..w {
        let mut sum = [0.0f32; 3];
        for k in 0..=(2 * r) {
            let y = (k as isize - r as isize).clamp(0, h as isize - 1) as usize;
            let c = src[y * w + x];
            sum[0] += c[0];
            sum[1] += c[1];
            sum[2] += c[2];
        }
        for y in 0..h {
            out[y * w + x] = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
            let y_out = (y as isize - r as isize).clamp(0, h as isize - 1) as usize;
            let y_in = (y as isize + r as isize + 1).clamp(0, h as isize - 1) as usize;
            let c_out = src[y_out * w + x];
            let c_in = src[y_in * w + x];
            sum[0] += c_in[0] - c_out[0];
            sum[1] += c_in[1] - c_out[1];
            sum[2] += c_in[2] - c_out[2];
        }
    }
    out
}

pub fn sample_equirect(src: &EquirectHdr, u: f32, v: f32) -> [f32; 3] {
    let u = u.rem_euclid(1.0);
    let v = v.clamp(0.0, 1.0);
    let x = u * (src.width as f32 - 1.0);
    let y = v * (src.height as f32 - 1.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1) % src.width;
    let y1 = (y0 + 1).min(src.height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let c00 = src.rgb[(y0 * src.width + x0) as usize];
    let c10 = src.rgb[(y0 * src.width + x1) as usize];
    let c01 = src.rgb[(y1 * src.width + x0) as usize];
    let c11 = src.rgb[(y1 * src.width + x1) as usize];
    [
        c00[0] + (c10[0] - c00[0]) * tx + (c01[0] - c00[0]) * ty + (c00[0] - c10[0] - c01[0] + c11[0]) * tx * ty,
        c00[1] + (c10[1] - c00[1]) * tx + (c01[1] - c00[1]) * ty + (c00[1] - c10[1] - c01[1] + c11[1]) * tx * ty,
        c00[2] + (c10[2] - c00[2]) * tx + (c01[2] - c00[2]) * ty + (c00[2] - c10[2] - c01[2] + c11[2]) * tx * ty,
    ]
}
