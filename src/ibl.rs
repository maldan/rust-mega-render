//! CPU IBL bake: equirect mips (diffuse/specular blur proxy) + BRDF LUT.

use glam::{Vec2, Vec3};
use std::f32::consts::PI;
use std::path::Path;

pub const EQUIRECT_WIDTH: u32 = 1024;
pub const BRDF_SIZE: u32 = 256;

pub struct EquirectHdr {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<[f32; 3]>,
}

pub struct BakedIbl {
    pub equirect_mips: Vec<EquirectHdr>,
    pub brdf: Vec<[f32; 2]>,
    pub brdf_size: u32,
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

pub fn bake_ibl(src: &EquirectHdr) -> BakedIbl {
    let base = if src.width > EQUIRECT_WIDTH {
        resize_equirect(src, EQUIRECT_WIDTH)
    } else {
        EquirectHdr {
            width: src.width,
            height: src.height,
            rgb: src.rgb.clone(),
        }
    };
    let equirect_mips = build_equirect_mips(base);
    let brdf = bake_brdf_lut(BRDF_SIZE);
    BakedIbl {
        equirect_mips,
        brdf,
        brdf_size: BRDF_SIZE,
    }
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

fn build_equirect_mips(base: EquirectHdr) -> Vec<EquirectHdr> {
    let mut mips = vec![base];
    while mips.last().unwrap().width > 4 {
        let prev = mips.last().unwrap();
        let w = (prev.width / 2).max(1);
        let h = (prev.height / 2).max(1);
        let mut rgb = vec![[0.0; 3]; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let v = (y as f32 + 0.5) / h as f32;
                // 4-tap box in source.
                let du = 0.25 / w as f32;
                let dv = 0.25 / h as f32;
                let c0 = sample_equirect(prev, u - du, v - dv);
                let c1 = sample_equirect(prev, u + du, v - dv);
                let c2 = sample_equirect(prev, u - du, v + dv);
                let c3 = sample_equirect(prev, u + du, v + dv);
                rgb[(y * w + x) as usize] = [
                    (c0[0] + c1[0] + c2[0] + c3[0]) * 0.25,
                    (c0[1] + c1[1] + c2[1] + c3[1]) * 0.25,
                    (c0[2] + c1[2] + c2[2] + c3[2]) * 0.25,
                ];
            }
        }
        mips.push(EquirectHdr {
            width: w,
            height: h,
            rgb,
        });
    }
    mips
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

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_3064e-10
}

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(i as f32 / n as f32, radical_inverse_vdc(i))
}

fn importance_sample_ggx(xi: Vec2, n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    let up = if n.z.abs() < 0.999 {
        Vec3::Z
    } else {
        Vec3::X
    };
    let tangent = up.cross(n).normalize_or_zero();
    let bitangent = n.cross(tangent);
    (tangent * h.x + bitangent * h.y + n * h.z).normalize_or_zero()
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0;
    n_dot_v / (n_dot_v * (1.0 - k) + k)
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness)
}

fn integrate_brdf(n_dot_v: f32, roughness: f32) -> [f32; 2] {
    const SAMPLE_COUNT: u32 = 64;
    let v = Vec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let n = Vec3::Z;
    let mut a = 0.0_f32;
    let mut b = 0.0_f32;
    for i in 0..SAMPLE_COUNT {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = (2.0 * v.dot(h) * h - v).normalize_or_zero();
        let n_dot_l = l.z.max(0.0);
        let n_dot_h = h.z.max(0.0);
        let v_dot_h = v.dot(h).max(0.0);
        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(1e-4);
            let fc = (1.0 - v_dot_h).powi(5);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    let inv = 1.0 / SAMPLE_COUNT as f32;
    [a * inv, b * inv]
}

fn bake_brdf_lut(size: u32) -> Vec<[f32; 2]> {
    let mut out = vec![[0.0; 2]; (size * size) as usize];
    for y in 0..size {
        for x in 0..size {
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let roughness = (y as f32 + 0.5) / size as f32;
            out[(y * size + x) as usize] =
                integrate_brdf(n_dot_v.max(0.001), roughness.clamp(0.001, 1.0));
        }
    }
    out
}
