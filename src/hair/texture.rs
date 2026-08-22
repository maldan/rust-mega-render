use super::curve::{sample_gradient, HairColorStop};
use super::params::HairParams;
use glam::Vec3;

#[derive(Clone)]
pub struct HairLayerBake {
    pub gap: f32,
    pub color_stops: Vec<HairColorStop>,
}

pub struct HairMaps {
    pub width: u32,
    pub height: u32,
    pub albedos: Vec<Vec<u8>>,
    pub roughness: Vec<u8>,
    pub normal: Vec<u8>,
}

pub fn bake_hair_maps(params: &HairParams, layers: &[HairLayerBake]) -> HairMaps {
    let (w, h) = params.tex_size();
    let strands = build_fiber_strands(params, w);
    let (cov, shade_map, rough_map) = rasterize_fibers(&strands, w, h);
    let gap_shade = params.fiber_gap_shade.max(0.0);
    let blur = params.fiber_blur.max(0.0);

    let layers: Vec<HairLayerBake> = if layers.is_empty() {
        vec![HairLayerBake {
            gap: params.fiber_gap,
            color_stops: params.color_stops.clone(),
        }]
    } else {
        layers.to_vec()
    };

    let mut albedos = Vec::with_capacity(layers.len());
    for layer in &layers {
        let gap_alpha = layer.gap.max(0.0);
        let stops = if layer.color_stops.is_empty() {
            &params.color_stops
        } else {
            &layer.color_stops
        };
        let mut albedo = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            let t = y as f32 / (h - 1) as f32;
            let c = sample_gradient(stops, t);
            let row = (y * w) as usize;
            for x in 0..w {
                let i = row + x as usize;
                let a = cov[i];
                let sh = if a > 1e-4 { shade_map[i] } else { gap_shade };
                let shade = sh * a + gap_shade * (1.0 - a);
                let alpha = a + gap_alpha * (1.0 - a);
                let pi = i * 4;
                albedo[pi] = (c[0] * shade * 255.0).clamp(0.0, 255.0).round() as u8;
                albedo[pi + 1] = (c[1] * shade * 255.0).clamp(0.0, 255.0).round() as u8;
                albedo[pi + 2] = (c[2] * shade * 255.0).clamp(0.0, 255.0).round() as u8;
                albedo[pi + 3] = (alpha * 255.0).clamp(0.0, 255.0).round() as u8;
            }
        }
        if blur > 0.05 {
            blur_rgba_separable(&mut albedo, w, h, blur);
        }
        albedos.push(albedo);
    }

    HairMaps {
        width: w,
        height: h,
        albedos,
        roughness: bake_roughness_rgba(params, w, h, &cov, &rough_map),
        normal: bake_normal_rgba(params, w, h, &strands, &cov),
    }
}

fn bake_roughness_rgba(
    params: &HairParams,
    w: u32,
    h: u32,
    cov: &[f32],
    rough: &[f32],
) -> Vec<u8> {
    let variance = params.rough_variance.max(0.0);
    let n = (w * h) as usize;
    let mut rgba = vec![0u8; n * 4];
    if variance <= 1e-4 {
        for i in 0..n {
            let pi = i * 4;
            rgba[pi] = 255;
            rgba[pi + 1] = 255;
            rgba[pi + 2] = 0;
            rgba[pi + 3] = 255;
        }
        return rgba;
    }
    for i in 0..n {
        let g = if cov[i] > 0.02 {
            (1.0 - variance * (1.0 - rough[i])).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let pi = i * 4;
        rgba[pi] = 255;
        rgba[pi + 1] = (g * 255.0).round() as u8;
        rgba[pi + 2] = 0;
        rgba[pi + 3] = 255;
    }
    rgba
}

fn bake_normal_rgba(
    params: &HairParams,
    w: u32,
    h: u32,
    strands: &[FiberStrand],
    cov: &[f32],
) -> Vec<u8> {
    let strength = params.normal_strength.max(0.0);
    let n = (w * h) as usize;
    let mut rgba = vec![0u8; n * 4];
    if strength <= 1e-4 {
        for i in 0..n {
            let pi = i * 4;
            rgba[pi] = 128;
            rgba[pi + 1] = 128;
            rgba[pi + 2] = 255;
            rgba[pi + 3] = 255;
        }
        return rgba;
    }
    let nx_map = fiber_normal_x(strands, w, h, strength);
    for y in 0..h {
        let t = y as f32 / (h - 1).max(1) as f32;
        let row = (y * w) as usize;
        for x in 0..w {
            let i = row + x as usize;
            let a = cov[i];
            let nx = nx_map[i];
            let ny = ((t * 190.0 + x as f32 * 0.17).sin() * 0.08 * strength) * a;
            let nz = (1.0 - (nx * nx + ny * ny)).max(0.05).sqrt();
            let nrm = Vec3::new(nx, ny, nz).normalize_or_zero();
            let flat = Vec3::Z;
            let blended = flat.lerp(nrm, a).normalize_or_zero();
            let pi = i * 4;
            rgba[pi] = ((blended.x * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0).round() as u8;
            rgba[pi + 1] = ((blended.y * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0).round() as u8;
            rgba[pi + 2] = ((blended.z * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0).round() as u8;
            rgba[pi + 3] = 255;
        }
    }
    rgba
}

fn blur_rgba_separable(rgba: &mut [u8], w: u32, h: u32, radius: f32) {
    let r = radius.round().clamp(1.0, 32.0) as i32;
    let n = (w * h) as usize;
    if rgba.len() < n * 4 || r < 1 {
        return;
    }
    let mut tmp = vec![0u8; n * 4];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = [0u32; 4];
            let mut cnt = 0u32;
            for dx in -r..=r {
                let xx = (x + dx).clamp(0, w as i32 - 1) as u32;
                let i = ((y as u32 * w + xx) * 4) as usize;
                for c in 0..4 {
                    acc[c] += rgba[i + c] as u32;
                }
                cnt += 1;
            }
            let o = ((y as u32 * w + x as u32) * 4) as usize;
            for c in 0..4 {
                tmp[o + c] = ((acc[c] + cnt / 2) / cnt) as u8;
            }
        }
    }
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = [0u32; 4];
            let mut cnt = 0u32;
            for dy in -r..=r {
                let yy = (y + dy).clamp(0, h as i32 - 1) as u32;
                let i = ((yy * w + x as u32) * 4) as usize;
                for c in 0..4 {
                    acc[c] += tmp[i + c] as u32;
                }
                cnt += 1;
            }
            let o = ((y as u32 * w + x as u32) * 4) as usize;
            for c in 0..4 {
                rgba[o + c] = ((acc[c] + cnt / 2) / cnt) as u8;
            }
        }
    }
}

fn hash01(n: u32) -> f32 {
    let mut x = n.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    x ^= x >> 22;
    (x as f32) * (1.0 / u32::MAX as f32)
}

fn smoothstep01(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

struct FiberStrand {
    u0: f32,
    half_w: f32,
    t0: f32,
    t1: f32,
    shade: f32,
    rough: f32,
    a1: f32,
    f1: f32,
    p1: f32,
    a2: f32,
    f2: f32,
    p2: f32,
    a3: f32,
    f3: f32,
    p3: f32,
}

impl FiberStrand {
    fn u_at(&self, t: f32) -> f32 {
        self.u0
            + self.a1 * (t * self.f1 + self.p1).sin()
            + self.a2 * (t * self.f2 + self.p2).sin()
            + self.a3 * (t * self.f3 + self.p3).sin()
    }
}

fn build_fiber_strands(params: &HairParams, tex_w: u32) -> Vec<FiberStrand> {
    let n_primary = params.card_strands.max(1) as usize;
    if n_primary <= 1 {
        return Vec::new();
    }
    let waviness = params.fiber_waviness.max(0.0);
    let width_var = params.fiber_width_variance.max(0.0);
    let overlap = params.fiber_overlap.max(0.0);
    let shade_var = params.fiber_shade_variance.max(0.0);
    let seed = params.seed;
    let px = 1.0 / tex_w.max(1) as f32;
    let base_hw = (1.15 * px).clamp(0.0025, 0.01);
    let n_clumps = ((n_primary as f32 / 5.0).round() as usize).clamp(2, 12);
    let n_fly = ((n_primary as f32) * overlap * 1.35).round() as usize;
    let total = n_primary + n_fly;
    let mut out = Vec::with_capacity(total);

    for i in 0..total {
        let is_fly = i >= n_primary;
        let id = seed
            .wrapping_mul(0x9E3779B9)
            .wrapping_add((i as u32).wrapping_mul(0x85EBCA6B))
            .wrapping_add(if is_fly { 0xC2B2AE35 } else { 0x27D4EB2F });
        let h0 = hash01(id);
        let h1 = hash01(id.wrapping_mul(3).wrapping_add(1));
        let h2 = hash01(id.wrapping_mul(5).wrapping_add(2));
        let h3 = hash01(id.wrapping_mul(7).wrapping_add(3));
        let h4 = hash01(id.wrapping_mul(11).wrapping_add(4));
        let h5 = hash01(id.wrapping_mul(13).wrapping_add(5));
        let h6 = hash01(id.wrapping_mul(17).wrapping_add(6));

        let clump = if is_fly {
            ((h0 * n_clumps as f32).floor() as usize).min(n_clumps - 1)
        } else {
            (i * n_clumps) / n_primary.max(1)
        };
        let clump_u = (clump as f32 + 0.5) / n_clumps as f32;
        let spread = 0.028 + 0.04 * (1.0 - overlap * 0.5);
        let gauss = ((h1 - 0.5) + (h2 - 0.5) * 0.55) * spread;
        let edge_pad = 0.02;
        let u0 = (clump_u + gauss).clamp(edge_pad, 1.0 - edge_pad);

        let thin = if is_fly {
            0.45 + h3 * 0.4
        } else {
            0.75 + h3 * 0.55
        };
        let var_mul = 1.0 + (h4 - 0.5) * 2.0 * width_var;
        let half_w = (base_hw * thin * var_mul).clamp(0.6 * px, 3.2 * px);

        let t0 = if is_fly { h5 * 0.12 } else { h5 * 0.04 };
        let tip_cut = if is_fly {
            0.55 + h6 * 0.42
        } else {
            0.78 + h6 * 0.22
        };
        let t1 = tip_cut.clamp(t0 + 0.15, 1.0);

        let wave = waviness * if is_fly { 1.35 } else { 1.0 };
        let a1 = wave * (0.006 + h0 * 0.014);
        let a2 = wave * (0.003 + h1 * 0.008);
        let a3 = wave * (0.001 + h2 * 0.0035);

        let shade = (1.0 + (h3 - 0.5) * 2.0 * shade_var * 0.55).clamp(0.55, 1.35);
        let rough = h4;

        out.push(FiberStrand {
            u0,
            half_w,
            t0,
            t1,
            shade,
            rough,
            a1,
            f1: 2.2 + h5 * 5.5,
            p1: h0 * 6.2832,
            a2,
            f2: 6.0 + h6 * 11.0,
            p2: h1 * 6.2832,
            a3,
            f3: 18.0 + h2 * 28.0,
            p3: h3 * 6.2832,
        });
    }
    out
}

fn rasterize_fibers(strands: &[FiberStrand], w: u32, h: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (w * h) as usize;
    let mut cov = vec![0f32; n];
    let mut shade = vec![1f32; n];
    let mut rough = vec![1f32; n];
    if strands.is_empty() {
        cov.fill(1.0);
        return (cov, shade, rough);
    }
    let wf = w as f32;
    let hf = (h - 1).max(1) as f32;

    for s in strands {
        let y0 = ((s.t0 * hf).floor() as i32).clamp(0, h as i32 - 1);
        let y1 = ((s.t1 * hf).ceil() as i32).clamp(0, h as i32 - 1);
        let tip_fade = ((s.t1 - s.t0) * 0.18).clamp(0.03, 0.2);
        let root_fade = 0.02f32;
        for y in y0..=y1 {
            let t = y as f32 / hf;
            if t < s.t0 || t > s.t1 {
                continue;
            }
            let fade = smoothstep01(s.t0, s.t0 + root_fade, t)
                * (1.0 - smoothstep01(s.t1 - tip_fade, s.t1, t));
            let along = ((t - s.t0) / (s.t1 - s.t0).max(1e-4)).clamp(0.0, 1.0);
            let taper = 1.0 - 0.55 * along * along;
            let uc = s.u_at(t).clamp(0.0, 1.0);
            let xc = uc * (wf - 1.0);
            let hw_px = (s.half_w * taper * wf).max(0.45);
            let x0 = ((xc - hw_px - 1.25).floor() as i32).max(0);
            let x1 = ((xc + hw_px + 1.25).ceil() as i32).min(w as i32 - 1);
            for x in x0..=x1 {
                let d = (x as f32 - xc).abs() / hw_px;
                let core = 1.0 - smoothstep01(0.15, 1.0, d);
                let a = core * fade;
                if a <= 1e-4 {
                    continue;
                }
                let i = (y as u32 * w + x as u32) as usize;
                let old = cov[i];
                let out_a = (a + old * (1.0 - a)).min(1.0);
                if out_a > 1e-5 {
                    shade[i] = (s.shade * a + shade[i] * old * (1.0 - a)) / out_a;
                    rough[i] = (s.rough * a + rough[i] * old * (1.0 - a)) / out_a;
                }
                cov[i] = out_a;
            }
        }
    }
    (cov, shade, rough)
}

fn fiber_normal_x(strands: &[FiberStrand], w: u32, h: u32, strength: f32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut nx = vec![0f32; n];
    if strands.is_empty() || strength <= 1e-4 {
        return nx;
    }
    let wf = w as f32;
    let hf = (h - 1).max(1) as f32;
    let mut best = vec![f32::MAX; n];
    for s in strands {
        let y0 = ((s.t0 * hf).floor() as i32).clamp(0, h as i32 - 1);
        let y1 = ((s.t1 * hf).ceil() as i32).clamp(0, h as i32 - 1);
        for y in y0..=y1 {
            let t = y as f32 / hf;
            if t < s.t0 || t > s.t1 {
                continue;
            }
            let along = ((t - s.t0) / (s.t1 - s.t0).max(1e-4)).clamp(0.0, 1.0);
            let taper = 1.0 - 0.55 * along * along;
            let uc = s.u_at(t).clamp(0.0, 1.0);
            let xc = uc * (wf - 1.0);
            let hw_px = (s.half_w * taper * wf).max(0.45);
            let x0 = ((xc - hw_px - 1.0).floor() as i32).max(0);
            let x1 = ((xc + hw_px + 1.0).ceil() as i32).min(w as i32 - 1);
            for x in x0..=x1 {
                let dx = x as f32 - xc;
                let d = dx.abs() / hw_px;
                if d > 1.15 {
                    continue;
                }
                let i = (y as u32 * w + x as u32) as usize;
                if d < best[i] {
                    best[i] = d;
                    nx[i] = (dx / hw_px).clamp(-1.0, 1.0) * strength;
                }
            }
        }
    }
    nx
}
