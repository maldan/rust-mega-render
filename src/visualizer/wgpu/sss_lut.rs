//! Pre-integrated diffuse SSS LUT (Jimenez / GPU Pro 2 style).
//!
//! X = N·L in [-1, 1] → u in [0, 1]
//! Y = curvature in [0, 1] → thinner features toward the top

use wgpu::util::DeviceExt;

const LUT_W: u32 = 128;
const LUT_H: u32 = 128;

pub struct GpuSssLut {
    pub view: wgpu::TextureView,
    pub samp: wgpu::Sampler,
    _tex: wgpu::Texture,
}

impl GpuSssLut {
    pub fn bake(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let pixels = bake_rgba8(LUT_W, LUT_H);
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("sss_lut"),
                size: wgpu::Extent3d {
                    width: LUT_W,
                    height: LUT_H,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &pixels,
        );
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sss_lut_samp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            view: tex.create_view(&Default::default()),
            samp,
            _tex: tex,
        }
    }
}

fn gaussian(variance: f32, r: f32) -> f32 {
    (-r * r / (2.0 * variance)).exp() / (std::f32::consts::TAU * variance).sqrt()
}

/// Skin diffusion profile (Separable SSS / Jimenez), `r` in mm.
fn skin_profile(r: f32) -> [f32; 3] {
    let g = |v: f32| gaussian(v, r);
    [
        0.233 * g(0.0064)
            + 0.100 * g(0.0484)
            + 0.118 * g(0.187)
            + 0.113 * g(0.567)
            + 0.358 * g(1.99)
            + 0.078 * g(7.41),
        0.455 * g(0.0064)
            + 0.336 * g(0.0484)
            + 0.198 * g(0.187)
            + 0.007 * g(0.567)
            + 0.004 * g(1.99),
        0.649 * g(0.0064) + 0.344 * g(0.0484) + 0.007 * g(0.567),
    ]
}

fn bake_rgba8(width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    const STEPS: i32 = 64;
    let pi = std::f32::consts::PI;

    for y in 0..height {
        // Curvature 0 → large radius (hard Lambert), 1 → small radius (strong wrap).
        let curv = (y as f32 + 0.5) / height as f32;
        let radius_mm = 1.0 / (curv * 0.25 + 0.004).max(0.004);

        for x in 0..width {
            let ndotl = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let theta = ndotl.clamp(-1.0, 1.0).acos();

            let mut rgb = [0.0f32; 3];
            let mut norm = [0.0f32; 3];
            for i in 0..=STEPS {
                let t = i as f32 / STEPS as f32;
                let x_ang = -pi + 2.0 * pi * t;
                let ndotl_i = (theta + x_ang).cos().max(0.0);
                let dist = (2.0 * radius_mm * (x_ang * 0.5).sin().abs()).max(0.0);
                let p = skin_profile(dist);
                for c in 0..3 {
                    rgb[c] += p[c] * ndotl_i;
                    norm[c] += p[c];
                }
            }

            let r = (rgb[0] / norm[0].max(1e-6)).clamp(0.0, 1.0);
            let g = (rgb[1] / norm[1].max(1e-6)).clamp(0.0, 1.0);
            let b = (rgb[2] / norm[2].max(1e-6)).clamp(0.0, 1.0);
            let i = ((y * width + x) * 4) as usize;
            out[i] = (r * 255.0) as u8;
            out[i + 1] = (g * 255.0) as u8;
            out[i + 2] = (b * 255.0) as u8;
            out[i + 3] = 255;
        }
    }
    out
}
