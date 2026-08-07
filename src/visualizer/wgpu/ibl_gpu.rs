//! GPU upload for sharp equirect + blurred texture2d_array (roughness).

use crate::ibl::{self, EquirectHdr, EnvMaps, BLUR_LEVELS};
use half::f16;

pub struct GpuIbl {
    pub sharp_view: wgpu::TextureView,
    pub blur_view: wgpu::TextureView,
    pub samp: wgpu::Sampler,
    /// Number of blur array layers (for shader blend).
    pub blur_levels: f32,
    pub loaded: bool,
    _sharp: wgpu::Texture,
    _blur: wgpu::Texture,
}

impl GpuIbl {
    pub fn black(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let pixel = EquirectHdr {
            width: 1,
            height: 1,
            rgb: vec![[0.05; 3]],
        };
        let maps = EnvMaps {
            sharp: pixel.clone(),
            blurred: vec![pixel; BLUR_LEVELS as usize],
        };
        Self::from_maps(device, queue, &maps, false)
    }

    pub fn from_maps(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        maps: &EnvMaps,
        loaded: bool,
    ) -> Self {
        let sharp = upload_equirect_2d(device, queue, &maps.sharp, "env_sharp");
        let blur = upload_equirect_array(device, queue, &maps.blurred, "env_blur");
        Self {
            sharp_view: sharp.create_view(&Default::default()),
            blur_view: blur.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
            samp: equirect_sampler(device),
            blur_levels: maps.blurred.len() as f32,
            loaded,
            _sharp: sharp,
            _blur: blur,
        }
    }
}

fn equirect_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("env_samp"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    })
}

fn f16_rgba(rgb: [f32; 3]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&f16::from_f32(rgb[0]).to_le_bytes());
    out[2..4].copy_from_slice(&f16::from_f32(rgb[1]).to_le_bytes());
    out[4..6].copy_from_slice(&f16::from_f32(rgb[2]).to_le_bytes());
    out[6..8].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
    out
}

fn align256(n: u32) -> u32 {
    (n + 255) & !255
}

fn padded_rgba16_rows(width: u32, height: u32, rgb: &[[f32; 3]]) -> (Vec<u8>, u32) {
    let unpadded = width * 8;
    let bpr = align256(unpadded);
    let mut bytes = vec![0u8; (bpr * height) as usize];
    for y in 0..height {
        let row = (y * bpr) as usize;
        for x in 0..width {
            let px = f16_rgba(rgb[(y * width + x) as usize]);
            let o = row + (x as usize) * 8;
            bytes[o..o + 8].copy_from_slice(&px);
        }
    }
    (bytes, bpr)
}

fn upload_equirect_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &EquirectHdr,
    label: &str,
) -> wgpu::Texture {
    let (bytes, bpr) = padded_rgba16_rows(img.width, img.height, &img.rgb);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bpr),
            rows_per_image: Some(img.height),
        },
        wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

fn upload_equirect_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: &[EquirectHdr],
    label: &str,
) -> wgpu::Texture {
    assert!(!layers.is_empty());
    let w = layers[0].width;
    let h = layers[0].height;
    for layer in layers {
        assert_eq!(layer.width, w);
        assert_eq!(layer.height, h);
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: layers.len() as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, layer) in layers.iter().enumerate() {
        let (bytes, bpr) = padded_rgba16_rows(layer.width, layer.height, &layer.rgb);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: i as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(layer.height),
            },
            wgpu::Extent3d {
                width: layer.width,
                height: layer.height,
                depth_or_array_layers: 1,
            },
        );
    }
    tex
}

pub fn load_cpu(path: impl AsRef<std::path::Path>) -> Result<EnvMaps, String> {
    let path = path.as_ref();
    let sharp = ibl::load_equirect_hdr(path)?;
    eprintln!(
        "Env map: {}x{} sharp — building {} blurred ≤{}px maps...",
        sharp.width,
        sharp.height,
        BLUR_LEVELS,
        ibl::BLUR_MAX_WIDTH
    );
    let maps = ibl::prepare_env_maps(sharp);
    eprintln!(
        "Env map: blur layers {}x{} x{}",
        maps.blurred[0].width,
        maps.blurred[0].height,
        maps.blurred.len()
    );
    Ok(maps)
}

pub fn load_and_upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: impl AsRef<std::path::Path>,
) -> Result<GpuIbl, String> {
    Ok(GpuIbl::from_maps(device, queue, &load_cpu(path)?, true))
}
