//! GPU upload / bind helpers for baked IBL.

use crate::ibl::{self, BakedIbl};
use half::f16;

pub struct GpuIbl {
    pub equirect_view: wgpu::TextureView,
    pub brdf_view: wgpu::TextureView,
    pub equirect_samp: wgpu::Sampler,
    pub clamp_samp: wgpu::Sampler,
    pub max_mip: f32,
    pub enabled: bool,
    _equirect: wgpu::Texture,
    _brdf: wgpu::Texture,
}

impl GpuIbl {
    pub fn black(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let equirect = solid_equirect(device, queue, [0.05; 3]);
        let brdf = solid_brdf(device, queue, [1.0, 0.0]);
        Self {
            equirect_view: equirect.create_view(&Default::default()),
            brdf_view: brdf.create_view(&Default::default()),
            equirect_samp: equirect_sampler(device),
            clamp_samp: clamp_sampler(device),
            max_mip: 0.0,
            enabled: false,
            _equirect: equirect,
            _brdf: brdf,
        }
    }

    pub fn from_baked(device: &wgpu::Device, queue: &wgpu::Queue, baked: &BakedIbl) -> Self {
        let equirect = upload_equirect_mips(device, queue, &baked.equirect_mips);
        let brdf = upload_brdf(device, queue, baked.brdf_size, &baked.brdf);
        let max_mip = (baked.equirect_mips.len().saturating_sub(1)) as f32;
        Self {
            equirect_view: equirect.create_view(&Default::default()),
            brdf_view: brdf.create_view(&Default::default()),
            equirect_samp: equirect_sampler(device),
            clamp_samp: clamp_sampler(device),
            max_mip,
            enabled: true,
            _equirect: equirect,
            _brdf: brdf,
        }
    }
}

fn equirect_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ibl_equirect_samp"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    })
}

fn clamp_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ibl_clamp_samp"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
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

fn upload_equirect_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mips: &[ibl::EquirectHdr],
) -> wgpu::Texture {
    let base = &mips[0];
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ibl_equirect"),
        size: wgpu::Extent3d {
            width: base.width,
            height: base.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, mip) in mips.iter().enumerate() {
        let (bytes, bpr) = padded_rgba16_rows(mip.width, mip.height, &mip.rgb);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(mip.height),
            },
            wgpu::Extent3d {
                width: mip.width,
                height: mip.height,
                depth_or_array_layers: 1,
            },
        );
    }
    tex
}

fn upload_brdf(device: &wgpu::Device, queue: &wgpu::Queue, size: u32, data: &[[f32; 2]]) -> wgpu::Texture {
    let rgb: Vec<[f32; 3]> = data.iter().map(|c| [c[0], c[1], 0.0]).collect();
    let (bytes, bpr) = padded_rgba16_rows(size, size, &rgb);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ibl_brdf"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
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
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    tex
}

fn solid_equirect(device: &wgpu::Device, queue: &wgpu::Queue, rgb: [f32; 3]) -> wgpu::Texture {
    let img = ibl::EquirectHdr {
        width: 1,
        height: 1,
        rgb: vec![rgb],
    };
    upload_equirect_mips(device, queue, &[img])
}

fn solid_brdf(device: &wgpu::Device, queue: &wgpu::Queue, rg: [f32; 2]) -> wgpu::Texture {
    upload_brdf(device, queue, 1, &[rg])
}

pub fn load_and_upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: impl AsRef<std::path::Path>,
) -> Result<GpuIbl, String> {
    let equirect = ibl::load_equirect_hdr(path)?;
    eprintln!(
        "IBL: preparing {}x{} → equirect {}px + mips + BRDF LUT...",
        equirect.width,
        equirect.height,
        ibl::EQUIRECT_WIDTH
    );
    let baked = ibl::bake_ibl(&equirect);
    Ok(GpuIbl::from_baked(device, queue, &baked))
}
