//! Screen-space HUD overlay pass (after post / present).

use bytemuck::{Pod, Zeroable};

use crate::hud::{Hud, HudLine, HudQuad};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudUniforms {
    screen: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

pub struct HudPass {
    pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    atlas_tex: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    atlas_uploaded: bool,
    /// Kept alive across frames — a per-draw VBO is dropped before `submit`
    /// and can come back as `hud_vbo is invalid` under GPU load (unwrap/paint).
    quad_vbo: Option<wgpu::Buffer>,
    quad_cap: u64,
    line_vbo: Option<wgpu::Buffer>,
    line_cap: u64,
}

impl HudPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("hud.wgsl"));
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud_uniform"),
            size: std::mem::size_of::<HudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Placeholder 1×1 white until first scene.hud atlas upload.
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud_atlas"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 1×1: omit bytes_per_row (Some(1) is not a multiple of 256).
                bytes_per_row: None,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let atlas_view = atlas_tex.create_view(&Default::default());
        let atlas_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud_bind"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud_bg"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&atlas_samp),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud_pl"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<HudVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x2,
                1 => Float32x2,
                2 => Float32x4,
            ],
        })];

        let color_target = [Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &color_target,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud_lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &color_target,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Keep sampler alive via bind group; store tex/view for reupload.
        // Sampler is owned by device graph through bind_group — fine.
        let _ = atlas_samp;

        Self {
            pipeline,
            line_pipeline,
            bind_group,
            uniform_buf,
            atlas_tex,
            atlas_view,
            atlas_uploaded: false,
            quad_vbo: None,
            quad_cap: 0,
            line_vbo: None,
            line_cap: 0,
        }
    }

    fn ensure_atlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, hud: &Hud) {
        if self.atlas_uploaded {
            return;
        }
        let (w, h) = hud.atlas_size();
        self.atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud_atlas"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (atlas_bytes, atlas_bpr) = pad_texel_rows(&hud.atlas_pixels, w, h, 1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_bpr),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.atlas_view = self.atlas_tex.create_view(&Default::default());

        // Rebuild bind group with new view (sampler + uniform unchanged — recreate layout entries).
        let atlas_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // Recreate bind group from pipeline's layout
        let bind_layout = self.pipeline.get_bind_group_layout(0);
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud_bg"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&atlas_samp),
                },
            ],
        });
        self.atlas_uploaded = true;
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hud: &Hud,
        target: &wgpu::TextureView,
        screen: (f32, f32),
    ) {
        let quads = hud.primitives();
        let lines = hud.lines();
        if quads.is_empty() && lines.is_empty() {
            return;
        }
        self.ensure_atlas(device, queue, hud);

        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&HudUniforms {
                screen: [screen.0, screen.1],
                _pad: [0.0; 2],
            }),
        );

        let mut quad_verts = Vec::with_capacity(quads.len() * 6);
        for q in quads {
            push_quad(&mut quad_verts, q);
        }
        let white_uv = hud.white_uv();
        let mut line_verts = Vec::with_capacity(lines.len() * 2);
        for l in lines {
            push_line(&mut line_verts, l, white_uv);
        }

        upload_hud_vbo(
            device,
            queue,
            "hud_vbo",
            &quad_verts,
            &mut self.quad_vbo,
            &mut self.quad_cap,
        );
        upload_hud_vbo(
            device,
            queue,
            "hud_line_vbo",
            &line_verts,
            &mut self.line_vbo,
            &mut self.line_cap,
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            if !quad_verts.is_empty() {
                if let Some(ref vbo) = self.quad_vbo {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_vertex_buffer(0, vbo.slice(..));
                    pass.draw(0..quad_verts.len() as u32, 0..1);
                }
            }
            if !line_verts.is_empty() {
                if let Some(ref vbo) = self.line_vbo {
                    pass.set_pipeline(&self.line_pipeline);
                    pass.set_vertex_buffer(0, vbo.slice(..));
                    pass.draw(0..line_verts.len() as u32, 0..1);
                }
            }
        }
    }
}

fn push_quad(out: &mut Vec<HudVertex>, q: &HudQuad) {
    let r = q.rect;
    let (u0, v0) = (q.uv_min[0], q.uv_min[1]);
    let (u1, v1) = (q.uv_max[0], q.uv_max[1]);
    let c = q.color;
    let tl = HudVertex {
        pos: [r.min.x, r.min.y],
        uv: [u0, v0],
        color: c,
    };
    let tr = HudVertex {
        pos: [r.max.x, r.min.y],
        uv: [u1, v0],
        color: c,
    };
    let br = HudVertex {
        pos: [r.max.x, r.max.y],
        uv: [u1, v1],
        color: c,
    };
    let bl = HudVertex {
        pos: [r.min.x, r.max.y],
        uv: [u0, v1],
        color: c,
    };
    out.extend_from_slice(&[tl, tr, br, tl, br, bl]);
}

fn push_line(out: &mut Vec<HudVertex>, l: &HudLine, white_uv: [f32; 2]) {
    out.push(HudVertex {
        pos: l.a.to_array(),
        uv: white_uv,
        color: l.color,
    });
    out.push(HudVertex {
        pos: l.b.to_array(),
        uv: white_uv,
        color: l.color,
    });
}

/// WebGPU `write_texture` requires `bytes_per_row` % 256 == 0.
fn pad_texel_rows(src: &[u8], w: u32, h: u32, bpp: u32) -> (Vec<u8>, u32) {
    let row = (w * bpp).max(1);
    let bpr = row.div_ceil(256) * 256;
    if bpr == row && src.len() >= (row * h) as usize {
        return (src.to_vec(), bpr);
    }
    let mut out = vec![0u8; (bpr * h.max(1)) as usize];
    let copy = row.min(bpr) as usize;
    for y in 0..h as usize {
        let s = y * row as usize;
        let d = y * bpr as usize;
        if s + copy <= src.len() && d + copy <= out.len() {
            out[d..d + copy].copy_from_slice(&src[s..s + copy]);
        }
    }
    (out, bpr)
}

fn upload_hud_vbo(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    verts: &[HudVertex],
    slot: &mut Option<wgpu::Buffer>,
    cap: &mut u64,
) {
    if verts.is_empty() {
        return;
    }
    let bytes: &[u8] = bytemuck::cast_slice(verts);
    let size = (bytes.len() as u64).max(wgpu::COPY_BUFFER_ALIGNMENT);
    if *cap < size {
        let next = size.next_power_of_two().max(4096);
        *slot = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: next,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        *cap = next;
    }
    if let Some(buf) = slot.as_ref() {
        queue.write_buffer(buf, 0, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_row_padded_to_256() {
        // HUD atlas is 128×128 R8; WebGPU needs bytes_per_row % 256 == 0.
        let w = 128u32;
        let h = 8u32;
        let dummy = vec![1u8; (w * h) as usize];
        let (padded, bpr) = pad_texel_rows(&dummy, w, h, 1);
        assert_eq!(bpr, 256);
        assert_eq!(padded.len(), (256 * h) as usize);
        assert_eq!(padded[0], 1);
        assert_eq!(padded[w as usize - 1], 1);
        assert_eq!(padded[w as usize], 0);
    }
}
