//! Screen-space HUD overlay pass (after post / present).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::hud::{Hud, HudQuad};

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
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    atlas_tex: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    atlas_uploaded: bool,
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
                bytes_per_row: Some(1),
                rows_per_image: Some(1),
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<HudVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Float32x4,
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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

        // Keep sampler alive via bind group; store tex/view for reupload.
        // Sampler is owned by device graph through bind_group — fine.
        let _ = atlas_samp;

        Self {
            pipeline,
            bind_group,
            uniform_buf,
            atlas_tex,
            atlas_view,
            atlas_uploaded: false,
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
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &hud.atlas_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w),
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
        if quads.is_empty() {
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

        let mut verts = Vec::with_capacity(quads.len() * 6);
        for q in quads {
            push_quad(&mut verts, q);
        }
        let vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hud_vbo"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, vbo.slice(..));
            pass.draw(0..verts.len() as u32, 0..1);
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
