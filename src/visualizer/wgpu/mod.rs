use super::Visualizer;
use crate::{DebugView, Mesh, PostProcessSettings, Scene, Texture};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

mod post;
mod ibl_gpu;
mod frame_targets;
mod debug_blit;
use ibl_gpu::GpuIbl;
use post::PostFx;
use frame_targets::FrameTargets;
use debug_blit::DebugBlit;

const SHADOW_SIZE: u32 = 2048;
const MAX_LIGHTS: usize = 8;
const MAX_BONES: usize = 128;
const OBJECT_UBO_SIZE: u64 = (std::mem::size_of::<ObjectUniforms>() as u64 + 255) & !255;
const OBJECT_STRIDE: u64 = OBJECT_UBO_SIZE;
const SHADOW_EXTENT: f32 = 14.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
    joints: [u16; 4],
    weights: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DebugVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DebugPointInstance {
    pos: [f32; 3],
    size: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DebugUniforms {
    view_proj: [[f32; 4]; 4],
    resolution: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    pos_or_dir: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    ambient: [f32; 4],
    camera_pos: [f32; 4],
    /// x = intensity, y = max mip, z = enabled, w unused
    ibl: [f32; 4],
    lights: [GpuLight; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    inv_view_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowFrame {
    light_view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 4],
    albedo: [f32; 4],
    params: [f32; 4], // metallic, roughness, skinned, _
    bones: [[[f32; 4]; 4]; MAX_BONES],
}

struct GpuMesh {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    synced: u64,
}

struct GpuTexture {
    view: wgpu::TextureView,
    _texture: wgpu::Texture,
    synced: u64,
}

pub struct WgpuVisualizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// MRT HDR mesh pipeline — 3 color targets (color / normal / orm).
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    line_overlay_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    point_overlay_pipeline: wgpu::RenderPipeline,
    frame_bind_group: wgpu::BindGroup,
    frame_bind_layout: wgpu::BindGroupLayout,
    debug_bind_group: wgpu::BindGroup,
    shadow_frame_bind_group: wgpu::BindGroup,
    frame_uniform_buf: wgpu::Buffer,
    debug_uniform_buf: wgpu::Buffer,
    shadow_frame_buf: wgpu::Buffer,
    sky_uniform_buf: wgpu::Buffer,
    object_bind_layout: wgpu::BindGroupLayout,
    shadow_object_layout: wgpu::BindGroupLayout,
    sky_bind_layout: wgpu::BindGroupLayout,
    sky_pipeline: wgpu::RenderPipeline,
    sky_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    shadow_samp: wgpu::Sampler,
    white: GpuTexture,
    flat_normal: GpuTexture,
    default_mr: GpuTexture,
    ibl: GpuIbl,
    meshes: HashMap<(u32, u32), GpuMesh>,
    textures: HashMap<(u32, u32), GpuTexture>,
    frames: FrameTargets,
    debug_blit: DebugBlit,
    debug_view: DebugView,
    post_fx: PostFx,
    post: PostProcessSettings,
    _shadow_tex: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    size: (u32, u32),
}

impl WgpuVisualizer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("mesh.wgsl"));
        let shadow_shader = device.create_shader_module(wgpu::include_wgsl!("shadow.wgsl"));
        let debug_shader = device.create_shader_module(wgpu::include_wgsl!("debug.wgsl"));

        let frame_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_uniform"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_frame"),
            size: std::mem::size_of::<ShadowFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_uniform"),
            size: std::mem::size_of::<DebugUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_uniform"),
            size: std::mem::size_of::<SkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (shadow_tex, shadow_view) = create_shadow_map(device);
        let ibl = GpuIbl::black(device, queue);

        let frame_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sky_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
        let debug_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let object_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(OBJECT_UBO_SIZE),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shadow_object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(OBJECT_UBO_SIZE),
                },
                count: None,
            }],
        });

        let frame_bind_group = make_frame_bind_group(
            device,
            &frame_bind_layout,
            &frame_uniform_buf,
            &shadow_view,
            &shadow_samp,
            &ibl,
        );
        let sky_bind_group = make_sky_bind_group(device, &sky_bind_layout, &sky_uniform_buf, &ibl);
        let debug_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &debug_bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: debug_uniform_buf.as_entire_binding(),
            }],
        });
        let shadow_frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &shadow_frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_frame_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&frame_bind_layout), Some(&object_bind_layout)],
            immediate_size: 0,
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&shadow_frame_layout), Some(&shadow_object_layout)],
            immediate_size: 0,
        });
        let debug_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&debug_bind_layout)],
            immediate_size: 0,
        });

        let mesh_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x2,
                3 => Float32x4,
                4 => Uint16x4,
                5 => Float32x4,
            ],
        })];

        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // Single MRT HDR mesh pipeline (3 color targets).
        let pipeline = mesh_pipeline(
            device,
            &pipeline_layout,
            &shader,
            &mesh_buffers,
            &depth_stencil,
            "mesh_mrt",
        );

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_main"),
                buffers: &mesh_buffers,
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Front), // reduce shadow acne
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 0.5,
                    clamp: 0.0,
                },
                ..depth_stencil.clone()
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sky_shader = device.create_shader_module(wgpu::include_wgsl!("skybox.wgsl"));
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky"),
            bind_group_layouts: &[Some(&sky_bind_layout)],
            immediate_size: 0,
        });
        let sky_depth = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        // Single MRT HDR sky pipeline.
        let sky_pipeline = make_sky_pipeline(device, &sky_layout, &sky_shader, &sky_depth, "sky_mrt");

        let line_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let line_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &line_attrs,
        })];
        let point_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x4];
        let point_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugPointInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &point_attrs,
        })];

        // MRT HDR debug pipelines only.
        let line_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_main",
            &line_buffers,
            wgpu::PrimitiveTopology::LineList,
            wgpu::CompareFunction::Less,
            &depth_stencil,
        );
        let line_overlay_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_main",
            &line_buffers,
            wgpu::PrimitiveTopology::LineList,
            wgpu::CompareFunction::Always,
            &depth_stencil,
        );
        let point_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_point",
            &point_buffers,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::CompareFunction::Less,
            &depth_stencil,
        );
        let point_overlay_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_point",
            &point_buffers,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::CompareFunction::Always,
            &depth_stencil,
        );

        let frames = FrameTargets::new(device, 1, 1);
        let white = upload_texture(device, queue, &Texture::solid(255, 255, 255, 255));
        let flat_normal = upload_texture(device, queue, &Texture::solid_linear(128, 128, 255, 255));
        let default_mr = upload_texture(device, queue, &Texture::solid_linear(255, 255, 255, 255));
        let post_fx = PostFx::new(device, queue);
        let debug_blit = DebugBlit::new(device, queue);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            shadow_pipeline,
            line_pipeline,
            line_overlay_pipeline,
            point_pipeline,
            point_overlay_pipeline,
            frame_bind_group,
            frame_bind_layout,
            debug_bind_group,
            shadow_frame_bind_group,
            frame_uniform_buf,
            debug_uniform_buf,
            shadow_frame_buf,
            sky_uniform_buf,
            object_bind_layout,
            shadow_object_layout,
            sky_bind_layout,
            sky_pipeline,
            sky_bind_group,
            sampler,
            shadow_samp,
            white,
            flat_normal,
            default_mr,
            ibl,
            meshes: HashMap::new(),
            textures: HashMap::new(),
            frames,
            debug_blit,
            debug_view: DebugView::Final,
            post_fx,
            post: PostProcessSettings::default(),
            _shadow_tex: shadow_tex,
            shadow_view,
            size: (1, 1),
        }
    }

    /// Load an equirectangular HDR/EXR, bake IBL maps, and enable env lighting + skybox.
    pub fn set_env_map(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let ibl = ibl_gpu::load_and_upload(&self.device, &self.queue, path)?;
        self.ibl = ibl;
        self.rebuild_ibl_bind_groups();
        Ok(())
    }

    pub fn clear_env_map(&mut self) {
        self.ibl = GpuIbl::black(&self.device, &self.queue);
        self.rebuild_ibl_bind_groups();
    }

    fn rebuild_ibl_bind_groups(&mut self) {
        self.frame_bind_group = make_frame_bind_group(
            &self.device,
            &self.frame_bind_layout,
            &self.frame_uniform_buf,
            &self.shadow_view,
            &self.shadow_samp,
            &self.ibl,
        );
        self.sky_bind_group = make_sky_bind_group(
            &self.device,
            &self.sky_bind_layout,
            &self.sky_uniform_buf,
            &self.ibl,
        );
    }

    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.frames.present_view
    }

    pub fn ensure_target(&mut self, w: u32, h: u32) -> bool {
        let w = w.max(1);
        let h = h.max(1);
        if self.size == (w, h) {
            return false;
        }
        self.size = (w, h);
        self.frames.resize(&self.device, w, h);
        self.post_fx.resize(&self.device, w, h);
        true
    }

    /// Render into an external color target (e.g. swapchain).
    ///
    /// `color` must be `Rgba8UnormSrgb`.
    pub fn render_to(&mut self, scene: &Scene, aspect: f32, color: &wgpu::TextureView) {
        self.render_inner(scene, aspect, Some(color));
    }

    fn render_inner(
        &mut self,
        scene: &Scene,
        aspect: f32,
        external: Option<&wgpu::TextureView>,
    ) {
        let shadow_dir = scene
            .shadow_directional()
            .map(|d| d.direction.normalize_or_zero());
        let light_vp = shadow_dir.map(sun_view_proj).unwrap_or(Mat4::IDENTITY);
        let (gpu_lights, light_count) = pack_lights(scene);

        let eye = scene.camera.eye;
        let view_proj = scene.camera.view_proj(aspect);

        // Always output linear HDR into the G-buffer.
        self.queue.write_buffer(
            &self.frame_uniform_buf,
            0,
            bytemuck::bytes_of(&FrameUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                light_view_proj: light_vp.to_cols_array_2d(),
                ambient: [
                    scene.ambient[0],
                    scene.ambient[1],
                    scene.ambient[2],
                    light_count as f32,
                ],
                // w = 1.0 → always output linear HDR
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                ibl: [
                    scene.ibl_intensity,
                    self.ibl.max_mip,
                    if self.ibl.enabled { 1.0 } else { 0.0 },
                    0.0,
                ],
                lights: gpu_lights,
            }),
        );
        self.queue.write_buffer(
            &self.debug_uniform_buf,
            0,
            bytemuck::bytes_of(&DebugUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                resolution: [self.size.0 as f32, self.size.1 as f32],
                _pad: [0.0; 2],
            }),
        );
        self.queue.write_buffer(
            &self.shadow_frame_buf,
            0,
            bytemuck::bytes_of(&ShadowFrame {
                light_view_proj: light_vp.to_cols_array_2d(),
            }),
        );

        let mut draws: Vec<(
            (u32, u32),
            Option<(u32, u32)>,
            Option<(u32, u32)>,
            Option<(u32, u32)>,
            Mat4,
            [f32; 4],
            [f32; 4],
            [[[f32; 4]; 4]; MAX_BONES],
        )> = Vec::new();
        for (h, node) in scene.nodes.iter() {
            if !node.visible {
                continue;
            }
            let Some(mesh_h) = node.mesh else { continue };
            let mesh_key = mesh_h.key();
            if !self.meshes.contains_key(&mesh_key) {
                continue;
            }
            let (albedo, mut params, albedo_key, normal_key, mr_key) =
                match node.material.and_then(|m| scene.materials.get(m)) {
                    Some(mat) => (
                        mat.albedo,
                        [mat.metallic, mat.roughness, 0.0, 0.0],
                        mat.albedo_map.map(|t| t.key()),
                        mat.normal_map.map(|t| t.key()),
                        mat.metallic_roughness_map.map(|t| t.key()),
                    ),
                    None => (
                        [1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.5, 0.0, 0.0],
                        None,
                        None,
                        None,
                    ),
                };
            let mut bones = [[[0.0; 4]; 4]; MAX_BONES];
            bones[0] = Mat4::IDENTITY.to_cols_array_2d();
            if let Some(skin_h) = node.skin {
                if scene.meshes.get(mesh_h).is_some_and(|m| m.joints.is_some()) {
                    let mats = scene.joint_matrices(skin_h, h);
                    params[2] = 1.0;
                    for (i, m) in mats.iter().take(MAX_BONES).enumerate() {
                        bones[i] = m.to_cols_array_2d();
                    }
                }
            }
            draws.push((
                mesh_key,
                albedo_key,
                normal_key,
                mr_key,
                scene.world_matrix(h),
                albedo,
                params,
                bones,
            ));
        }

        let object_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_uniforms"),
            size: OBJECT_STRIDE * draws.len().max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for (i, (_, _, _, _, model, albedo, params, bones)) in draws.iter().enumerate() {
            let data = ObjectUniforms {
                model: model.to_cols_array_2d(),
                albedo: *albedo,
                params: *params,
                bones: *bones,
            };
            self.queue
                .write_buffer(&object_buf, i as u64 * OBJECT_STRIDE, bytemuck::bytes_of(&data));
        }

        let shadow_object_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.shadow_object_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(OBJECT_UBO_SIZE),
                }),
            }],
        });

        let debug = build_debug_batches(scene);
        let line_buf = upload_debug_verts(&self.device, &debug.lines);
        let line_overlay_buf = upload_debug_verts(&self.device, &debug.lines_overlay);
        let point_buf = upload_debug_points(&self.device, &debug.points);
        let point_overlay_buf = upload_debug_points(&self.device, &debug.points_overlay);

        let mut bg_cache: HashMap<[(u32, u32); 3], wgpu::BindGroup> = HashMap::new();
        let default_maps = (
            (u32::MAX, 0),
            (u32::MAX, 1),
            (u32::MAX, 2),
        );
        let default_bg = object_bind_group(
            &self.device,
            &self.object_bind_layout,
            &object_buf,
            &self.white.view,
            &self.flat_normal.view,
            &self.default_mr.view,
            &self.sampler,
        );

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Shadow pass (unchanged).
        if scene.shadow_directional().is_some() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_frame_bind_group, &[]);
            for (i, (mesh_key, _, _, _, _, _, _, _)) in draws.iter().enumerate() {
                let Some(mesh) = self.meshes.get(mesh_key) else {
                    continue;
                };
                pass.set_bind_group(1, &shadow_object_bg, &[(i as u32) * OBJECT_STRIDE as u32]);
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        // Geometry pass: always render into the G-buffer MRT.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_pass"),
                color_attachments: &[
                    // color0: HDR scene color (dark background)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.frames.color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.08,
                                g: 0.09,
                                b: 0.12,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // color1: world-space normals (clear to black = no normal)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.frames.normal_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // color2: ORM (clear to white = full roughness / occlusion)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.frames.orm_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.frames.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);

            for (i, (mesh_key, albedo_key, normal_key, mr_key, _, _, _, _)) in draws.iter().enumerate()
            {
                let Some(mesh) = self.meshes.get(mesh_key) else {
                    continue;
                };
                let cache_key = [
                    albedo_key.unwrap_or(default_maps.0),
                    normal_key.unwrap_or(default_maps.1),
                    mr_key.unwrap_or(default_maps.2),
                ];
                if !bg_cache.contains_key(&cache_key) {
                    let albedo_view = albedo_key
                        .and_then(|k| self.textures.get(&k))
                        .map(|t| &t.view)
                        .unwrap_or(&self.white.view);
                    let normal_view = normal_key
                        .and_then(|k| self.textures.get(&k))
                        .map(|t| &t.view)
                        .unwrap_or(&self.flat_normal.view);
                    let mr_view = mr_key
                        .and_then(|k| self.textures.get(&k))
                        .map(|t| &t.view)
                        .unwrap_or(&self.default_mr.view);
                    let bg = object_bind_group(
                        &self.device,
                        &self.object_bind_layout,
                        &object_buf,
                        albedo_view,
                        normal_view,
                        mr_view,
                        &self.sampler,
                    );
                    bg_cache.insert(cache_key, bg);
                }
                let bg = if cache_key == [default_maps.0, default_maps.1, default_maps.2] {
                    &default_bg
                } else {
                    &bg_cache[&cache_key]
                };
                pass.set_bind_group(1, bg, &[(i as u32) * OBJECT_STRIDE as u32]);
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            draw_debug_lines(&mut pass, &self.line_pipeline, &self.debug_bind_group, &line_buf, debug.lines.len());
            draw_debug_points(
                &mut pass,
                &self.point_pipeline,
                &self.debug_bind_group,
                &point_buf,
                debug.points.len(),
            );
            draw_debug_lines(
                &mut pass,
                &self.line_overlay_pipeline,
                &self.debug_bind_group,
                &line_overlay_buf,
                debug.lines_overlay.len(),
            );
            draw_debug_points(
                &mut pass,
                &self.point_overlay_pipeline,
                &self.debug_bind_group,
                &point_overlay_buf,
                debug.points_overlay.len(),
            );

            if self.ibl.enabled {
                let view = glam::camera::lh::view::look_at_mat4(
                    scene.camera.eye,
                    scene.camera.target,
                    scene.camera.up,
                );
                let view_rot = Mat4::from_cols(view.x_axis, view.y_axis, view.z_axis, glam::Vec4::W);
                let proj = glam::camera::lh::proj::directx::perspective(
                    scene.camera.fov_y,
                    aspect,
                    scene.camera.near,
                    scene.camera.far,
                );
                let inv_view_proj = (proj * view_rot).inverse();
                self.queue.write_buffer(
                    &self.sky_uniform_buf,
                    0,
                    bytemuck::bytes_of(&SkyUniforms {
                        inv_view_proj: inv_view_proj.to_cols_array_2d(),
                        params: [scene.ibl_intensity, 0.0, 0.0, 0.0],
                    }),
                );
                pass.set_pipeline(&self.sky_pipeline);
                pass.set_bind_group(0, &self.sky_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // Present: G-buffer → output.
        let out = external.unwrap_or(&self.frames.present_view);
        let exposure = self.post.tonemap.exposure;
        let near = scene.camera.near;
        let far = scene.camera.far;
        let proj = glam::camera::lh::proj::directx::perspective(
            scene.camera.fov_y,
            aspect,
            near,
            far,
        );

        if self.debug_view == DebugView::Final && self.post.any_enabled() {
            let view = glam::camera::lh::view::look_at_mat4(
                scene.camera.eye,
                scene.camera.target,
                scene.camera.up,
            );
            let light_dir = scene.lights.iter().find_map(|l| match l {
                crate::Light::Directional(d) if d.enabled => Some(d.direction),
                _ => None,
            });
            self.post_fx.apply(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.post,
                &self.frames.color_view,
                &self.frames.depth_view,
                &self.frames.normal_view,
                out,
                proj,
                view,
                view_proj,
                [eye.x, eye.y, eye.z],
                light_dir,
                self.size,
            );
        } else {
            let view = glam::camera::lh::view::look_at_mat4(
                scene.camera.eye,
                scene.camera.target,
                scene.camera.up,
            );
            let ao = if self.debug_view == DebugView::Ao {
                self.post_fx.generate_ao(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &self.post.ao,
                    &self.frames.depth_view,
                    &self.frames.normal_view,
                    proj,
                    view,
                    self.size,
                );
                Some(self.post_fx.ao_view())
            } else if self.debug_view == DebugView::ContactShadow {
                if let Some(dir) = scene.lights.iter().find_map(|l| match l {
                    crate::Light::Directional(d) if d.enabled => Some(d.direction),
                    _ => None,
                }) {
                    self.post_fx.generate_contact_shadow(
                        &self.device,
                        &self.queue,
                        &mut encoder,
                        &self.post.contact_shadow,
                        &self.frames.depth_view,
                        &self.frames.normal_view,
                        dir,
                        proj,
                        view,
                        self.size,
                    );
                    Some(self.post_fx.contact_view())
                } else {
                    None
                }
            } else {
                None
            };
            let occlusion_intensity = match self.debug_view {
                DebugView::Ao => self.post.ao.intensity,
                DebugView::ContactShadow => self.post.contact_shadow.intensity,
                _ => 1.0,
            };
            self.debug_blit.blit(
                &self.device,
                &self.queue,
                &mut encoder,
                self.debug_view,
                &self.frames.color_view,
                &self.frames.normal_view,
                &self.frames.orm_view,
                &self.frames.depth_view,
                ao,
                out,
                exposure,
                near,
                far,
                occlusion_intensity,
            );
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

impl Visualizer for WgpuVisualizer {
    fn sync(&mut self, scene: &Scene) {
        let mut live = HashMap::new();
        for (h, mesh) in scene.meshes.iter() {
            let key = h.key();
            live.insert(key, ());
            if self.meshes.get(&key).map(|g| g.synced) != Some(mesh.version) {
                self.meshes.insert(key, upload_mesh(&self.device, mesh));
            }
        }
        self.meshes.retain(|k, _| live.contains_key(k));

        live.clear();
        for (h, tex) in scene.textures.iter() {
            let key = h.key();
            live.insert(key, ());
            if self.textures.get(&key).map(|g| g.synced) != Some(tex.version) {
                self.textures
                    .insert(key, upload_texture(&self.device, &self.queue, tex));
            }
        }
        self.textures.retain(|k, _| live.contains_key(k));
    }

    fn post_process(&mut self) -> &mut PostProcessSettings {
        &mut self.post
    }

    fn render(&mut self, scene: &Scene, aspect: f32) {
        self.render_inner(scene, aspect, None);
    }

    fn debug_view(&self) -> DebugView {
        self.debug_view
    }

    fn set_debug_view(&mut self, view: DebugView) {
        self.debug_view = view;
    }
}

fn pack_lights(scene: &Scene) -> ([GpuLight; MAX_LIGHTS], u32) {
    use crate::Light;
    let mut out = [GpuLight {
        pos_or_dir: [0.0; 4],
        color: [0.0; 4],
        params: [0.0; 4],
    }; MAX_LIGHTS];
    let mut count = 0u32;
    let mut shadow_assigned = false;
    for light in &scene.lights {
        if count as usize >= MAX_LIGHTS {
            break;
        }
        let gpu = match light {
            Light::Directional(d) => {
                if !d.enabled {
                    continue;
                }
                let dir = d.direction.normalize_or_zero();
                let cast = if d.cast_shadows && !shadow_assigned {
                    shadow_assigned = true;
                    1.0
                } else {
                    0.0
                };
                GpuLight {
                    pos_or_dir: [dir.x, dir.y, dir.z, 0.0],
                    color: [d.color[0], d.color[1], d.color[2], d.intensity],
                    params: [0.0, cast, 0.0, 0.0],
                }
            }
            Light::Point(p) => {
                if !p.enabled {
                    continue;
                }
                GpuLight {
                    pos_or_dir: [p.position.x, p.position.y, p.position.z, 1.0],
                    color: [p.color[0], p.color[1], p.color[2], p.intensity],
                    params: [p.range, 0.0, 0.0, 0.0],
                }
            }
        };
        out[count as usize] = gpu;
        count += 1;
    }
    (out, count)
}

fn sun_view_proj(dir: Vec3) -> Mat4 {
    let dir = dir.normalize_or_zero();
    let center = Vec3::new(0.0, 1.0, 2.0);
    let eye = center - dir * 35.0;
    let up = if dir.cross(Vec3::Y).length_squared() < 1e-4 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = glam::camera::lh::view::look_at_mat4(eye, center, up);
    let e = SHADOW_EXTENT;
    let proj = glam::camera::lh::proj::directx::orthographic(-e, e, -e, e, 1.0, 70.0);
    proj * view
}

fn build_debug_batches(scene: &Scene) -> DebugBatches {
    let mut batches = DebugBatches::default();
    for line in &scene.debug.lines {
        let a = DebugVertex {
            pos: line.start.to_array(),
            color: line.color,
        };
        let b = DebugVertex {
            pos: line.end.to_array(),
            color: line.color,
        };
        let dst = if line.depth_test {
            &mut batches.lines
        } else {
            &mut batches.lines_overlay
        };
        dst.push(a);
        dst.push(b);
    }
    for p in &scene.debug.points {
        let inst = DebugPointInstance {
            pos: p.position.to_array(),
            size: p.size,
            color: p.color,
        };
        if p.depth_test {
            batches.points.push(inst);
        } else {
            batches.points_overlay.push(inst);
        }
    }
    batches
}

#[derive(Default)]
struct DebugBatches {
    lines: Vec<DebugVertex>,
    lines_overlay: Vec<DebugVertex>,
    points: Vec<DebugPointInstance>,
    points_overlay: Vec<DebugPointInstance>,
}

fn upload_debug_verts(device: &wgpu::Device, verts: &[DebugVertex]) -> Option<wgpu::Buffer> {
    if verts.is_empty() {
        return None;
    }
    Some(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug_lines"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        }),
    )
}

fn upload_debug_points(device: &wgpu::Device, points: &[DebugPointInstance]) -> Option<wgpu::Buffer> {
    if points.is_empty() {
        return None;
    }
    Some(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug_points"),
            contents: bytemuck::cast_slice(points),
            usage: wgpu::BufferUsages::VERTEX,
        }),
    )
}

fn draw_debug_lines(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    buf: &Option<wgpu::Buffer>,
    count: usize,
) {
    let Some(buf) = buf else { return };
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, buf.slice(..));
    pass.draw(0..count as u32, 0..1);
}

fn draw_debug_points(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    buf: &Option<wgpu::Buffer>,
    count: usize,
) {
    let Some(buf) = buf else { return };
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, buf.slice(..));
    pass.draw(0..6, 0..count as u32);
}

fn make_frame_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    frame_ubo: &wgpu::Buffer,
    shadow_view: &wgpu::TextureView,
    shadow_samp: &wgpu::Sampler,
    ibl: &GpuIbl,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frame"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(shadow_samp),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&ibl.equirect_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&ibl.brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&ibl.equirect_samp),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(&ibl.clamp_samp),
            },
        ],
    })
}

fn make_sky_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sky_ubo: &wgpu::Buffer,
    ibl: &GpuIbl,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&ibl.equirect_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&ibl.equirect_samp),
            },
        ],
    })
}

/// Sky pipeline targeting the G-buffer MRT (3 color targets).
fn make_sky_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    depth_stencil: &wgpu::DepthStencilState,
    label: &str,
) -> wgpu::RenderPipeline {
    let formats = FrameTargets::color_formats();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: formats[0],
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: formats[1],
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: formats[2],
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(depth_stencil.clone()),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Mesh pipeline targeting the G-buffer MRT (3 color targets).
fn mesh_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    depth_stencil: &wgpu::DepthStencilState,
    label: &str,
) -> wgpu::RenderPipeline {
    let formats = FrameTargets::color_formats();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: formats[0],
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: formats[1],
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: formats[2],
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Cw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(depth_stencil.clone()),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Debug line/point pipeline targeting the G-buffer MRT (3 color targets).
/// Alpha blend on color0; no writes to normal/ORM targets.
fn debug_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs: &str,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    topology: wgpu::PrimitiveTopology,
    depth_compare: wgpu::CompareFunction,
    depth_stencil: &wgpu::DepthStencilState,
) -> wgpu::RenderPipeline {
    let formats = FrameTargets::color_formats();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("debug"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs),
            buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[
                // color0: alpha-blended debug overlay
                Some(wgpu::ColorTargetState {
                    format: formats[0],
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // color1: normals — no write (debug doesn't modify G-buffer)
                Some(wgpu::ColorTargetState {
                    format: formats[1],
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                }),
                // color2: ORM — no write
                Some(wgpu::ColorTargetState {
                    format: formats[2],
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            depth_write_enabled: Some(false),
            depth_compare: Some(depth_compare),
            ..depth_stencil.clone()
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn object_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    object_buf: &wgpu::Buffer,
    albedo: &wgpu::TextureView,
    normal: &wgpu::TextureView,
    mr: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: object_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(OBJECT_UBO_SIZE),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(albedo),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(normal),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(mr),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn upload_mesh(device: &wgpu::Device, mesh: &Mesh) -> GpuMesh {
    let verts: Vec<GpuVertex> = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| GpuVertex {
            pos: *p,
            normal: mesh.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: mesh.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            tangent: mesh.tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
            joints: mesh
                .joints
                .as_ref()
                .and_then(|j| j.get(i).copied())
                .unwrap_or([0; 4]),
            weights: mesh
                .weights
                .as_ref()
                .and_then(|w| w.get(i).copied())
                .unwrap_or([1.0, 0.0, 0.0, 0.0]),
        })
        .collect();
    GpuMesh {
        vertex_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        index_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_ib"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        }),
        index_count: mesh.indices.len() as u32,
        synced: mesh.version,
    }
}

fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &Texture) -> GpuTexture {
    let format = if tex.srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene_tex"),
        size: wgpu::Extent3d {
            width: tex.width.max(1),
            height: tex.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &tex.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * tex.width.max(1)),
            rows_per_image: Some(tex.height.max(1)),
        },
        wgpu::Extent3d {
            width: tex.width.max(1),
            height: tex.height.max(1),
            depth_or_array_layers: 1,
        },
    );
    GpuTexture {
        view: texture.create_view(&Default::default()),
        _texture: texture,
        synced: tex.version,
    }
}

fn create_shadow_map(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow_map"),
        size: wgpu::Extent3d {
            width: SHADOW_SIZE,
            height: SHADOW_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
