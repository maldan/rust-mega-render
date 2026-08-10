use super::Visualizer;
use crate::{
    DebugView, Handle, Mesh, PostProcessSettings, Scene, ShadowFilter, ShadowSettings, Texture,
};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use wgpu::util::DeviceExt;

mod post;
mod ibl_gpu;
mod frame_targets;
mod debug_blit;
mod hud_pass;
mod sss_lut;
use ibl_gpu::GpuIbl;
use post::{PostFx, SsrEnvInput};
use frame_targets::FrameTargets;
use debug_blit::DebugBlit;
use hud_pass::HudPass;
use sss_lut::GpuSssLut;

const DEFAULT_SHADOW_SIZE: u32 = 2048;
const MAX_LIGHTS: usize = 8;
const MAX_BONES: usize = 128;
const OBJECT_UBO_SIZE: u64 = (std::mem::size_of::<ObjectUniforms>() as u64 + 255) & !255;
const OBJECT_STRIDE: u64 = OBJECT_UBO_SIZE;
const SHADOW_EXTENT: f32 = 14.0;

/// `write_texture` requires bytes_per_row % 256 == 0 → width % 16 == 0 for RGBA32Float.
fn bone_tex_width(joint_count: usize) -> u32 {
    let w = joint_count.max(1).min(MAX_BONES) as u32 * 4;
    w.div_ceil(16) * 16
}

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
struct DebugLineInstance {
    start: [f32; 3],
    width_from: f32,
    end: [f32; 3],
    width_to: f32,
    color_from: [f32; 4],
    color_to: [f32; 4],
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
struct DebugTriVertex {
    pos: [f32; 3],
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
    /// x = env intensity, y = blur layer count, z = active, w = pcss filter samples
    ibl: [f32; 4],
    /// x = filter (0=pcf, 1=pcss), y = light_size, z = 1/map_size, w = blocker samples
    shadow: [f32; 4],
    /// x = constant-ambient scale, y = env yaw (radians), z = shadow bias scale
    gi: [f32; 4],
    lights: [GpuLight; MAX_LIGHTS],
    prev_view_proj: [[f32; 4]; 4],
    /// xy = framebuffer pixels, zw unused
    resolution: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    /// x = intensity, y = yaw, zw = resolution
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
    /// metallic, roughness, skinned, sss_strength
    params: [f32; 4],
    /// rgb scatter tint, w = curvature
    sss: [f32; 4],
    prev_model: [[f32; 4]; 4],
}

/// Per-skin bone palette texture (RGBA32F, row0=current, row1=prev).
struct GpuSkinBones {
    texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    prev: Vec<Mat4>,
    has_history: bool,
}

struct GpuMesh {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    synced: u64,
}

struct GpuTexture {
    view: wgpu::TextureView,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    srgb: bool,
    /// Matches [`Texture::gpu_resident`] — STORAGE + host-owned pixels.
    resident: bool,
    synced: u64,
}

pub struct WgpuVisualizer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// MRT HDR mesh pipeline — 5 color targets (color / normal / orm / albedo / velocity).
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    line_overlay_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    point_overlay_pipeline: wgpu::RenderPipeline,
    tri_pipeline: wgpu::RenderPipeline,
    tri_overlay_pipeline: wgpu::RenderPipeline,
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
    bone_bind_layout: wgpu::BindGroupLayout,
    sky_bind_layout: wgpu::BindGroupLayout,
    sky_pipeline: wgpu::RenderPipeline,
    sky_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    shadow_samp: wgpu::Sampler,
    shadow_depth_samp: wgpu::Sampler,
    white: GpuTexture,
    flat_normal: GpuTexture,
    default_mr: GpuTexture,
    ibl: GpuIbl,
    sss_lut: GpuSssLut,
    /// Background CPU env prepare; uploaded on [`Self::poll_env_map`].
    pending_env: Option<mpsc::Receiver<Result<crate::ibl::EnvMaps, String>>>,
    meshes: HashMap<(u32, u32), GpuMesh>,
    textures: HashMap<(u32, u32), GpuTexture>,
    frames: FrameTargets,
    debug_blit: DebugBlit,
    hud_pass: HudPass,
    debug_view: DebugView,
    post_fx: PostFx,
    post: PostProcessSettings,
    shadow: ShadowSettings,
    _shadow_tex: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    shadow_map_size: u32,
    size: (u32, u32),
    /// Persistent dynamic object UBO (grown as needed; not recreated every frame).
    object_buf: wgpu::Buffer,
    object_slots: u64,
    shadow_object_bg: wgpu::BindGroup,
    default_object_bg: wgpu::BindGroup,
    /// Material bind groups keyed by albedo/normal/MR texture keys.
    object_material_bgs: HashMap<[(u32, u32); 3], wgpu::BindGroup>,
    /// One small bone texture per skin.
    skin_bones: HashMap<(u32, u32), GpuSkinBones>,
    _identity_bone_tex: wgpu::Texture,
    identity_bone_bg: wgpu::BindGroup,
    /// Previous model matrix per mesh node (for velocity).
    prev_models: HashMap<(u32, u32), Mat4>,
    prev_view_proj: Mat4,
    motion_has_history: bool,
    last_frame: Instant,
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
        let shadow_depth_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (shadow_tex, shadow_view) = create_shadow_map(device, DEFAULT_SHADOW_SIZE);
        let ibl = GpuIbl::black(device, queue);
        let sss_lut = GpuSssLut::bake(device, queue);

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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
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
        let bone_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bone_palette"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
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
            &shadow_depth_samp,
            &ibl,
            &sss_lut,
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
            bind_group_layouts: &[
                Some(&frame_bind_layout),
                Some(&object_bind_layout),
                Some(&bone_bind_layout),
            ],
            immediate_size: 0,
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&shadow_frame_layout),
                Some(&shadow_object_layout),
                Some(&bone_bind_layout),
            ],
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

        let line_attrs = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32,
            2 => Float32x3,
            3 => Float32,
            4 => Float32x4,
            5 => Float32x4,
        ];
        let line_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugLineInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &line_attrs,
        })];
        let point_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x4];
        let point_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugPointInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &point_attrs,
        })];
        let tri_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let tri_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugTriVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &tri_attrs,
        })];

        // MRT HDR debug pipelines only.
        let line_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_line",
            &line_buffers,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::CompareFunction::Less,
            &depth_stencil,
        );
        let line_overlay_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_line",
            &line_buffers,
            wgpu::PrimitiveTopology::TriangleList,
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
        let tri_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_tri",
            &tri_buffers,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::CompareFunction::Less,
            &depth_stencil,
        );
        let tri_overlay_pipeline = debug_pipeline(
            device,
            &debug_layout,
            &debug_shader,
            "vs_tri",
            &tri_buffers,
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
        let hud_pass = HudPass::new(device, queue);

        let object_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_uniforms"),
            size: OBJECT_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_object_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_object"),
            layout: &shadow_object_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(OBJECT_UBO_SIZE),
                }),
            }],
        });
        let default_object_bg = object_bind_group(
            device,
            &object_bind_layout,
            &object_buf,
            &white.view,
            &flat_normal.view,
            &default_mr.view,
            &sampler,
        );
        let (identity_bone_tex, identity_bone_bg) =
            identity_bone_palette(device, queue, &bone_bind_layout);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            shadow_pipeline,
            line_pipeline,
            line_overlay_pipeline,
            point_pipeline,
            point_overlay_pipeline,
            tri_pipeline,
            tri_overlay_pipeline,
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
            bone_bind_layout,
            sky_bind_layout,
            sky_pipeline,
            sky_bind_group,
            sampler,
            shadow_samp,
            shadow_depth_samp,
            white,
            flat_normal,
            default_mr,
            ibl,
            sss_lut,
            pending_env: None,
            meshes: HashMap::new(),
            textures: HashMap::new(),
            frames,
            debug_blit,
            hud_pass,
            debug_view: DebugView::Final,
            post_fx,
            post: PostProcessSettings::default(),
            shadow: ShadowSettings::default(),
            _shadow_tex: shadow_tex,
            shadow_view,
            shadow_map_size: DEFAULT_SHADOW_SIZE,
            size: (1, 1),
            object_buf,
            object_slots: 1,
            shadow_object_bg,
            default_object_bg,
            object_material_bgs: HashMap::new(),
            skin_bones: HashMap::new(),
            _identity_bone_tex: identity_bone_tex,
            identity_bone_bg,
            prev_models: HashMap::new(),
            prev_view_proj: Mat4::IDENTITY,
            motion_has_history: false,
            last_frame: Instant::now(),
        }
    }

    fn invalidate_object_bind_groups(&mut self) {
        self.object_material_bgs.clear();
        self.default_object_bg = object_bind_group(
            &self.device,
            &self.object_bind_layout,
            &self.object_buf,
            &self.white.view,
            &self.flat_normal.view,
            &self.default_mr.view,
            &self.sampler,
        );
        self.shadow_object_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_object"),
            layout: &self.shadow_object_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.object_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(OBJECT_UBO_SIZE),
                }),
            }],
        });
    }

    fn ensure_object_slots(&mut self, slots: u64) {
        let slots = slots.max(1);
        if slots <= self.object_slots {
            return;
        }
        let new_slots = slots.next_power_of_two().max(slots);
        self.object_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_uniforms"),
            size: OBJECT_STRIDE * new_slots,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.object_slots = new_slots;
        self.invalidate_object_bind_groups();
    }

    fn ensure_skin_bones(&mut self, skin_key: (u32, u32), joint_count: usize) {
        let width = bone_tex_width(joint_count);
        let needs_new = match self.skin_bones.get(&skin_key) {
            Some(s) => s.width < width,
            None => true,
        };
        if !needs_new {
            return;
        }
        let prev = self
            .skin_bones
            .remove(&skin_key)
            .map(|s| (s.prev, s.has_history))
            .unwrap_or_else(|| (Vec::new(), false));
        let gpu = create_skin_bone_tex(
            &self.device,
            &self.bone_bind_layout,
            width,
            prev.0,
            prev.1,
        );
        self.skin_bones.insert(skin_key, gpu);
    }

    fn upload_skin_bones(&mut self, skin_key: (u32, u32), mats: &[Mat4]) {
        self.ensure_skin_bones(skin_key, mats.len());
        let Some(gpu) = self.skin_bones.get_mut(&skin_key) else {
            return;
        };
        let width = gpu.width as usize;
        let mut pixels = vec![[0.0f32; 4]; width * 2];
        let n = mats.len().min(MAX_BONES);
        for i in 0..n {
            let cur = mats[i].to_cols_array_2d();
            let prev = if gpu.has_history {
                gpu.prev.get(i).copied().unwrap_or(mats[i])
            } else {
                mats[i]
            }
            .to_cols_array_2d();
            for c in 0..4 {
                pixels[i * 4 + c] = cur[c];
                pixels[width + i * 4 + c] = prev[c];
            }
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(gpu.width * 16),
                rows_per_image: Some(2),
            },
            wgpu::Extent3d {
                width: gpu.width,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        gpu.prev.clear();
        gpu.prev.extend_from_slice(mats);
        gpu.has_history = true;
    }

    fn ensure_shadow_map(&mut self, requested: u32) {
        let size = match requested {
            1024 | 2048 | 4096 => requested,
            _ => DEFAULT_SHADOW_SIZE,
        };
        if self.shadow_map_size == size {
            return;
        }
        let (tex, view) = create_shadow_map(&self.device, size);
        self._shadow_tex = tex;
        self.shadow_view = view;
        self.shadow_map_size = size;
        self.rebuild_ibl_bind_groups();
    }
    /// Load an equirectangular HDR/EXR on the calling thread (blocks on decode + blur).
    pub fn set_env_map(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        self.pending_env = None;
        let ibl = ibl_gpu::load_and_upload(&self.device, &self.queue, path)?;
        self.ibl = ibl;
        self.rebuild_ibl_bind_groups();
        Ok(())
    }

    /// Decode + blur EXR/HDR on a background thread; GPU upload happens in [`Self::poll_env_map`].
    /// App keeps running with the previous / black env until ready.
    pub fn set_env_map_async(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(ibl_gpu::load_cpu(&path));
        });
        self.pending_env = Some(rx);
    }

    /// Whether a background env prepare is still in flight.
    pub fn env_map_loading(&self) -> bool {
        self.pending_env.is_some()
    }

    /// Apply finished background env maps (call once per frame, e.g. from render).
    pub fn poll_env_map(&mut self) {
        let Some(rx) = &self.pending_env else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(maps)) => {
                self.ibl = GpuIbl::from_maps(&self.device, &self.queue, &maps, true);
                self.rebuild_ibl_bind_groups();
                self.pending_env = None;
                eprintln!("Env map: uploaded to GPU");
            }
            Ok(Err(e)) => {
                eprintln!("env map: {e}");
                self.pending_env = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.pending_env = None;
            }
        }
    }

    pub fn clear_env_map(&mut self) {
        self.pending_env = None;
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
            &self.shadow_depth_samp,
            &self.ibl,
            &self.sss_lut,
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
        self.motion_has_history = false;
        true
    }

    /// Render into an external color target (e.g. swapchain).
    ///
    /// `color` must be `Rgba8UnormSrgb`.
    pub fn render_to(&mut self, scene: &Scene, aspect: f32, color: &wgpu::TextureView) {
        self.render_inner(scene, aspect, Some(color));
    }

    /// GPU texture for a scene [`Texture`] after [`Visualizer::sync`].
    ///
    /// When [`Texture::gpu_resident`] is set, this resource includes `STORAGE_BINDING`
    /// and `COPY_SRC` so the host can paint with compute and read back for export.
    pub fn texture_gpu(&self, handle: Handle<Texture>) -> Option<&wgpu::Texture> {
        self.textures.get(&handle.key()).map(|g| &g.texture)
    }

    /// Sampling view used by the mesh pass.
    ///
    /// For [`Texture::gpu_resident`] this is always an `rgba8unorm` view (no sRGB
    /// reinterpret — STORAGE textures cannot expose sRGB views).
    pub fn texture_view(&self, handle: Handle<Texture>) -> Option<&wgpu::TextureView> {
        self.textures.get(&handle.key()).map(|g| &g.view)
    }

    fn render_inner(
        &mut self,
        scene: &Scene,
        aspect: f32,
        external: Option<&wgpu::TextureView>,
    ) {
        self.poll_env_map();

        let shadow_dir = scene.shadow_directional();
        self.ensure_shadow_map(self.shadow.map_size);

        let light_vp = shadow_dir
            .map(|d| sun_view_proj(d.direction.normalize_or_zero()))
            .unwrap_or(Mat4::IDENTITY);
        let (gpu_lights, light_count) = pack_lights(scene);
        let shadow_filter = match self.shadow.filter {
            ShadowFilter::Pcf => 0.0,
            ShadowFilter::Pcss => 1.0,
        };
        let pcss_light_size = self.shadow.pcss_light_size.clamp(0.0, 1.0);
        let blocker_samples = self.shadow.pcss_blocker_samples.clamp(4, 16) as f32;
        let filter_samples = self.shadow.pcss_filter_samples.clamp(8, 48) as f32;
        let shadow_bias = self.shadow.bias.max(0.0);

        let eye = scene.camera.eye;
        let view_proj = scene.camera.view_proj(aspect);
        let now = Instant::now();
        let frame_dt = (now - self.last_frame).as_secs_f32().clamp(1.0 / 240.0, 0.05);
        self.last_frame = now;

        let ambient_gi = if self.post.ssgi.enabled {
            (1.0 - self.post.ssgi.ambient_dim.clamp(0.0, 1.0)).max(0.0)
        } else {
            1.0
        };
        let env_on = self.ibl.loaded && self.post.env.enabled;
        // When SSR is on it owns specular env (confidence blend with screen hits).
        let mesh_env_on = env_on && !self.post.ssr.enabled;
        let env_rot = self.post.env.rotation_y.to_radians();
        let env_intensity = self.post.env.intensity;

        // Always output linear HDR into the G-buffer.
        let prev_vp = if self.motion_has_history {
            self.prev_view_proj
        } else {
            view_proj
        };
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
                    env_intensity,
                    self.ibl.blur_levels,
                    if mesh_env_on { 1.0 } else { 0.0 },
                    filter_samples,
                ],
                shadow: [
                    shadow_filter,
                    pcss_light_size,
                    1.0 / self.shadow_map_size as f32,
                    blocker_samples,
                ],
                gi: [ambient_gi, env_rot, shadow_bias, 0.0],
                lights: gpu_lights,
                prev_view_proj: prev_vp.to_cols_array_2d(),
                resolution: [self.size.0 as f32, self.size.1 as f32, 0.0, 0.0],
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

        let world = scene.world_matrices();
        let mut draws: Vec<(
            (u32, u32),
            (u32, u32),
            Option<(u32, u32)>,
            Option<(u32, u32)>,
            Option<(u32, u32)>,
            Mat4,
            [f32; 4],
            [f32; 4],
            [f32; 4],
            Option<(u32, u32)>,
        )> = Vec::new();
        // One palette upload per skin (shared by all mesh prims).
        let mut skins_to_upload: HashMap<(u32, u32), (crate::Handle<crate::Skin>, crate::Handle<crate::Node>)> =
            HashMap::new();
        for (h, node) in scene.nodes.iter() {
            if !node.visible {
                continue;
            }
            let Some(mesh_h) = node.mesh else { continue };
            let mesh_key = mesh_h.key();
            if !self.meshes.contains_key(&mesh_key) {
                continue;
            }
            let (albedo, mut params, sss, albedo_key, normal_key, mr_key) =
                match node.material.and_then(|m| scene.materials.get(m)) {
                    Some(mat) => (
                        mat.albedo,
                        [mat.metallic, mat.roughness, 0.0, mat.sss_strength],
                        [
                            mat.sss_color[0],
                            mat.sss_color[1],
                            mat.sss_color[2],
                            mat.sss_curvature,
                        ],
                        mat.albedo_map.map(|t| t.key()),
                        mat.normal_map.map(|t| t.key()),
                        mat.metallic_roughness_map.map(|t| t.key()),
                    ),
                    None => (
                        [1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.5, 0.0, 0.0],
                        [1.0, 0.35, 0.2, 0.3],
                        None,
                        None,
                        None,
                    ),
                };
            let mut skin_key = None;
            if let Some(skin_h) = node.skin {
                if scene.meshes.get(mesh_h).is_some_and(|m| m.joints.is_some()) {
                    params[2] = 1.0;
                    let key = skin_h.key();
                    skins_to_upload.entry(key).or_insert((skin_h, h));
                    skin_key = Some(key);
                }
            }
            let model = world.get(&h.key()).copied().unwrap_or(Mat4::IDENTITY);
            draws.push((
                h.key(),
                mesh_key,
                albedo_key,
                normal_key,
                mr_key,
                model,
                albedo,
                params,
                sss,
                skin_key,
            ));
        }

        for (skin_key, (skin_h, mesh_node)) in &skins_to_upload {
            let mats = scene.joint_matrices_with_cache(*skin_h, *mesh_node, &world);
            if mats.is_empty() {
                continue;
            }
            self.upload_skin_bones(*skin_key, &mats);
        }

        self.ensure_object_slots(draws.len() as u64);
        let mut next_models: HashMap<(u32, u32), Mat4> = HashMap::with_capacity(draws.len());
        for (i, (node_key, _, _, _, _, model, albedo, params, sss, _)) in draws.iter().enumerate() {
            let prev_model = match self.prev_models.get(node_key) {
                Some(m) if self.motion_has_history => *m,
                _ => *model,
            };
            let base = i as u64 * OBJECT_STRIDE;
            let uniforms = ObjectUniforms {
                model: model.to_cols_array_2d(),
                albedo: *albedo,
                params: *params,
                sss: *sss,
                prev_model: prev_model.to_cols_array_2d(),
            };
            self.queue
                .write_buffer(&self.object_buf, base, bytemuck::bytes_of(&uniforms));
            next_models.insert(*node_key, *model);
        }
        self.prev_models = next_models;
        self.prev_view_proj = view_proj;
        self.motion_has_history = true;

        let debug = build_debug_batches(scene);
        let line_buf = upload_debug_lines(&self.device, &debug.lines);
        let line_overlay_buf = upload_debug_lines(&self.device, &debug.lines_overlay);
        let point_buf = upload_debug_points(&self.device, &debug.points);
        let point_overlay_buf = upload_debug_points(&self.device, &debug.points_overlay);
        let tri_buf = upload_debug_tris(&self.device, &debug.tris);
        let tri_overlay_buf = upload_debug_tris(&self.device, &debug.tris_overlay);

        let default_maps = ((u32::MAX, 0), (u32::MAX, 1), (u32::MAX, 2));
        for (_, _, albedo_key, normal_key, mr_key, _, _, _, _, _) in &draws {
            let cache_key = [
                albedo_key.unwrap_or(default_maps.0),
                normal_key.unwrap_or(default_maps.1),
                mr_key.unwrap_or(default_maps.2),
            ];
            if cache_key == [default_maps.0, default_maps.1, default_maps.2] {
                continue;
            }
            if self.object_material_bgs.contains_key(&cache_key) {
                continue;
            }
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
                &self.object_buf,
                albedo_view,
                normal_view,
                mr_view,
                &self.sampler,
            );
            self.object_material_bgs.insert(cache_key, bg);
        }

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
            for (i, (_, mesh_key, _, _, _, _, _, _, _, skin_key)) in draws.iter().enumerate() {
                let Some(mesh) = self.meshes.get(mesh_key) else {
                    continue;
                };
                pass.set_bind_group(1, &self.shadow_object_bg, &[(i as u32) * OBJECT_STRIDE as u32]);
                if let Some(s) = skin_key.as_ref().and_then(|k| self.skin_bones.get(k)) {
                    pass.set_bind_group(2, &s.bind_group, &[]);
                } else {
                    pass.set_bind_group(2, &self.identity_bone_bg, &[]);
                }
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
                    // color0: HDR scene color
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.frames.color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: scene.clear_color[0] as f64,
                                g: scene.clear_color[1] as f64,
                                b: scene.clear_color[2] as f64,
                                a: scene.clear_color[3] as f64,
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
                    // color2: velocity in pixels (clear to zero)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.frames.velocity_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // color3: ORM (clear to white = full roughness / occlusion)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.frames.orm_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // color4: albedo (clear to black)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.frames.albedo_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

            for (i, (_, mesh_key, albedo_key, normal_key, mr_key, _, _, _, _, skin_key)) in
                draws.iter().enumerate()
            {
                let Some(mesh) = self.meshes.get(mesh_key) else {
                    continue;
                };
                let cache_key = [
                    albedo_key.unwrap_or(default_maps.0),
                    normal_key.unwrap_or(default_maps.1),
                    mr_key.unwrap_or(default_maps.2),
                ];
                let bg = if cache_key == [default_maps.0, default_maps.1, default_maps.2] {
                    &self.default_object_bg
                } else {
                    &self.object_material_bgs[&cache_key]
                };
                pass.set_bind_group(1, bg, &[(i as u32) * OBJECT_STRIDE as u32]);
                if let Some(s) = skin_key.as_ref().and_then(|k| self.skin_bones.get(k)) {
                    pass.set_bind_group(2, &s.bind_group, &[]);
                } else {
                    pass.set_bind_group(2, &self.identity_bone_bg, &[]);
                }
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            draw_debug_tris(
                &mut pass,
                &self.tri_pipeline,
                &self.debug_bind_group,
                &tri_buf,
                debug.tris.len(),
            );
            draw_debug_lines(&mut pass, &self.line_pipeline, &self.debug_bind_group, &line_buf, debug.lines.len());
            draw_debug_points(
                &mut pass,
                &self.point_pipeline,
                &self.debug_bind_group,
                &point_buf,
                debug.points.len(),
            );
            draw_debug_tris(
                &mut pass,
                &self.tri_overlay_pipeline,
                &self.debug_bind_group,
                &tri_overlay_buf,
                debug.tris_overlay.len(),
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

            if env_on {
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
                        prev_view_proj: prev_vp.to_cols_array_2d(),
                        params: [
                            env_intensity,
                            env_rot,
                            self.size.0 as f32,
                            self.size.1 as f32,
                        ],
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
            let env = SsrEnvInput {
                sharp: &self.ibl.sharp_view,
                blur: &self.ibl.blur_view,
                samp: &self.ibl.samp,
                blur_levels: self.ibl.blur_levels,
                intensity: env_intensity,
                rotation_y_rad: env_rot,
                enabled: env_on,
            };
            self.post_fx.apply(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.post,
                &self.frames.color_view,
                &self.frames.depth_view,
                &self.frames.normal_view,
                &self.frames.albedo_view,
                &self.frames.orm_view,
                &self.frames.velocity_view,
                out,
                proj,
                view,
                view_proj,
                [eye.x, eye.y, eye.z],
                scene.camera.focus_distance,
                scene.camera.f_stop,
                light_dir,
                &env,
                self.size,
                frame_dt,
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
            } else if self.debug_view == DebugView::Ssgi {
                self.post_fx.generate_ssgi(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &self.post.ssgi,
                    &self.frames.color_view,
                    &self.frames.depth_view,
                    &self.frames.normal_view,
                    &self.frames.albedo_view,
                    &self.frames.orm_view,
                    &self.frames.velocity_view,
                    proj,
                    view,
                    view_proj,
                    self.size,
                );
                Some(self.post_fx.ssgi_view())
            } else if self.debug_view == DebugView::Ssr {
                let env = SsrEnvInput {
                    sharp: &self.ibl.sharp_view,
                    blur: &self.ibl.blur_view,
                    samp: &self.ibl.samp,
                    blur_levels: self.ibl.blur_levels,
                    intensity: env_intensity,
                    rotation_y_rad: env_rot,
                    enabled: env_on,
                };
                self.post_fx.generate_ssr(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &self.post.ssr,
                    &self.frames.color_view,
                    &self.frames.depth_view,
                    &self.frames.normal_view,
                    &self.frames.albedo_view,
                    &self.frames.orm_view,
                    &env,
                    proj,
                    view,
                    view_proj,
                    [eye.x, eye.y, eye.z],
                    self.size,
                );
                Some(self.post_fx.ssr_view())
            } else if self.debug_view == DebugView::DofCoc {
                self.post_fx.generate_dof_coc(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &self.post.dof,
                    &self.frames.depth_view,
                    proj,
                    scene.camera.focus_distance,
                    scene.camera.f_stop,
                    self.size,
                );
                Some(self.post_fx.dof_coc_view())
            } else if self.debug_view == DebugView::Dof {
                self.post_fx.generate_dof(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &self.post.dof,
                    &self.frames.color_view,
                    &self.frames.depth_view,
                    proj,
                    view_proj,
                    scene.camera.focus_distance,
                    scene.camera.f_stop,
                    self.size,
                );
                Some(self.post_fx.dof_view())
            } else if self.debug_view == DebugView::Albedo {
                Some(&self.frames.albedo_view)
            } else if self.debug_view == DebugView::Velocity {
                Some(&self.frames.velocity_view)
            } else {
                None
            };
            let occlusion_intensity = match self.debug_view {
                DebugView::Ao => self.post.ao.intensity,
                DebugView::ContactShadow => self.post.contact_shadow.intensity,
                DebugView::Ssgi => self.post.ssgi.intensity,
                DebugView::Ssr => self.post.ssr.intensity,
                // Velocity display scale in pixels (matches typical max_blur).
                DebugView::Velocity => 40.0,
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

        // HUD overlay on present (after post / debug blit).
        self.hud_pass.draw(
            &self.device,
            &self.queue,
            &mut encoder,
            &scene.hud,
            out,
            (self.size.0 as f32, self.size.1 as f32),
        );

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
        let mut textures_changed = false;
        for (h, tex) in scene.textures.iter() {
            let key = h.key();
            live.insert(key, ());
            let needs = self.textures.get(&key).map(|g| g.synced) != Some(tex.version);
            if !needs {
                continue;
            }
            let w = tex.width.max(1);
            let h = tex.height.max(1);
            if let Some(gpu) = self.textures.get_mut(&key) {
                if gpu.width == w
                    && gpu.height == h
                    && gpu.srgb == tex.srgb
                    && gpu.resident == tex.gpu_resident
                {
                    // Same GPU resource. CPU maps get pixel uploads; resident maps are
                    // host-owned after the initial create (skip write_texture).
                    if !tex.gpu_resident {
                        write_texture_pixels(&self.queue, &gpu.texture, tex);
                    }
                    gpu.synced = tex.version;
                    continue;
                }
            }
            self.textures
                .insert(key, upload_texture(&self.device, &self.queue, tex));
            textures_changed = true;
        }
        let tex_count_before = self.textures.len();
        self.textures.retain(|k, _| live.contains_key(k));
        if textures_changed || self.textures.len() != tex_count_before {
            self.invalidate_object_bind_groups();
        }
    }

    fn post_process(&mut self) -> &mut PostProcessSettings {
        &mut self.post
    }

    fn shadow_settings(&mut self) -> &mut ShadowSettings {
        &mut self.shadow
    }

    fn effect_settings(&mut self) -> (&mut PostProcessSettings, &mut ShadowSettings) {
        (&mut self.post, &mut self.shadow)
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
        let inst = DebugLineInstance {
            start: line.start.to_array(),
            width_from: line.opts.width_from,
            end: line.end.to_array(),
            width_to: line.opts.width_to,
            color_from: line.opts.color_from,
            color_to: line.opts.color_to,
        };
        if line.opts.depth_test {
            batches.lines.push(inst);
        } else {
            batches.lines_overlay.push(inst);
        }
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
    for t in &scene.debug.tris {
        let push = |dst: &mut Vec<DebugTriVertex>, p: glam::Vec3| {
            dst.push(DebugTriVertex {
                pos: p.to_array(),
                color: t.color,
            });
        };
        let dst = if t.depth_test {
            &mut batches.tris
        } else {
            &mut batches.tris_overlay
        };
        push(dst, t.a);
        push(dst, t.b);
        push(dst, t.c);
    }
    batches
}

#[derive(Default)]
struct DebugBatches {
    lines: Vec<DebugLineInstance>,
    lines_overlay: Vec<DebugLineInstance>,
    points: Vec<DebugPointInstance>,
    points_overlay: Vec<DebugPointInstance>,
    tris: Vec<DebugTriVertex>,
    tris_overlay: Vec<DebugTriVertex>,
}

fn upload_debug_lines(device: &wgpu::Device, lines: &[DebugLineInstance]) -> Option<wgpu::Buffer> {
    if lines.is_empty() {
        return None;
    }
    Some(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug_lines"),
            contents: bytemuck::cast_slice(lines),
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

fn upload_debug_tris(device: &wgpu::Device, tris: &[DebugTriVertex]) -> Option<wgpu::Buffer> {
    if tris.is_empty() {
        return None;
    }
    Some(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug_tris"),
            contents: bytemuck::cast_slice(tris),
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
    pass.draw(0..6, 0..count as u32);
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

fn draw_debug_tris(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    buf: &Option<wgpu::Buffer>,
    vert_count: usize,
) {
    let Some(buf) = buf else { return };
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, buf.slice(..));
    pass.draw(0..vert_count as u32, 0..1);
}

fn make_frame_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    frame_ubo: &wgpu::Buffer,
    shadow_view: &wgpu::TextureView,
    shadow_samp: &wgpu::Sampler,
    shadow_depth_samp: &wgpu::Sampler,
    ibl: &GpuIbl,
    sss_lut: &GpuSssLut,
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
                resource: wgpu::BindingResource::TextureView(&ibl.sharp_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&ibl.blur_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&ibl.samp),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(shadow_depth_samp),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&sss_lut.view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Sampler(&sss_lut.samp),
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
                resource: wgpu::BindingResource::TextureView(&ibl.sharp_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&ibl.samp),
            },
        ],
    })
}

/// Sky pipeline targeting the G-buffer MRT (5 color targets).
fn make_sky_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    depth_stencil: &wgpu::DepthStencilState,
    label: &str,
) -> wgpu::RenderPipeline {
    let formats = FrameTargets::color_formats();
    let targets = gbuffer_targets(&formats, false);
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
            targets: &targets,
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

/// Mesh pipeline targeting the G-buffer MRT (5 color targets).
fn mesh_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    depth_stencil: &wgpu::DepthStencilState,
    label: &str,
) -> wgpu::RenderPipeline {
    let formats = FrameTargets::color_formats();
    let targets = gbuffer_targets(&formats, false);
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
            targets: &targets,
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

/// Debug line/point pipeline targeting the G-buffer MRT (5 color targets).
/// Alpha blend on color0; no writes to normal/ORM/albedo/velocity targets.
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
    let targets = gbuffer_targets(&formats, true);
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
            targets: &targets,
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

fn gbuffer_targets(
    formats: &[wgpu::TextureFormat; 5],
    debug_overlay: bool,
) -> [Option<wgpu::ColorTargetState>; 5] {
    let side_mask = if debug_overlay {
        wgpu::ColorWrites::empty()
    } else {
        wgpu::ColorWrites::ALL
    };
    let color0_blend = if debug_overlay {
        Some(wgpu::BlendState::ALPHA_BLENDING)
    } else {
        None
    };
    [
        Some(wgpu::ColorTargetState {
            format: formats[0],
            blend: color0_blend,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: formats[1],
            blend: None,
            write_mask: side_mask,
        }),
        Some(wgpu::ColorTargetState {
            format: formats[2],
            blend: None,
            write_mask: side_mask,
        }),
        Some(wgpu::ColorTargetState {
            format: formats[3],
            blend: None,
            write_mask: side_mask,
        }),
        Some(wgpu::ColorTargetState {
            format: formats[4],
            blend: None,
            write_mask: side_mask,
        }),
    ]
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

fn create_skin_bone_tex(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    width: u32,
    prev: Vec<Mat4>,
    has_history: bool,
) -> GpuSkinBones {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skin_bones"),
        size: wgpu::Extent3d {
            width,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skin_bones_bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    GpuSkinBones {
        texture,
        _view: view,
        bind_group,
        width,
        prev,
        has_history,
    }
}

fn identity_bone_palette(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let width = bone_tex_width(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("identity_bones"),
        size: wgpu::Extent3d {
            width,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let id = Mat4::IDENTITY.to_cols_array_2d();
    let mut pixels = vec![[0.0f32; 4]; width as usize * 2];
    for c in 0..4 {
        pixels[c] = id[c];
        pixels[width as usize + c] = id[c];
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(2),
        },
        wgpu::Extent3d {
            width,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("identity_bones_bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    (texture, bind_group)
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

fn write_texture_pixels(queue: &wgpu::Queue, texture: &wgpu::Texture, tex: &Texture) {
    let full_w = tex.width.max(1);
    let full_h = tex.height.max(1);

    let (x, y, w, h) = match tex.dirty {
        Some((x, y, w, h)) if w > 0 && h > 0 && w < full_w && h < full_h => {
            let x = x.min(full_w - 1);
            let y = y.min(full_h - 1);
            let w = w.min(full_w - x).max(1);
            let h = h.min(full_h - y).max(1);
            // Skip tiny edge case where packing a sub-copy is awkward; still upload region.
            (x, y, w, h)
        }
        _ => (0, 0, full_w, full_h),
    };

    // Pack rows into a tight buffer for the subrect (avoids bytes_per_row alignment issues).
    if x == 0 && y == 0 && w == full_w && h == full_h {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tex.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * full_w),
                rows_per_image: Some(full_h),
            },
            wgpu::Extent3d {
                width: full_w,
                height: full_h,
                depth_or_array_layers: 1,
            },
        );
        return;
    }

    let mut packed = Vec::with_capacity((w * h * 4) as usize);
    for row in y..y + h {
        let start = ((row * full_w + x) * 4) as usize;
        let end = start + (w * 4) as usize;
        packed.extend_from_slice(&tex.rgba[start..end]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &packed,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &Texture) -> GpuTexture {
    let w = tex.width.max(1);
    let h = tex.height.max(1);

    // STORAGE cannot use *-srgb formats *or* sRGB views on the same texture (WebGPU).
    // Resident maps are always rgba8unorm + default Unorm view.
    let format = if tex.gpu_resident {
        wgpu::TextureFormat::Rgba8Unorm
    } else if tex.srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let usage = if tex.gpu_resident {
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
    } else {
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(if tex.gpu_resident {
            "scene_tex_gpu_resident"
        } else {
            "scene_tex"
        }),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });

    // Seed from CPU once (including first create of a resident map).
    let need_pixels = (w * h * 4) as usize;
    if tex.rgba.len() >= need_pixels {
        write_texture_pixels(queue, &texture, tex);
    }

    GpuTexture {
        view: texture.create_view(&Default::default()),
        texture,
        width: w,
        height: h,
        srgb: tex.srgb,
        resident: tex.gpu_resident,
        synced: tex.version,
    }
}

fn create_shadow_map(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let size = size.max(256);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow_map"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
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
