use crate::{
    AoMethod, AoSettings, ContactShadowSettings, DofSettings, PostProcessSettings, SsgiSettings,
    SsrSettings,
};
use glam::{Mat4, Vec3};

const SSAO_KERNEL: usize = 32;
const BLOOM_LEVELS: usize = 4;
const NOISE_SIZE: u32 = 4;
const HIZ_MAX_LEVELS: u32 = 7;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniforms {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    kernel: [[f32; 4]; SSAO_KERNEL],
    resolution: [f32; 2],
    radius: f32,
    bias: f32,
    noise_scale: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GtaoUniforms {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    resolution: [f32; 2],
    radius: f32,
    thickness: f32,
    params: [f32; 4],
    noise_scale: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContactUniforms {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    light_dir_world: [f32; 4],
    resolution: [f32; 2],
    length: f32,
    thickness: f32,
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsgiUniforms {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    resolution: [f32; 2],
    radius: f32,
    thickness: f32,
    params: [f32; 4],
    full_resolution: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrUniforms {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    resolution: [f32; 2],
    max_distance: f32,
    thickness: f32,
    params: [f32; 4],
    env: [f32; 4],
    hiz: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsgiTemporalUniforms {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsgiUpsampleUniforms {
    half_texel: [f32; 2],
    depth_sigma: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniforms {
    direction: [f32; 2],
    depth_sigma: f32,
    use_depth: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniforms {
    texel: [f32; 2],
    threshold: f32,
    intensity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    fog_color: [f32; 4],
    fog_height: f32,
    fog_height_falloff: f32,
    fog_enabled: f32,
    ao_intensity: f32,
    bloom_intensity: f32,
    exposure: f32,
    tonemap_mode: f32,
    contrast: f32,
    saturation: f32,
    brightness: f32,
    vignette_intensity: f32,
    vignette_smoothness: f32,
    grain_intensity: f32,
    contact_intensity: f32,
    ssgi_intensity: f32,
    ssr_intensity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FxaaUniforms {
    texel: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofUniforms {
    inv_proj: [[f32; 4]; 4],
    resolution: [f32; 2],
    focus_distance: f32,
    aperture: f32,
    max_coc: f32,
    samples: f32,
    frame: f32,
    focus_range: f32,
    bokeh_blades: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofPrelitUniforms {
    intensities: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DofUpUniforms {
    inv_proj: [[f32; 4]; 4],
    focus_distance: f32,
    aperture: f32,
    max_coc: f32,
    focus_range: f32,
    half_texel: [f32; 2],
    depth_sigma: f32,
    _pad: f32,
}

struct Rt {
    _tex: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// Min-depth pyramid for hierarchical SSR march.
struct HizRt {
    _tex: wgpu::Texture,
    /// All mips — sampled in SSR.
    srv: wgpu::TextureView,
    /// Per-mip RT views for building.
    mips: Vec<wgpu::TextureView>,
    _size: (u32, u32),
    levels: u32,
}

/// Env map bindings for SSR / deferred specular resolve.
pub struct SsrEnvInput<'a> {
    pub sharp: &'a wgpu::TextureView,
    pub blur: &'a wgpu::TextureView,
    pub samp: &'a wgpu::Sampler,
    pub blur_levels: f32,
    pub intensity: f32,
    pub rotation_y_rad: f32,
    pub enabled: bool,
}

pub struct PostFx {
    ssao_pipe: wgpu::RenderPipeline,
    gtao_pipe: wgpu::RenderPipeline,
    contact_pipe: wgpu::RenderPipeline,
    ssgi_pipe: wgpu::RenderPipeline,
    ssr_pipe: wgpu::RenderPipeline,
    ssr_temporal_pipe: wgpu::RenderPipeline,
    hiz_copy_pipe: wgpu::RenderPipeline,
    hiz_down_pipe: wgpu::RenderPipeline,
    ssgi_temporal_pipe: wgpu::RenderPipeline,
    ssgi_upsample_pipe: wgpu::RenderPipeline,
    copy_hdr_pipe: wgpu::RenderPipeline,
    blur_pipe: wgpu::RenderPipeline,
    blur_hdr_pipe: wgpu::RenderPipeline,
    bloom_extract_pipe: wgpu::RenderPipeline,
    bloom_down_pipe: wgpu::RenderPipeline,
    bloom_up_pipe: wgpu::RenderPipeline,
    composite_pipe: wgpu::RenderPipeline,
    fxaa_pipe: wgpu::RenderPipeline,
    dof_pipe: wgpu::RenderPipeline,
    dof_coc_pipe: wgpu::RenderPipeline,
    dof_temporal_pipe: wgpu::RenderPipeline,
    dof_prelit_pipe: wgpu::RenderPipeline,
    dof_up_pipe: wgpu::RenderPipeline,

    ssao_bgl: wgpu::BindGroupLayout,
    gtao_bgl: wgpu::BindGroupLayout,
    contact_bgl: wgpu::BindGroupLayout,
    ssgi_bgl: wgpu::BindGroupLayout,
    ssr_bgl: wgpu::BindGroupLayout,
    ssr_temporal_bgl: wgpu::BindGroupLayout,
    hiz_copy_bgl: wgpu::BindGroupLayout,
    hiz_down_bgl: wgpu::BindGroupLayout,
    ssgi_temporal_bgl: wgpu::BindGroupLayout,
    ssgi_upsample_bgl: wgpu::BindGroupLayout,
    copy_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    bloom_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    fxaa_bgl: wgpu::BindGroupLayout,
    dof_bgl: wgpu::BindGroupLayout,
    dof_temporal_bgl: wgpu::BindGroupLayout,
    dof_prelit_bgl: wgpu::BindGroupLayout,
    dof_up_bgl: wgpu::BindGroupLayout,

    ssao_ubo: wgpu::Buffer,
    gtao_ubo: wgpu::Buffer,
    contact_ubo: wgpu::Buffer,
    ssgi_ubo: wgpu::Buffer,
    ssr_ubo: wgpu::Buffer,
    ssgi_temporal_ubo: wgpu::Buffer,
    ssgi_upsample_ubo: wgpu::Buffer,
    blur_ubo: wgpu::Buffer,
    bloom_ubo: wgpu::Buffer,
    composite_ubo: wgpu::Buffer,
    fxaa_ubo: wgpu::Buffer,
    dof_ubo: wgpu::Buffer,
    dof_prelit_ubo: wgpu::Buffer,
    dof_up_ubo: wgpu::Buffer,

    kernel: [[f32; 4]; SSAO_KERNEL],
    noise_view: wgpu::TextureView,
    _noise: wgpu::Texture,
    noise_samp: wgpu::Sampler,
    linear_samp: wgpu::Sampler,
    nearest_samp: wgpu::Sampler,

    ao: Rt,
    ao_temp: Rt,
    contact: Rt,
    ssgi: Rt,
    ssgi_temp: Rt,
    ssgi_hist: Rt,
    ssgi_full: Rt,
    ssr: Rt,
    ssr_temp: Rt,
    ssr_hist: Rt,
    hiz: HizRt,
    bloom: Vec<Rt>,
    composite_temp: Rt,
    dof: Rt,
    dof_temp: Rt,
    dof_hist: Rt,
    dof_pre: Rt,
    dof_half: Rt,
    dof_half_temp: Rt,
    dof_half_hist: Rt,
    white_view: wgpu::TextureView,
    _white: wgpu::Texture,
    black_view: wgpu::TextureView,
    _black: wgpu::Texture,

    prev_view_proj: Mat4,
    ssgi_has_history: bool,
    ssgi_frame: u32,
    ssr_prev_view_proj: Mat4,
    ssr_has_history: bool,
    ssr_frame: u32,
    dof_prev_view_proj: Mat4,
    dof_has_history: bool,
    dof_frame: u32,
    dof_prev_focus: f32,
    dof_prev_fstop: f32,
    dof_prev_scale: f32,
}

impl PostFx {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let ssao_shader = device.create_shader_module(wgpu::include_wgsl!("ssao.wgsl"));
        let gtao_shader = device.create_shader_module(wgpu::include_wgsl!("gtao.wgsl"));
        let contact_shader = device.create_shader_module(wgpu::include_wgsl!("contact_shadow.wgsl"));
        let ssgi_shader = device.create_shader_module(wgpu::include_wgsl!("ssgi.wgsl"));
        let ssr_shader = device.create_shader_module(wgpu::include_wgsl!("ssr.wgsl"));
        let ssr_temporal_shader =
            device.create_shader_module(wgpu::include_wgsl!("ssr_temporal.wgsl"));
        let hiz_copy_shader = device.create_shader_module(wgpu::include_wgsl!("hiz_copy.wgsl"));
        let hiz_down_shader =
            device.create_shader_module(wgpu::include_wgsl!("hiz_downsample.wgsl"));
        let ssgi_temporal_shader =
            device.create_shader_module(wgpu::include_wgsl!("ssgi_temporal.wgsl"));
        let ssgi_upsample_shader =
            device.create_shader_module(wgpu::include_wgsl!("ssgi_upsample.wgsl"));
        let copy_shader = device.create_shader_module(wgpu::include_wgsl!("copy.wgsl"));
        let blur_shader = device.create_shader_module(wgpu::include_wgsl!("blur.wgsl"));
        let bloom_shader = device.create_shader_module(wgpu::include_wgsl!("bloom.wgsl"));
        let composite_shader = device.create_shader_module(wgpu::include_wgsl!("composite.wgsl"));
        let fxaa_shader = device.create_shader_module(wgpu::include_wgsl!("fxaa.wgsl"));
        let dof_shader = device.create_shader_module(wgpu::include_wgsl!("dof.wgsl"));
        let dof_prelit_shader = device.create_shader_module(wgpu::include_wgsl!("dof_prelit.wgsl"));
        let dof_up_shader = device.create_shader_module(wgpu::include_wgsl!("dof_up.wgsl"));
        let dof_temporal_shader =
            device.create_shader_module(wgpu::include_wgsl!("dof_temporal.wgsl"));

        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                depth_entry(1),
                nearest_samp_entry(2),
                tex_entry(3, true),
                nearest_samp_entry(4),
            ],
        });
        let gtao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gtao"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                depth_entry(1),
                nearest_samp_entry(2),
                tex_entry(3, true),
                filter_samp_entry(4),
                tex_entry(5, true),
                nearest_samp_entry(6),
            ],
        });
        let contact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact_shadow"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                depth_entry(1),
                nearest_samp_entry(2),
                tex_entry(3, true),
                filter_samp_entry(4),
            ],
        });
        let ssgi_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                depth_entry(1),
                nearest_samp_entry(2),
                tex_entry(3, true),
                filter_samp_entry(4),
                tex_entry(5, true),
                filter_samp_entry(6),
                tex_entry(7, true),
                filter_samp_entry(8),
                tex_entry(9, true),
                filter_samp_entry(10),
            ],
        });
        let ssr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                depth_entry(1),
                nearest_samp_entry(2),
                tex_entry(3, true),
                filter_samp_entry(4),
                tex_entry(5, true),
                filter_samp_entry(6),
                tex_entry(7, true),
                filter_samp_entry(8),
                tex_entry(9, true),
                filter_samp_entry(10),
                tex_entry(11, true),
                tex_array_entry(12, true),
                filter_samp_entry(13),
                tex_entry(14, false),
                nearest_samp_entry(15),
            ],
        });
        let ssr_temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_temporal"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                tex_entry(2, true),
                nearest_samp_entry(3),
                depth_entry(4),
                nearest_samp_entry(5),
            ],
        });
        let hiz_copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_copy"),
            entries: &[depth_entry(0), nearest_samp_entry(1)],
        });
        let hiz_down_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_down"),
            entries: &[tex_entry(0, false), nearest_samp_entry(1)],
        });
        let ssgi_temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_temporal"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                tex_entry(2, true),
                nearest_samp_entry(3),
                depth_entry(4),
                nearest_samp_entry(5),
            ],
        });
        let ssgi_upsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_upsample"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                filter_samp_entry(2),
                depth_entry(3),
                nearest_samp_entry(4),
            ],
        });
        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy_hdr"),
            entries: &[tex_entry(0, true), nearest_samp_entry(1)],
        });
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                filter_samp_entry(2),
                depth_entry(3),
                nearest_samp_entry(4),
            ],
        });
        let bloom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                filter_samp_entry(2),
            ],
        });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                tex_entry(2, true),
                tex_entry(3, true),
                depth_entry(4),
                filter_samp_entry(5),
                nearest_samp_entry(6),
                tex_entry(7, true),
                tex_entry(8, true),
                tex_entry(9, true),
            ],
        });
        let fxaa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fxaa"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                filter_samp_entry(2),
            ],
        });
        let dof_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                filter_samp_entry(2),
                depth_entry(3),
                nearest_samp_entry(4),
            ],
        });
        let dof_temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_temporal"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                tex_entry(2, true),
                nearest_samp_entry(3),
                depth_entry(4),
                nearest_samp_entry(5),
            ],
        });
        let dof_prelit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_prelit"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                tex_entry(2, true),
                tex_entry(3, true),
                tex_entry(4, true),
                tex_entry(5, true),
                filter_samp_entry(6),
            ],
        });
        let dof_up_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_up"),
            entries: &[
                ubo_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1, true),
                tex_entry(2, true),
                filter_samp_entry(3),
                depth_entry(4),
                nearest_samp_entry(5),
            ],
        });

        let ssao_pipe = fullscreen_pipe(
            device,
            "ssao",
            &ssao_bgl,
            &ssao_shader,
            "fs",
            wgpu::TextureFormat::R8Unorm,
            None,
        );
        let gtao_pipe = fullscreen_pipe(
            device,
            "gtao",
            &gtao_bgl,
            &gtao_shader,
            "fs",
            wgpu::TextureFormat::R8Unorm,
            None,
        );
        let contact_pipe = fullscreen_pipe(
            device,
            "contact_shadow",
            &contact_bgl,
            &contact_shader,
            "fs",
            wgpu::TextureFormat::R8Unorm,
            None,
        );
        let ssgi_pipe = fullscreen_pipe(
            device,
            "ssgi",
            &ssgi_bgl,
            &ssgi_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let ssr_pipe = fullscreen_pipe(
            device,
            "ssr",
            &ssr_bgl,
            &ssr_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let ssr_temporal_pipe = fullscreen_pipe(
            device,
            "ssr_temporal",
            &ssr_temporal_bgl,
            &ssr_temporal_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let hiz_copy_pipe = fullscreen_pipe(
            device,
            "hiz_copy",
            &hiz_copy_bgl,
            &hiz_copy_shader,
            "fs",
            wgpu::TextureFormat::R32Float,
            None,
        );
        let hiz_down_pipe = fullscreen_pipe(
            device,
            "hiz_down",
            &hiz_down_bgl,
            &hiz_down_shader,
            "fs",
            wgpu::TextureFormat::R32Float,
            None,
        );
        let ssgi_temporal_pipe = fullscreen_pipe(
            device,
            "ssgi_temporal",
            &ssgi_temporal_bgl,
            &ssgi_temporal_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let ssgi_upsample_pipe = fullscreen_pipe(
            device,
            "ssgi_upsample",
            &ssgi_upsample_bgl,
            &ssgi_upsample_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let copy_hdr_pipe = fullscreen_pipe(
            device,
            "copy_hdr",
            &copy_bgl,
            &copy_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let blur_pipe = fullscreen_pipe(
            device,
            "blur",
            &blur_bgl,
            &blur_shader,
            "fs",
            wgpu::TextureFormat::R8Unorm,
            None,
        );
        let blur_hdr_pipe = fullscreen_pipe(
            device,
            "blur_hdr",
            &blur_bgl,
            &blur_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let bloom_extract_pipe = fullscreen_pipe(
            device,
            "bloom_extract",
            &bloom_bgl,
            &bloom_shader,
            "fs_extract",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let bloom_down_pipe = fullscreen_pipe(
            device,
            "bloom_down",
            &bloom_bgl,
            &bloom_shader,
            "fs_downsample",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let bloom_up_pipe = fullscreen_pipe(
            device,
            "bloom_up",
            &bloom_bgl,
            &bloom_shader,
            "fs_upsample",
            wgpu::TextureFormat::Rgba16Float,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
        );
        let composite_pipe = fullscreen_pipe(
            device,
            "composite",
            &composite_bgl,
            &composite_shader,
            "fs",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            None,
        );
        let fxaa_pipe = fullscreen_pipe(
            device,
            "fxaa",
            &fxaa_bgl,
            &fxaa_shader,
            "fs",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            None,
        );
        let dof_pipe = fullscreen_pipe(
            device,
            "dof",
            &dof_bgl,
            &dof_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let dof_coc_pipe = fullscreen_pipe(
            device,
            "dof_coc",
            &dof_bgl,
            &dof_shader,
            "fs_coc",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let dof_temporal_pipe = fullscreen_pipe(
            device,
            "dof_temporal",
            &dof_temporal_bgl,
            &dof_temporal_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let dof_prelit_pipe = fullscreen_pipe(
            device,
            "dof_prelit",
            &dof_prelit_bgl,
            &dof_prelit_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );
        let dof_up_pipe = fullscreen_pipe(
            device,
            "dof_up",
            &dof_up_bgl,
            &dof_up_shader,
            "fs",
            wgpu::TextureFormat::Rgba16Float,
            None,
        );

        let ssao_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_ubo"),
            size: std::mem::size_of::<SsaoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gtao_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gtao_ubo"),
            size: std::mem::size_of::<GtaoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let contact_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_ubo"),
            size: std::mem::size_of::<ContactUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssgi_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi_ubo"),
            size: std::mem::size_of::<SsgiUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssr_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr_ubo"),
            size: std::mem::size_of::<SsrUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssgi_temporal_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi_temporal_ubo"),
            size: std::mem::size_of::<SsgiTemporalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssgi_upsample_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi_upsample_ubo"),
            size: std::mem::size_of::<SsgiUpsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur_ubo"),
            size: std::mem::size_of::<BlurUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bloom_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_ubo"),
            size: std::mem::size_of::<BloomUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite_ubo"),
            size: std::mem::size_of::<CompositeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fxaa_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fxaa_ubo"),
            size: std::mem::size_of::<FxaaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dof_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_ubo"),
            size: std::mem::size_of::<DofUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dof_prelit_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_prelit_ubo"),
            size: std::mem::size_of::<DofPrelitUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dof_up_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dof_up_ubo"),
            size: std::mem::size_of::<DofUpUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kernel = make_kernel();
        let (noise, noise_view) = make_noise(device, queue);
        let (white, white_view) = solid(device, queue, [255, 255, 255, 255]);
        let (black, black_view) = solid(device, queue, [0, 0, 0, 255]);

        let noise_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let linear_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let ao = make_rt(device, 1, 1, wgpu::TextureFormat::R8Unorm, "ao");
        let ao_temp = make_rt(device, 1, 1, wgpu::TextureFormat::R8Unorm, "ao_temp");
        let contact = make_rt(device, 1, 1, wgpu::TextureFormat::R8Unorm, "contact");
        let ssgi = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "ssgi");
        let ssgi_temp = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "ssgi_temp");
        let ssgi_hist = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "ssgi_hist");
        let ssgi_full = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "ssgi_full");
        let ssr = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "ssr");
        let ssr_temp = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "ssr_temp");
        let ssr_hist = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "ssr_hist");
        let hiz = make_hiz(device, 1, 1);
        let bloom = (0..BLOOM_LEVELS)
            .map(|i| make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, &format!("bloom{i}")))
            .collect();
        let composite_temp =
            make_rt(device, 1, 1, wgpu::TextureFormat::Rgba8UnormSrgb, "composite_temp");
        let dof = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "dof");
        let dof_temp = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "dof_temp");
        let dof_hist = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "dof_hist");
        let dof_pre = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "dof_pre");
        let dof_half = make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "dof_half");
        let dof_half_temp =
            make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "dof_half_temp");
        let dof_half_hist =
            make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, "dof_half_hist");

        Self {
            ssao_pipe,
            gtao_pipe,
            contact_pipe,
            ssgi_pipe,
            ssr_pipe,
            ssr_temporal_pipe,
            hiz_copy_pipe,
            hiz_down_pipe,
            ssgi_temporal_pipe,
            ssgi_upsample_pipe,
            copy_hdr_pipe,
            blur_pipe,
            blur_hdr_pipe,
            bloom_extract_pipe,
            bloom_down_pipe,
            bloom_up_pipe,
            composite_pipe,
            fxaa_pipe,
            dof_pipe,
            dof_coc_pipe,
            dof_temporal_pipe,
            dof_prelit_pipe,
            dof_up_pipe,
            ssao_bgl,
            gtao_bgl,
            contact_bgl,
            ssgi_bgl,
            ssr_bgl,
            ssr_temporal_bgl,
            hiz_copy_bgl,
            hiz_down_bgl,
            ssgi_temporal_bgl,
            ssgi_upsample_bgl,
            copy_bgl,
            blur_bgl,
            bloom_bgl,
            composite_bgl,
            fxaa_bgl,
            dof_bgl,
            dof_temporal_bgl,
            dof_prelit_bgl,
            dof_up_bgl,
            ssao_ubo,
            gtao_ubo,
            contact_ubo,
            ssgi_ubo,
            ssr_ubo,
            ssgi_temporal_ubo,
            ssgi_upsample_ubo,
            blur_ubo,
            bloom_ubo,
            composite_ubo,
            fxaa_ubo,
            dof_ubo,
            dof_prelit_ubo,
            dof_up_ubo,
            kernel,
            noise_view,
            _noise: noise,
            noise_samp,
            linear_samp,
            nearest_samp,
            ao,
            ao_temp,
            contact,
            ssgi,
            ssgi_temp,
            ssgi_hist,
            ssgi_full,
            ssr,
            ssr_temp,
            ssr_hist,
            hiz,
            bloom,
            composite_temp,
            dof,
            dof_temp,
            dof_hist,
            dof_pre,
            dof_half,
            dof_half_temp,
            dof_half_hist,
            white_view,
            _white: white,
            black_view,
            _black: black,
            prev_view_proj: Mat4::IDENTITY,
            ssgi_has_history: false,
            ssgi_frame: 0,
            ssr_prev_view_proj: Mat4::IDENTITY,
            ssr_has_history: false,
            ssr_frame: 0,
            dof_prev_view_proj: Mat4::IDENTITY,
            dof_has_history: false,
            dof_frame: 0,
            dof_prev_focus: -1.0,
            dof_prev_fstop: -1.0,
            dof_prev_scale: -1.0,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.ao.size != (w, h) {
            self.ao = make_rt(device, w, h, wgpu::TextureFormat::R8Unorm, "ao");
            self.ao_temp = make_rt(device, w, h, wgpu::TextureFormat::R8Unorm, "ao_temp");
            self.contact = make_rt(device, w, h, wgpu::TextureFormat::R8Unorm, "contact");
            self.ssgi_full = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "ssgi_full");
            self.ssr = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "ssr");
            self.ssr_temp = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "ssr_temp");
            self.ssr_hist = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "ssr_hist");
            self.hiz = make_hiz(device, w, h);
            self.ssr_has_history = false;
            self.dof = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "dof");
            self.dof_temp = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "dof_temp");
            self.dof_hist = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "dof_hist");
            self.dof_pre = make_rt(device, w, h, wgpu::TextureFormat::Rgba16Float, "dof_pre");
            self.dof_has_history = false;
            self.composite_temp =
                make_rt(device, w, h, wgpu::TextureFormat::Rgba8UnormSrgb, "composite_temp");
        }
        let dw = (w / 2).max(1);
        let dh = (h / 2).max(1);
        if self.dof_half.size != (dw, dh) {
            self.dof_half = make_rt(device, dw, dh, wgpu::TextureFormat::Rgba16Float, "dof_half");
            self.dof_half_temp =
                make_rt(device, dw, dh, wgpu::TextureFormat::Rgba16Float, "dof_half_temp");
            self.dof_half_hist =
                make_rt(device, dw, dh, wgpu::TextureFormat::Rgba16Float, "dof_half_hist");
            self.dof_has_history = false;
        }
        // SSGI gather at half-res; upsampled to ssgi_full for composite.
        let sw = (w / 2).max(1);
        let sh = (h / 2).max(1);
        if self.ssgi.size != (sw, sh) {
            self.ssgi = make_rt(device, sw, sh, wgpu::TextureFormat::Rgba16Float, "ssgi");
            self.ssgi_temp = make_rt(device, sw, sh, wgpu::TextureFormat::Rgba16Float, "ssgi_temp");
            self.ssgi_hist = make_rt(device, sw, sh, wgpu::TextureFormat::Rgba16Float, "ssgi_hist");
            self.ssgi_has_history = false;
        }
        let mut bw = (w / 2).max(1);
        let mut bh = (h / 2).max(1);
        for (i, slot) in self.bloom.iter_mut().enumerate() {
            if slot.size != (bw, bh) {
                *slot = make_rt(
                    device,
                    bw,
                    bh,
                    wgpu::TextureFormat::Rgba16Float,
                    &format!("bloom{i}"),
                );
            }
            bw = (bw / 2).max(1);
            bh = (bh / 2).max(1);
        }
    }

    /// Blurred AO result (`R8Unorm`). Valid after [`Self::generate_ao`] / [`Self::apply`].
    pub fn ao_view(&self) -> &wgpu::TextureView {
        &self.ao.view
    }

    /// Contact-shadow result (`R8Unorm`). Valid after [`Self::generate_contact_shadow`] / [`Self::apply`].
    pub fn contact_view(&self) -> &wgpu::TextureView {
        &self.contact.view
    }

    /// SSGI result (`Rgba16Float`, full-res after upsample). Valid after [`Self::generate_ssgi`] / [`Self::apply`].
    pub fn ssgi_view(&self) -> &wgpu::TextureView {
        &self.ssgi_full.view
    }

    /// SSR specular result (`Rgba16Float`). Valid after [`Self::generate_ssr`] / [`Self::apply`].
    pub fn ssr_view(&self) -> &wgpu::TextureView {
        &self.ssr.view
    }

    /// DOF HDR result. Valid after [`Self::generate_dof`].
    pub fn dof_view(&self) -> &wgpu::TextureView {
        &self.dof.view
    }

    /// DOF CoC debug (near/far). Valid after [`Self::generate_dof_coc`].
    pub fn dof_coc_view(&self) -> &wgpu::TextureView {
        &self.dof_temp.view
    }

    fn write_dof_uniforms(
        &self,
        queue: &wgpu::Queue,
        settings: &DofSettings,
        proj: Mat4,
        focus_distance: f32,
        f_stop: f32,
        size: (u32, u32),
        max_coc_scale: f32,
    ) {
        let (w, h) = size;
        let aperture = settings.scale / f_stop.max(0.5);
        let blades = if settings.bokeh_blades == 0 {
            0.0
        } else {
            settings.bokeh_blades.clamp(5, 8) as f32
        };
        queue.write_buffer(
            &self.dof_ubo,
            0,
            bytemuck::bytes_of(&DofUniforms {
                inv_proj: proj.inverse().to_cols_array_2d(),
                resolution: [w as f32, h as f32],
                focus_distance: focus_distance.max(0.01),
                aperture: aperture.max(0.0),
                max_coc: (settings.max_coc_px * max_coc_scale).max(1.0),
                samples: settings.samples.clamp(4, 24) as f32,
                frame: self.dof_frame as f32,
                focus_range: settings.focus_range.max(0.0),
                bokeh_blades: blades,
                _pad: [0.0; 3],
            }),
        );
    }

    fn dof_invalidate_if_needed(
        &mut self,
        focus_distance: f32,
        f_stop: f32,
        scale: f32,
    ) {
        let focus_delta = (focus_distance - self.dof_prev_focus).abs();
        let fstop_delta = (f_stop - self.dof_prev_fstop).abs();
        let scale_delta = (scale - self.dof_prev_scale).abs();
        if self.dof_prev_focus < 0.0
            || focus_delta > 0.35
            || fstop_delta > 0.4
            || scale_delta > 1.5
        {
            self.dof_has_history = false;
        }
        self.dof_prev_focus = focus_distance;
        self.dof_prev_fstop = f_stop;
        self.dof_prev_scale = scale;
    }

    /// Near/far CoC visualization into `dof_temp` (magenta = near, green = far).
    pub fn generate_dof_coc(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &DofSettings,
        depth: &wgpu::TextureView,
        proj: Mat4,
        focus_distance: f32,
        f_stop: f32,
        size: (u32, u32),
    ) {
        self.write_dof_uniforms(queue, settings, proj, focus_distance, f_stop, size, 1.0);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_coc"),
            layout: &self.dof_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dof_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.black_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });
        let mut pass = color_pass(
            encoder,
            "dof_coc",
            &self.dof_temp.view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        );
        pass.set_pipeline(&self.dof_coc_pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Bake lighting layers into `dof_pre` so DOF blurs AO/SSGI/SSR together.
    pub fn generate_dof_prelit(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &PostProcessSettings,
        scene_color: &wgpu::TextureView,
        light_dir_world: Option<Vec3>,
    ) {
        let ao_i = if settings.ao.enabled {
            settings.ao.intensity
        } else {
            0.0
        };
        let cs_i = if settings.contact_shadow.enabled && light_dir_world.is_some() {
            settings.contact_shadow.intensity
        } else {
            0.0
        };
        let ssgi_i = if settings.ssgi.enabled {
            settings.ssgi.intensity
        } else {
            0.0
        };
        let ssr_i = if settings.ssr.enabled {
            settings.ssr.intensity
        } else {
            0.0
        };
        queue.write_buffer(
            &self.dof_prelit_ubo,
            0,
            bytemuck::bytes_of(&DofPrelitUniforms {
                intensities: [ao_i, cs_i, ssgi_i, ssr_i],
            }),
        );
        let ao_view = if settings.ao.enabled {
            &self.ao.view
        } else {
            &self.white_view
        };
        let contact_view = if settings.contact_shadow.enabled && light_dir_world.is_some() {
            &self.contact.view
        } else {
            &self.white_view
        };
        let ssgi_view = if settings.ssgi.enabled {
            &self.ssgi_full.view
        } else {
            &self.black_view
        };
        let ssr_view = if settings.ssr.enabled {
            &self.ssr.view
        } else {
            &self.black_view
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_prelit"),
            layout: &self.dof_prelit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dof_prelit_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_color),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(contact_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ssgi_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(ssr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
            ],
        });
        let mut pass = color_pass(
            encoder,
            "dof_prelit",
            &self.dof_pre.view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        );
        pass.set_pipeline(&self.dof_prelit_pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Dual-field CoC gather + temporal + optional half-res upsample into `dof`.
    pub fn generate_dof(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &DofSettings,
        scene_color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        proj: Mat4,
        view_proj: Mat4,
        focus_distance: f32,
        f_stop: f32,
        size: (u32, u32),
    ) {
        self.dof_invalidate_if_needed(focus_distance, f_stop, settings.scale);

        let half = settings.half_res;
        let (gw, gh) = if half {
            ((size.0 / 2).max(1), (size.1 / 2).max(1))
        } else {
            size
        };
        let coc_scale = if half { 0.5 } else { 1.0 };
        self.write_dof_uniforms(
            queue,
            settings,
            proj,
            focus_distance,
            f_stop,
            (gw, gh),
            coc_scale,
        );

        let gather_dst = if half {
            if settings.temporal {
                &self.dof_half_temp.view
            } else {
                &self.dof_half.view
            }
        } else if settings.temporal {
            &self.dof_temp.view
        } else {
            &self.dof.view
        };

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof"),
            layout: &self.dof_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dof_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_color),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });
        {
            let mut pass =
                color_pass(encoder, "dof", gather_dst, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            pass.set_pipeline(&self.dof_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        let half_resolved = if settings.temporal {
            let (cur, hist, dst) = if half {
                (
                    &self.dof_half_temp.view,
                    &self.dof_half_hist.view,
                    &self.dof_half.view,
                )
            } else {
                (&self.dof_temp.view, &self.dof_hist.view, &self.dof.view)
            };
            let has_hist = self.dof_has_history;
            queue.write_buffer(
                &self.ssgi_temporal_ubo,
                0,
                bytemuck::bytes_of(&SsgiTemporalUniforms {
                    inv_view_proj: view_proj.inverse().to_cols_array_2d(),
                    prev_view_proj: self.dof_prev_view_proj.to_cols_array_2d(),
                    params: [
                        settings.history.clamp(0.0, 0.98),
                        settings.depth_reject.max(0.001),
                        if has_hist { 1.0 } else { 0.0 },
                        0.0,
                    ],
                }),
            );
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dof_temporal"),
                layout: &self.dof_temporal_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ssgi_temporal_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(cur),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(hist),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                ],
            });
            {
                let mut pass =
                    color_pass(encoder, "dof_temporal", dst, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
                pass.set_pipeline(&self.dof_temporal_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            self.copy_hdr(device, encoder, dst, hist);
            self.dof_prev_view_proj = view_proj;
            self.dof_has_history = true;
            dst
        } else {
            self.dof_has_history = false;
            gather_dst
        };

        if half {
            let aperture = settings.scale / f_stop.max(0.5);
            queue.write_buffer(
                &self.dof_up_ubo,
                0,
                bytemuck::bytes_of(&DofUpUniforms {
                    inv_proj: proj.inverse().to_cols_array_2d(),
                    focus_distance: focus_distance.max(0.01),
                    aperture: aperture.max(0.0),
                    max_coc: settings.max_coc_px.max(1.0),
                    focus_range: settings.focus_range.max(0.0),
                    half_texel: [1.0 / gw as f32, 1.0 / gh as f32],
                    depth_sigma: 80.0,
                    _pad: 0.0,
                }),
            );
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dof_up"),
                layout: &self.dof_up_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.dof_up_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(half_resolved),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(scene_color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                ],
            });
            let mut pass = color_pass(
                encoder,
                "dof_up",
                &self.dof.view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
            pass.set_pipeline(&self.dof_up_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        self.dof_frame = self.dof_frame.wrapping_add(1);
    }

    /// Screen-space contact shadows (+ light bilateral blur) into the contact target.
    pub fn generate_contact_shadow(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &ContactShadowSettings,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        light_dir_world: Vec3,
        proj: Mat4,
        view: Mat4,
        size: (u32, u32),
    ) {
        let (w, h) = size;
        let dir = light_dir_world.normalize_or_zero();
        queue.write_buffer(
            &self.contact_ubo,
            0,
            bytemuck::bytes_of(&ContactUniforms {
                proj: proj.to_cols_array_2d(),
                inv_proj: proj.inverse().to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                light_dir_world: [dir.x, dir.y, dir.z, 0.0],
                resolution: [w as f32, h as f32],
                length: settings.length.max(0.01),
                thickness: settings.thickness.max(0.001),
                params: [
                    settings.samples.clamp(4, 32) as f32,
                    settings.bias.max(0.0),
                    0.0,
                    0.0,
                ],
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact_shadow"),
            layout: &self.contact_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.contact_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normals),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
            ],
        });
        {
            let mut pass = color_pass(
                encoder,
                "contact_shadow",
                &self.contact.view,
                wgpu::LoadOp::Clear(wgpu::Color::WHITE),
            );
            pass.set_pipeline(&self.contact_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        self.blur_pass(
            device,
            queue,
            encoder,
            &self.contact.view,
            &self.ao_temp.view,
            depth,
            [1.0 / w as f32, 0.0],
            true,
            40.0,
            false,
        );
        self.blur_pass(
            device,
            queue,
            encoder,
            &self.ao_temp.view,
            &self.contact.view,
            depth,
            [0.0, 1.0 / h as f32],
            true,
            40.0,
            false,
        );
    }

    /// Spatial SSGI (+ optional temporal + HDR bilateral blur) into the SSGI target.
    pub fn generate_ssgi(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &SsgiSettings,
        scene_color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        albedo: &wgpu::TextureView,
        orm: &wgpu::TextureView,
        proj: Mat4,
        view: Mat4,
        view_proj: Mat4,
        size: (u32, u32),
    ) {
        // Gather at half-res target size.
        let (w, h) = self.ssgi.size;
        let (fw, fh) = size;
        let frame = (self.ssgi_frame % 1024) as f32;
        self.ssgi_frame = self.ssgi_frame.wrapping_add(1);

        queue.write_buffer(
            &self.ssgi_ubo,
            0,
            bytemuck::bytes_of(&SsgiUniforms {
                proj: proj.to_cols_array_2d(),
                inv_proj: proj.inverse().to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                resolution: [w as f32, h as f32],
                radius: settings.radius.max(0.05),
                thickness: settings.thickness.max(0.001),
                params: [
                    settings.samples.clamp(4, 32) as f32,
                    settings.bias.max(0.0),
                    settings.max_steps.clamp(4, 32) as f32,
                    frame,
                ],
                full_resolution: [fw.max(1) as f32, fh.max(1) as f32],
                _pad: [0.0; 2],
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssgi"),
            layout: &self.ssgi_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ssgi_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normals),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(scene_color),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(orm),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });

        // Spatial always lands in ssgi_temp (scratch).
        {
            let mut pass = color_pass(
                encoder,
                "ssgi",
                &self.ssgi_temp.view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
            pass.set_pipeline(&self.ssgi_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if settings.temporal {
            let has_hist = self.ssgi_has_history;
            queue.write_buffer(
                &self.ssgi_temporal_ubo,
                0,
                bytemuck::bytes_of(&SsgiTemporalUniforms {
                    inv_view_proj: view_proj.inverse().to_cols_array_2d(),
                    prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
                    params: [
                        settings.history.clamp(0.0, 0.98),
                        settings.depth_reject.max(0.001),
                        if has_hist { 1.0 } else { 0.0 },
                        0.0,
                    ],
                }),
            );
            let tbg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssgi_temporal"),
                layout: &self.ssgi_temporal_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ssgi_temporal_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.ssgi_temp.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.ssgi_hist.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                ],
            });
            {
                let mut pass = color_pass(
                    encoder,
                    "ssgi_temporal",
                    &self.ssgi.view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                );
                pass.set_pipeline(&self.ssgi_temporal_pipe);
                pass.set_bind_group(0, &tbg, &[]);
                pass.draw(0..3, 0..1);
            }
        } else {
            // No temporal: copy spatial into ssgi.
            self.copy_hdr(device, encoder, &self.ssgi_temp.view, &self.ssgi.view);
            self.ssgi_has_history = false;
        }

        // Mild bilateral denoise (preserves alpha / depth).
        self.blur_pass(
            device,
            queue,
            encoder,
            &self.ssgi.view,
            &self.ssgi_temp.view,
            depth,
            [2.0 / w as f32, 0.0],
            true,
            60.0,
            true,
        );
        self.blur_pass(
            device,
            queue,
            encoder,
            &self.ssgi_temp.view,
            &self.ssgi.view,
            depth,
            [0.0, 2.0 / h as f32],
            true,
            60.0,
            true,
        );

        if settings.temporal {
            self.copy_hdr(device, encoder, &self.ssgi.view, &self.ssgi_hist.view);
            self.prev_view_proj = view_proj;
            self.ssgi_has_history = true;
        }

        // Depth-aware upsample half → full (avoids bilinear stripe/block artifacts).
        queue.write_buffer(
            &self.ssgi_upsample_ubo,
            0,
            bytemuck::bytes_of(&SsgiUpsampleUniforms {
                half_texel: [1.0 / w as f32, 1.0 / h as f32],
                depth_sigma: 80.0,
                _pad: 0.0,
            }),
        );
        let ubg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssgi_upsample"),
            layout: &self.ssgi_upsample_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ssgi_upsample_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ssgi.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });
        {
            let mut pass = color_pass(
                encoder,
                "ssgi_upsample",
                &self.ssgi_full.view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
            pass.set_pipeline(&self.ssgi_upsample_pipe);
            pass.set_bind_group(0, &ubg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// SSR + env specular resolve into the SSR target (full-res HDR).
    /// Builds Hi-Z, hierarchical march, optional temporal + bilateral denoise.
    pub fn generate_ssr(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &SsrSettings,
        scene_color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        albedo: &wgpu::TextureView,
        orm: &wgpu::TextureView,
        env: &SsrEnvInput<'_>,
        proj: Mat4,
        view: Mat4,
        view_proj: Mat4,
        camera_pos: [f32; 3],
        size: (u32, u32),
    ) {
        let (w, h) = size;
        let frame = (self.ssr_frame % 1024) as f32;
        self.ssr_frame = self.ssr_frame.wrapping_add(1);

        self.build_hiz(device, encoder, depth);

        let max_mip = (self.hiz.levels.saturating_sub(1)) as f32;
        queue.write_buffer(
            &self.ssr_ubo,
            0,
            bytemuck::bytes_of(&SsrUniforms {
                proj: proj.to_cols_array_2d(),
                inv_proj: proj.inverse().to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                inv_view: view.inverse().to_cols_array_2d(),
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
                resolution: [w as f32, h as f32],
                max_distance: settings.max_distance,
                thickness: settings.thickness,
                params: [
                    settings.max_steps.max(8) as f32,
                    settings.bias,
                    settings.roughness_cutoff,
                    frame,
                ],
                env: [
                    env.intensity,
                    env.blur_levels,
                    if env.enabled { 1.0 } else { 0.0 },
                    env.rotation_y_rad,
                ],
                hiz: [max_mip, 0.0, 0.0, 0.0],
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr"),
            layout: &self.ssr_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ssr_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normals),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(scene_color),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(orm),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(env.sharp),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(env.blur),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(env.samp),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(&self.hiz.srv),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });
        {
            let mut pass = color_pass(
                encoder,
                "ssr",
                &self.ssr_temp.view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
            pass.set_pipeline(&self.ssr_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if settings.temporal {
            let has_hist = self.ssr_has_history;
            queue.write_buffer(
                &self.ssgi_temporal_ubo,
                0,
                bytemuck::bytes_of(&SsgiTemporalUniforms {
                    inv_view_proj: view_proj.inverse().to_cols_array_2d(),
                    prev_view_proj: self.ssr_prev_view_proj.to_cols_array_2d(),
                    params: [
                        settings.history.clamp(0.0, 0.98),
                        settings.depth_reject.max(0.001),
                        if has_hist { 1.0 } else { 0.0 },
                        0.0,
                    ],
                }),
            );
            let tbg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssr_temporal"),
                layout: &self.ssr_temporal_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ssgi_temporal_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.ssr_temp.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.ssr_hist.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                ],
            });
            {
                let mut pass = color_pass(
                    encoder,
                    "ssr_temporal",
                    &self.ssr.view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                );
                pass.set_pipeline(&self.ssr_temporal_pipe);
                pass.set_bind_group(0, &tbg, &[]);
                pass.draw(0..3, 0..1);
            }
        } else {
            self.copy_hdr(device, encoder, &self.ssr_temp.view, &self.ssr.view);
            self.ssr_has_history = false;
        }

        self.blur_pass(
            device,
            queue,
            encoder,
            &self.ssr.view,
            &self.ssr_temp.view,
            depth,
            [1.25 / w as f32, 0.0],
            true,
            70.0,
            true,
        );
        self.blur_pass(
            device,
            queue,
            encoder,
            &self.ssr_temp.view,
            &self.ssr.view,
            depth,
            [0.0, 1.25 / h as f32],
            true,
            70.0,
            true,
        );

        if settings.temporal {
            self.copy_hdr(device, encoder, &self.ssr.view, &self.ssr_hist.view);
            self.ssr_prev_view_proj = view_proj;
            self.ssr_has_history = true;
        }
    }

    fn build_hiz(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
    ) {
        // Mip 0: copy hardware depth → R32Float.
        let copy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_copy"),
            layout: &self.hiz_copy_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });
        {
            let mut pass = color_pass(
                encoder,
                "hiz_copy",
                &self.hiz.mips[0],
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
            pass.set_pipeline(&self.hiz_copy_pipe);
            pass.set_bind_group(0, &copy_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        for level in 1..self.hiz.levels as usize {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hiz_down"),
                layout: &self.hiz_down_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.hiz.mips[level - 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                    },
                ],
            });
            let mut pass = color_pass(
                encoder,
                "hiz_down",
                &self.hiz.mips[level],
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
            pass.set_pipeline(&self.hiz_down_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    fn copy_hdr(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_hdr"),
            layout: &self.copy_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });
        let mut pass = color_pass(encoder, "copy_hdr", dst, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        pass.set_pipeline(&self.copy_hdr_pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Run SSAO or GTAO (+ bilateral blur) into the internal AO target.
    pub fn generate_ao(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &AoSettings,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        proj: Mat4,
        view: Mat4,
        size: (u32, u32),
    ) {
        let (w, h) = size;
        match settings.method {
            AoMethod::Ssao => {
                queue.write_buffer(
                    &self.ssao_ubo,
                    0,
                    bytemuck::bytes_of(&SsaoUniforms {
                        proj: proj.to_cols_array_2d(),
                        inv_proj: proj.inverse().to_cols_array_2d(),
                        kernel: self.kernel,
                        resolution: [w as f32, h as f32],
                        radius: settings.radius,
                        bias: settings.bias,
                        noise_scale: [w as f32 / NOISE_SIZE as f32, h as f32 / NOISE_SIZE as f32],
                        _pad: [0.0; 2],
                    }),
                );
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ssao"),
                    layout: &self.ssao_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.ssao_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.noise_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.noise_samp),
                        },
                    ],
                });
                {
                    let mut pass = color_pass(
                        encoder,
                        "ssao",
                        &self.ao.view,
                        wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    );
                    pass.set_pipeline(&self.ssao_pipe);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            AoMethod::Gtao => {
                queue.write_buffer(
                    &self.gtao_ubo,
                    0,
                    bytemuck::bytes_of(&GtaoUniforms {
                        proj: proj.to_cols_array_2d(),
                        inv_proj: proj.inverse().to_cols_array_2d(),
                        view: view.to_cols_array_2d(),
                        resolution: [w as f32, h as f32],
                        radius: settings.radius,
                        thickness: settings.thickness,
                        params: [
                            settings.directions.max(2) as f32,
                            settings.steps.max(2) as f32,
                            0.0,
                            0.0,
                        ],
                        noise_scale: [w as f32 / NOISE_SIZE as f32, h as f32 / NOISE_SIZE as f32],
                        _pad: [0.0; 2],
                    }),
                );
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gtao"),
                    layout: &self.gtao_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.gtao_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(normals),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.noise_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(&self.noise_samp),
                        },
                    ],
                });
                {
                    let mut pass = color_pass(
                        encoder,
                        "gtao",
                        &self.ao.view,
                        wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    );
                    pass.set_pipeline(&self.gtao_pipe);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }

        // GTAO is lower-frequency / slice-noisy → slightly wider bilateral blur.
        let (blur_px, blur_sigma) = match settings.method {
            AoMethod::Gtao => (1.6, 55.0),
            AoMethod::Ssao => (1.0, 80.0),
        };

        self.blur_pass(
            device,
            queue,
            encoder,
            &self.ao.view,
            &self.ao_temp.view,
            depth,
            [blur_px / w as f32, 0.0],
            true,
            blur_sigma,
            false,
        );
        self.blur_pass(
            device,
            queue,
            encoder,
            &self.ao_temp.view,
            &self.ao.view,
            depth,
            [0.0, blur_px / h as f32],
            true,
            blur_sigma,
            false,
        );
    }

    pub fn apply(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &PostProcessSettings,
        scene_color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        albedo: &wgpu::TextureView,
        orm: &wgpu::TextureView,
        target: &wgpu::TextureView,
        proj: Mat4,
        view: Mat4,
        view_proj: Mat4,
        camera_pos: [f32; 3],
        focus_distance: f32,
        f_stop: f32,
        light_dir_world: Option<Vec3>,
        env: &SsrEnvInput<'_>,
        size: (u32, u32),
    ) {
        let (w, h) = size;
        if settings.ao.enabled {
            self.generate_ao(
                device,
                queue,
                encoder,
                &settings.ao,
                depth,
                normals,
                proj,
                view,
                size,
            );
        }

        if settings.contact_shadow.enabled {
            if let Some(dir) = light_dir_world {
                self.generate_contact_shadow(
                    device,
                    queue,
                    encoder,
                    &settings.contact_shadow,
                    depth,
                    normals,
                    dir,
                    proj,
                    view,
                    size,
                );
            }
        }

        if settings.ssgi.enabled {
            self.generate_ssgi(
                device,
                queue,
                encoder,
                &settings.ssgi,
                scene_color,
                depth,
                normals,
                albedo,
                orm,
                proj,
                view,
                view_proj,
                size,
            );
        } else {
            self.ssgi_has_history = false;
        }

        if settings.ssr.enabled {
            self.generate_ssr(
                device,
                queue,
                encoder,
                &settings.ssr,
                scene_color,
                depth,
                normals,
                albedo,
                orm,
                env,
                proj,
                view,
                view_proj,
                camera_pos,
                size,
            );
        } else {
            self.ssr_has_history = false;
        }

        let lit_color = if settings.dof.enabled {
            self.generate_dof_prelit(
                device,
                queue,
                encoder,
                settings,
                scene_color,
                light_dir_world,
            );
            let prelit = self.dof_pre.view.clone();
            self.generate_dof(
                device,
                queue,
                encoder,
                &settings.dof,
                &prelit,
                depth,
                proj,
                view_proj,
                focus_distance,
                f_stop,
                size,
            );
            &self.dof.view
        } else {
            self.dof_has_history = false;
            scene_color
        };

        let dof_baked = settings.dof.enabled;

        if settings.bloom.enabled {
            let thr = settings.bloom.threshold;
            self.bloom_pass(
                device,
                queue,
                encoder,
                &self.bloom_extract_pipe,
                lit_color,
                &self.bloom[0].view,
                self.bloom[0].size,
                thr,
                true,
            );
            for i in 0..BLOOM_LEVELS - 1 {
                self.bloom_pass(
                    device,
                    queue,
                    encoder,
                    &self.bloom_down_pipe,
                    &self.bloom[i].view,
                    &self.bloom[i + 1].view,
                    self.bloom[i].size,
                    thr,
                    true,
                );
            }
            for i in (0..BLOOM_LEVELS - 1).rev() {
                self.bloom_pass(
                    device,
                    queue,
                    encoder,
                    &self.bloom_up_pipe,
                    &self.bloom[i + 1].view,
                    &self.bloom[i].view,
                    self.bloom[i + 1].size,
                    thr,
                    false,
                );
            }
        }

        let ao_view = if settings.ao.enabled {
            &self.ao.view
        } else {
            &self.white_view
        };
        let contact_view = if settings.contact_shadow.enabled && light_dir_world.is_some() {
            &self.contact.view
        } else {
            &self.white_view
        };
        let ssgi_view = if settings.ssgi.enabled {
            &self.ssgi_full.view
        } else {
            &self.black_view
        };
        let ssr_view = if settings.ssr.enabled {
            &self.ssr.view
        } else {
            &self.black_view
        };
        let bloom_view = if settings.bloom.enabled {
            &self.bloom[0].view
        } else {
            &self.black_view
        };

        let tonemap_mode = if settings.tonemap.enabled {
            if settings.tonemap.aces {
                2.0
            } else {
                1.0
            }
        } else {
            // Scene was rendered linear because post is on — still need a curve.
            1.0
        };
        let (contrast, saturation, brightness) = if settings.color_grade.enabled {
            (
                settings.color_grade.contrast,
                settings.color_grade.saturation,
                settings.color_grade.brightness,
            )
        } else {
            (1.0, 1.0, 0.0)
        };
        let (vig_i, vig_s) = if settings.vignette.enabled {
            (settings.vignette.intensity, settings.vignette.smoothness)
        } else {
            (0.0, 0.5)
        };
        let grain = if settings.grain.enabled {
            settings.grain.intensity
        } else {
            0.0
        };
        let fog = &settings.fog;

        queue.write_buffer(
            &self.composite_ubo,
            0,
            bytemuck::bytes_of(&CompositeUniforms {
                inv_view_proj: view_proj.inverse().to_cols_array_2d(),
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
                fog_color: [fog.color[0], fog.color[1], fog.color[2], fog.density],
                fog_height: fog.height,
                fog_height_falloff: fog.height_falloff,
                fog_enabled: if fog.enabled { 1.0 } else { 0.0 },
                ao_intensity: if settings.ao.enabled && !dof_baked {
                    settings.ao.intensity
                } else {
                    0.0
                },
                bloom_intensity: if settings.bloom.enabled {
                    settings.bloom.intensity
                } else {
                    0.0
                },
                exposure: if settings.tonemap.enabled {
                    settings.tonemap.exposure
                } else {
                    1.0
                },
                tonemap_mode,
                contrast,
                saturation,
                brightness,
                vignette_intensity: vig_i,
                vignette_smoothness: vig_s,
                grain_intensity: grain,
                contact_intensity: if settings.contact_shadow.enabled
                    && light_dir_world.is_some()
                    && !dof_baked
                {
                    settings.contact_shadow.intensity
                } else {
                    0.0
                },
                ssgi_intensity: if settings.ssgi.enabled && !dof_baked {
                    settings.ssgi.intensity
                } else {
                    0.0
                },
                ssr_intensity: if settings.ssr.enabled && !dof_baked {
                    settings.ssr.intensity
                } else {
                    0.0
                },
            }),
        );

        let composite_dst = if settings.fxaa.enabled {
            &self.composite_temp.view
        } else {
            target
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.composite_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lit_color),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(contact_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(ssgi_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(ssr_view),
                },
            ],
        });
        {
            let mut pass =
                color_pass(encoder, "composite", composite_dst, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            pass.set_pipeline(&self.composite_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if settings.fxaa.enabled {
            queue.write_buffer(
                &self.fxaa_ubo,
                0,
                bytemuck::bytes_of(&FxaaUniforms {
                    texel: [1.0 / w as f32, 1.0 / h as f32],
                    _pad: [0.0; 2],
                }),
            );
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fxaa"),
                layout: &self.fxaa_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.fxaa_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.composite_temp.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                    },
                ],
            });
            let mut pass =
                color_pass(encoder, "fxaa", target, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            pass.set_pipeline(&self.fxaa_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    fn blur_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        direction: [f32; 2],
        use_depth: bool,
        depth_sigma: f32,
        hdr: bool,
    ) {
        queue.write_buffer(
            &self.blur_ubo,
            0,
            bytemuck::bytes_of(&BlurUniforms {
                direction,
                depth_sigma,
                use_depth: if use_depth { 1.0 } else { 0.0 },
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur"),
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blur_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_samp),
                },
            ],
        });
        let clear = if hdr {
            wgpu::Color::BLACK
        } else {
            wgpu::Color::WHITE
        };
        let mut pass = color_pass(encoder, "blur", dst, wgpu::LoadOp::Clear(clear));
        pass.set_pipeline(if hdr {
            &self.blur_hdr_pipe
        } else {
            &self.blur_pipe
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    fn bloom_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        src_size: (u32, u32),
        threshold: f32,
        clear: bool,
    ) {
        queue.write_buffer(
            &self.bloom_ubo,
            0,
            bytemuck::bytes_of(&BloomUniforms {
                texel: [1.0 / src_size.0 as f32, 1.0 / src_size.1 as f32],
                threshold,
                intensity: 0.0,
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom"),
            layout: &self.bloom_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bloom_ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
            ],
        });
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = color_pass(encoder, "bloom", dst, load);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn color_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &str,
    view: &'a wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    })
}

fn fullscreen_pipe(
    device: &wgpu::Device,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    fs: &str,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn make_rt(device: &wgpu::Device, w: u32, h: u32, format: wgpu::TextureFormat, label: &str) -> Rt {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    Rt {
        _tex: tex,
        view,
        size: (w.max(1), h.max(1)),
    }
}

fn make_hiz(device: &wgpu::Device, w: u32, h: u32) -> HizRt {
    let w = w.max(1);
    let h = h.max(1);
    let levels = ((w.max(h) as f32).log2().floor() as u32 + 1)
        .clamp(1, HIZ_MAX_LEVELS);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hiz"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let srv = tex.create_view(&Default::default());
    let mips = (0..levels)
        .map(|m| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz_mip"),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: m,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    HizRt {
        _tex: tex,
        srv,
        mips,
        _size: (w, h),
        levels,
    }
}

fn make_kernel() -> [[f32; 4]; SSAO_KERNEL] {
    let mut kernel = [[0.0; 4]; SSAO_KERNEL];
    for i in 0..SSAO_KERNEL {
        let t = (i as f32 + 1.0) / SSAO_KERNEL as f32;
        let z = t;
        let a = i as f32 * 2.399963;
        let r = (1.0 - z * z).max(0.0).sqrt() * t * t;
        kernel[i] = [a.cos() * r, a.sin() * r, z, 0.0];
    }
    kernel
}

fn make_noise(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    let mut pixels = [0u8; (NOISE_SIZE * NOISE_SIZE * 4) as usize];
    for i in 0..(NOISE_SIZE * NOISE_SIZE) as usize {
        let a = i as f32 * 2.399963;
        pixels[i * 4] = ((a.cos() * 0.5 + 0.5) * 255.0) as u8;
        pixels[i * 4 + 1] = ((a.sin() * 0.5 + 0.5) * 255.0) as u8;
        pixels[i * 4 + 2] = 0;
        pixels[i * 4 + 3] = 255;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ssao_noise"),
        size: wgpu::Extent3d {
            width: NOISE_SIZE,
            height: NOISE_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
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
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(NOISE_SIZE * 4),
            rows_per_image: Some(NOISE_SIZE),
        },
        wgpu::Extent3d {
            width: NOISE_SIZE,
            height: NOISE_SIZE,
            depth_or_array_layers: 1,
        },
    );
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn solid(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: [u8; 4],
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
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
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn ubo_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn tex_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn tex_array_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn filter_samp_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn nearest_samp_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
        count: None,
    }
}
