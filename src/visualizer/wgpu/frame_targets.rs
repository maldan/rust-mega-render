//! Thin G-buffer / frame targets for the wgpu visualizer.
//!
//! Layout (foundation for GTAO / SSR / SSGI / …):
//! - `color`    — HDR lighting (`Rgba16Float`)
//! - `normal`   — world-space normals (`Rgba16Float`, xyz)
//! - `velocity` — screen-space motion in pixels (`Rg16Float`); needs device
//!                `max_color_attachment_bytes_per_sample` ≥ 36 (rgba8 costs 8
//!                in the WebGPU attachment budget, so the old 4-target G-buffer
//!                already filled the default 32)
//! - `orm`      — occlusion / roughness / metallic (`Rgba8Unorm`)
//! - `albedo`   — base color (`Rgba8Unorm`, rgb) for diffuse SSGI
//! - `depth`    — `Depth32Float`
//! - `present`  — internal sRGB present target when no external view is given

pub struct FrameTargets {
    pub _color: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub _normal: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub _velocity: wgpu::Texture,
    pub velocity_view: wgpu::TextureView,
    pub _orm: wgpu::Texture,
    pub orm_view: wgpu::TextureView,
    pub _albedo: wgpu::Texture,
    pub albedo_view: wgpu::TextureView,
    pub _depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub _present: wgpu::Texture,
    pub present_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl FrameTargets {
    /// `present_format` should match the visualizer's negotiated output format
    /// (see [`super::WgpuVisualizer::new`]) so the offscreen `present` target stays
    /// consistent with the pipelines that may render into it.
    pub fn new(device: &wgpu::Device, w: u32, h: u32, present_format: wgpu::TextureFormat) -> Self {
        let w = w.max(1);
        let h = h.max(1);
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let make_tex = |label, format, extra_usage| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | extra_usage,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (tex, view)
        };

        let (color, color_view) = make_tex(
            "gbuffer_color",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::empty(),
        );
        let (normal, normal_view) = make_tex(
            "gbuffer_normal",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::empty(),
        );
        let (velocity, velocity_view) = make_tex(
            "gbuffer_velocity",
            wgpu::TextureFormat::Rg16Float,
            wgpu::TextureUsages::empty(),
        );
        let (orm, orm_view) = make_tex(
            "gbuffer_orm",
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::empty(),
        );
        let (albedo, albedo_view) = make_tex(
            "gbuffer_albedo",
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::empty(),
        );
        let (present, present_view) = make_tex(
            "present",
            present_format,
            wgpu::TextureUsages::empty(),
        );
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());

        Self {
            _color: color,
            color_view,
            _normal: normal,
            normal_view,
            _velocity: velocity,
            velocity_view,
            _orm: orm,
            orm_view,
            _albedo: albedo,
            albedo_view,
            _depth: depth,
            depth_view,
            _present: present,
            present_view,
            size: (w, h),
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        w: u32,
        h: u32,
        present_format: wgpu::TextureFormat,
    ) -> bool {
        let w = w.max(1);
        let h = h.max(1);
        if self.size == (w, h) {
            return false;
        }
        *self = Self::new(device, w, h, present_format);
        true
    }

    /// Color formats for the opaque G-buffer pass (mesh / sky / debug overlay).
    pub fn color_formats() -> [wgpu::TextureFormat; 5] {
        [
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rg16Float,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
    }
}
