use crate::{AoMethod, AoSettings, PostProcessSettings};
use glam::Mat4;

const SSAO_KERNEL: usize = 32;
const BLOOM_LEVELS: usize = 4;
const NOISE_SIZE: u32 = 4;

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
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FxaaUniforms {
    texel: [f32; 2],
    _pad: [f32; 2],
}

struct Rt {
    _tex: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

pub struct PostFx {
    ssao_pipe: wgpu::RenderPipeline,
    gtao_pipe: wgpu::RenderPipeline,
    blur_pipe: wgpu::RenderPipeline,
    bloom_extract_pipe: wgpu::RenderPipeline,
    bloom_down_pipe: wgpu::RenderPipeline,
    bloom_up_pipe: wgpu::RenderPipeline,
    composite_pipe: wgpu::RenderPipeline,
    fxaa_pipe: wgpu::RenderPipeline,

    ssao_bgl: wgpu::BindGroupLayout,
    gtao_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    bloom_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    fxaa_bgl: wgpu::BindGroupLayout,

    ssao_ubo: wgpu::Buffer,
    gtao_ubo: wgpu::Buffer,
    blur_ubo: wgpu::Buffer,
    bloom_ubo: wgpu::Buffer,
    composite_ubo: wgpu::Buffer,
    fxaa_ubo: wgpu::Buffer,

    kernel: [[f32; 4]; SSAO_KERNEL],
    noise_view: wgpu::TextureView,
    _noise: wgpu::Texture,
    noise_samp: wgpu::Sampler,
    linear_samp: wgpu::Sampler,
    nearest_samp: wgpu::Sampler,

    ao: Rt,
    ao_temp: Rt,
    bloom: Vec<Rt>,
    composite_temp: Rt,
    white_view: wgpu::TextureView,
    _white: wgpu::Texture,
    black_view: wgpu::TextureView,
    _black: wgpu::Texture,
}

impl PostFx {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let ssao_shader = device.create_shader_module(wgpu::include_wgsl!("ssao.wgsl"));
        let gtao_shader = device.create_shader_module(wgpu::include_wgsl!("gtao.wgsl"));
        let blur_shader = device.create_shader_module(wgpu::include_wgsl!("blur.wgsl"));
        let bloom_shader = device.create_shader_module(wgpu::include_wgsl!("bloom.wgsl"));
        let composite_shader = device.create_shader_module(wgpu::include_wgsl!("composite.wgsl"));
        let fxaa_shader = device.create_shader_module(wgpu::include_wgsl!("fxaa.wgsl"));

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
        let blur_pipe = fullscreen_pipe(
            device,
            "blur",
            &blur_bgl,
            &blur_shader,
            "fs",
            wgpu::TextureFormat::R8Unorm,
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
        let bloom = (0..BLOOM_LEVELS)
            .map(|i| make_rt(device, 1, 1, wgpu::TextureFormat::Rgba16Float, &format!("bloom{i}")))
            .collect();
        let composite_temp =
            make_rt(device, 1, 1, wgpu::TextureFormat::Rgba8UnormSrgb, "composite_temp");

        Self {
            ssao_pipe,
            gtao_pipe,
            blur_pipe,
            bloom_extract_pipe,
            bloom_down_pipe,
            bloom_up_pipe,
            composite_pipe,
            fxaa_pipe,
            ssao_bgl,
            gtao_bgl,
            blur_bgl,
            bloom_bgl,
            composite_bgl,
            fxaa_bgl,
            ssao_ubo,
            gtao_ubo,
            blur_ubo,
            bloom_ubo,
            composite_ubo,
            fxaa_ubo,
            kernel,
            noise_view,
            _noise: noise,
            noise_samp,
            linear_samp,
            nearest_samp,
            ao,
            ao_temp,
            bloom,
            composite_temp,
            white_view,
            _white: white,
            black_view,
            _black: black,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.ao.size != (w, h) {
            self.ao = make_rt(device, w, h, wgpu::TextureFormat::R8Unorm, "ao");
            self.ao_temp = make_rt(device, w, h, wgpu::TextureFormat::R8Unorm, "ao_temp");
            self.composite_temp =
                make_rt(device, w, h, wgpu::TextureFormat::Rgba8UnormSrgb, "composite_temp");
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
        );
    }

    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &PostProcessSettings,
        scene_color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        normals: &wgpu::TextureView,
        target: &wgpu::TextureView,
        proj: Mat4,
        view: Mat4,
        view_proj: Mat4,
        camera_pos: [f32; 3],
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

        if settings.bloom.enabled {
            let thr = settings.bloom.threshold;
            self.bloom_pass(
                device,
                queue,
                encoder,
                &self.bloom_extract_pipe,
                scene_color,
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
                ao_intensity: if settings.ao.enabled {
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
                _pad: [0.0; 3],
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
                    resource: wgpu::BindingResource::TextureView(scene_color),
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
        let mut pass = color_pass(encoder, "blur", dst, wgpu::LoadOp::Clear(wgpu::Color::WHITE));
        pass.set_pipeline(&self.blur_pipe);
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
