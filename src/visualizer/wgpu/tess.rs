use super::{DrawItem, GpuMesh, GpuTexture};
use glam::Vec3;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

pub const MAX_TESS: u32 = 32;
const VERT_BYTES: u64 = 88;
const PARAMS_SIZE: u64 = 96;
const LOD_NEAR: f32 = 4.0;
const LOD_FAR: f32 = 28.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TessParamsGpu {
    tri_count: u32,
    scale: f32,
    lod_near: f32,
    lod_far: f32,
    camera_pos: [f32; 3],
    tess_factor: u32,
    model: [[f32; 4]; 4],
}

struct TessSlot {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    vert_bytes: u64,
    idx_bytes: u64,
    index_count: u32,
    params_buf: wgpu::Buffer,
}

pub struct TessPass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    slots: Vec<TessSlot>,
    /// node_key → slot index for this frame.
    live: HashMap<(u32, u32), usize>,
}

impl TessPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("tess.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tess"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tess"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tess"),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("tess_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout,
            slots: Vec::new(),
            live: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.live.clear();
    }

    pub fn buffers(&self, node: (u32, u32)) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32)> {
        let i = *self.live.get(&node)?;
        let s = self.slots.get(i)?;
        Some((&s.vertex_buf, &s.index_buf, s.index_count))
    }

    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sampler: &wgpu::Sampler,
        meshes: &HashMap<(u32, u32), GpuMesh>,
        textures: &HashMap<(u32, u32), GpuTexture>,
        draws: &[DrawItem],
        eye: Vec3,
    ) {
        self.live.clear();
        struct Job {
            bg: wgpu::BindGroup,
            groups: u32,
        }
        let mut packed: Vec<Job> = Vec::new();
        for d in draws {
            if !wants_tess(d) {
                continue;
            }
            let Some(mesh) = meshes.get(&d.mesh_key) else {
                continue;
            };
            let Some(hk) = d.height_key else {
                continue;
            };
            let Some(height) = textures.get(&hk) else {
                continue;
            };
            let tri_count = mesh.index_count / 3;
            if tri_count == 0 {
                continue;
            }
            let dest_verts = tri_count as u64 * verts_per_tri(MAX_TESS) as u64;
            let dest_idx = tri_count as u64 * idx_per_tri(MAX_TESS) as u64;
            let slot_i = packed.len();
            self.ensure_slot(device, slot_i, dest_verts * VERT_BYTES, dest_idx * 4);
            queue.write_buffer(
                &self.slots[slot_i].params_buf,
                0,
                bytemuck::bytes_of(&TessParamsGpu {
                    tri_count,
                    scale: d.displacement_scale,
                    lod_near: LOD_NEAR,
                    lod_far: LOD_FAR,
                    camera_pos: eye.to_array(),
                    tess_factor: d.tess_factor.max(1).min(MAX_TESS),
                    model: d.model.to_cols_array_2d(),
                }),
            );
            self.slots[slot_i].index_count = dest_idx as u32;
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tess_bg"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.slots[slot_i].params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&height.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mesh.vertex_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: mesh.index_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.slots[slot_i].vertex_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.slots[slot_i].index_buf.as_entire_binding(),
                    },
                ],
            });
            self.live.insert(d.node_key, slot_i);
            packed.push(Job {
                bg,
                groups: tri_count.div_ceil(64),
            });
        }

        if packed.is_empty() {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tess"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for job in &packed {
            pass.set_bind_group(0, &job.bg, &[]);
            pass.dispatch_workgroups(job.groups.max(1), 1, 1);
        }
    }

    fn ensure_slot(&mut self, device: &wgpu::Device, i: usize, vert_bytes: u64, idx_bytes: u64) {
        let vert_bytes = vert_bytes.max(VERT_BYTES);
        let idx_bytes = idx_bytes.max(12);
        if i < self.slots.len() {
            let s = &self.slots[i];
            if s.vert_bytes >= vert_bytes && s.idx_bytes >= idx_bytes {
                return;
            }
        }
        let slot = TessSlot {
            vertex_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tess_vb"),
                size: vert_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            index_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tess_ib"),
                size: idx_bytes,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            vert_bytes,
            idx_bytes,
            index_count: 0,
            params_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tess_params"),
                contents: &[0u8; 96],
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
        };
        if i < self.slots.len() {
            self.slots[i] = slot;
        } else {
            debug_assert_eq!(i, self.slots.len());
            self.slots.push(slot);
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn wants_tess(d: &DrawItem) -> bool {
    d.height_key.is_some()
        && d.displacement_scale > 0.0
        && d.skin_key.is_none()
        && !d.is_hair
        && !d.is_udim
}

pub fn verts_per_tri(t: u32) -> u32 {
    (t + 1) * (t + 2) / 2
}

pub fn idx_per_tri(t: u32) -> u32 {
    t * t * 3
}

pub fn mesh_bounds(positions: &[[f32; 3]]) -> (Vec3, f32) {
    if positions.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for p in positions {
        let v = Vec3::from(*p);
        min = min.min(v);
        max = max.max(v);
    }
    let c = (min + max) * 0.5;
    let r = (max - min).length() * 0.5;
    (c, r)
}
