use super::{DrawItem, GpuMesh, GpuTexture};
use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

pub const MAX_TESS: u32 = 32;
const VERT_BYTES: u64 = 88;
const PARAMS_SIZE: u64 = 192;
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
    /// World-space frustum planes (left, right, bottom, top, near, far),
    /// each normalized so `dot(xyz, p) + w` is a true signed distance.
    planes: [[f32; 4]; 6],
}

const _: () = assert!(std::mem::size_of::<TessParamsGpu>() == PARAMS_SIZE as usize);

/// Extracts the 6 world-space frustum planes from a camera `view_proj`
/// matrix (Gribb–Hartmann). Assumes wgpu's 0..1 depth range. Each plane is
/// normalized so `dot(plane.xyz, p) + plane.w` gives the true signed
/// distance from world point `p` to the plane.
fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let row = |r: usize| -> Vec4 {
        Vec4::new(
            view_proj.x_axis[r],
            view_proj.y_axis[r],
            view_proj.z_axis[r],
            view_proj.w_axis[r],
        )
    };
    let row0 = row(0);
    let row1 = row(1);
    let row2 = row(2);
    let row3 = row(3);
    let mut planes = [
        row3 + row0, // left
        row3 - row0, // right
        row3 + row1, // bottom
        row3 - row1, // top
        row2,        // near (depth >= 0)
        row3 - row2, // far (depth <= w)
    ];
    for p in &mut planes {
        let len = p.truncate().length().max(1e-8);
        *p /= len;
    }
    planes
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
    /// Storage dest buffers must fit both max_buffer_size and max_storage_buffer_binding_size.
    max_storage_bytes: u64,
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
            max_storage_bytes: tess_storage_limit(&device.limits()),
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
        view_proj: Mat4,
    ) {
        self.live.clear();
        struct Job {
            bg: wgpu::BindGroup,
            groups: u32,
        }
        let mut packed: Vec<Job> = Vec::new();
        let planes = frustum_planes(view_proj).map(|p| p.to_array());
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
            let tess = d
                .tess_factor
                .max(1)
                .min(MAX_TESS)
                .min(max_tess_fitting(tri_count, self.max_storage_bytes));
            let dest_verts = tri_count as u64 * verts_per_tri(tess) as u64;
            let dest_idx = tri_count as u64 * idx_per_tri(tess) as u64;
            let vert_bytes = dest_verts * VERT_BYTES;
            let idx_bytes = dest_idx * 4;
            if vert_bytes > self.max_storage_bytes || idx_bytes > self.max_storage_bytes {
                continue;
            }
            let slot_i = packed.len();
            self.ensure_slot(device, slot_i, vert_bytes, idx_bytes);
            queue.write_buffer(
                &self.slots[slot_i].params_buf,
                0,
                bytemuck::bytes_of(&TessParamsGpu {
                    tri_count,
                    scale: d.displacement_scale,
                    lod_near: LOD_NEAR,
                    lod_far: LOD_FAR,
                    camera_pos: eye.to_array(),
                    tess_factor: tess,
                    model: d.model.to_cols_array_2d(),
                    planes,
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
                contents: &[0u8; PARAMS_SIZE as usize],
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

fn tess_storage_limit(limits: &wgpu::Limits) -> u64 {
    limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size)
}

fn max_tess_fitting(tri_count: u32, max_buf: u64) -> u32 {
    if tri_count == 0 {
        return 1;
    }
    let mut t = MAX_TESS;
    while t >= 1 {
        let vb = tri_count as u64 * verts_per_tri(t) as u64 * VERT_BYTES;
        let ib = tri_count as u64 * idx_per_tri(t) as u64 * 4;
        if vb <= max_buf && ib <= max_buf {
            return t;
        }
        if t == 1 {
            break;
        }
        t -= 1;
    }
    1
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_cube_tess_fits_storage_bind() {
        let tris = 24 * 24 * 2 * 6;
        // Default WebGPU max_storage_buffer_binding_size (128 MiB).
        let max = 128 << 20;
        let t = max_tess_fitting(tris, max);
        let vb = tris as u64 * verts_per_tri(t) as u64 * VERT_BYTES;
        let ib = tris as u64 * idx_per_tri(t) as u64 * 4;
        assert!(vb <= max, "verts t={t} bytes={vb}");
        assert!(ib <= max, "idx t={t} bytes={ib}");
        assert!(t >= 8, "tess collapsed too far: {t}");
        assert!(t < MAX_TESS);
    }
}
