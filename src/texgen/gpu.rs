//! GPU graph eval: one compute pipeline, textures stay on GPU between nodes.
//! Flood fill is union-find on a readback (connected components).

use std::collections::HashMap;
use std::time::Instant;

use super::{
    ancestors_of, compute_out_fingerprints, find_link, flood_fill_gray, sample_gradient, topo_order,
    BlendMode, ColorRampParams, GradientMode, GraphNode, NodeKind, NoiseType, ShapeKind,
    SlopeBlurMode, TexLink,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NodeParamsGpu {
    res: u32,
    op: u32,
    kind: u32,
    octaves: u32,
    flags: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    scale: f32,
    seed: f32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
    f4: f32,
    f5: f32,
    color: [f32; 4],
    color_b: [f32; 4],
}

struct GpuTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    fp: u64,
    res: u32,
}

pub struct GpuEval {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    param_bufs: Vec<wgpu::Buffer>,
    param_i: usize,
    #[allow(dead_code)]
    dummy_tex: wgpu::Texture,
    dummy_view: wgpu::TextureView,
    ramp_tex: wgpu::Texture,
    ramp_view: wgpu::TextureView,
    pool: HashMap<(String, u32), GpuTex>,
    scratch: Vec<GpuTex>,
    staging: Option<wgpu::Buffer>,
    staging_bytes: u64,
    pending_bgs: Vec<wgpu::BindGroup>,
}

impl GpuEval {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph_nodes"),
            source: wgpu::ShaderSource::Wgsl(include_str!("graph.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph_bgl"),
            entries: &[
                buf_entry(0, wgpu::ShaderStages::COMPUTE, true),
                store_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                tex_entry(5),
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("graph_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("graph_nodes"),
            layout: Some(&pipe_layout),
            module: &shader,
            entry_point: Some("node_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dummy = make_tex(device, 1, "graph_dummy");
        queue.write_texture(
            dummy.tex.as_image_copy(),
            bytemuck::bytes_of(&[1.0f32, 1.0, 1.0, 1.0]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let ramp_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("graph_ramp"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ramp_view = ramp_tex.create_view(&Default::default());
        Self {
            pipeline,
            layout,
            param_bufs: Vec::new(),
            param_i: 0,
            dummy_tex: dummy.tex,
            dummy_view: dummy.view,
            ramp_tex,
            ramp_view,
            pool: HashMap::new(),
            scratch: Vec::new(),
            staging: None,
            staging_bytes: 0,
            pending_bgs: Vec::new(),
        }
    }

    pub fn eval_material(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        nodes: &[GraphNode],
        links: &[TexLink],
        output_id: &str,
        res: u32,
    ) -> ((Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), HashMap<String, f32>) {
        let needed = ancestors_of(links, output_id);
        let mut read: Vec<String> = Vec::new();
        for port in ["albedo", "metallic", "roughness", "normal", "height"] {
            if let Some(l) = find_link(links, output_id, port) {
                read.push(l.from_node.clone());
            }
        }
        let (maps, ms) = self.eval_nodes(device, queue, nodes, links, &needed, res, &read);
        let n = (res * res) as usize;
        let albedo = maps
            .get(&src_id(links, output_id, "albedo").unwrap_or_default())
            .cloned()
            .unwrap_or_else(|| solid_u8(n, [0.7, 0.7, 0.75, 1.0]));
        let metal = maps
            .get(&src_id(links, output_id, "metallic").unwrap_or_default())
            .cloned()
            .unwrap_or_else(|| solid_u8(n, [0.0, 0.0, 0.0, 1.0]));
        let rough = maps
            .get(&src_id(links, output_id, "roughness").unwrap_or_default())
            .cloned()
            .unwrap_or_else(|| solid_u8(n, [0.45, 0.45, 0.45, 1.0]));
        let normal = maps
            .get(&src_id(links, output_id, "normal").unwrap_or_default())
            .cloned()
            .unwrap_or_else(|| solid_u8(n, [0.5, 0.5, 1.0, 1.0]));
        let height = maps
            .get(&src_id(links, output_id, "height").unwrap_or_default())
            .cloned()
            .unwrap_or_else(|| solid_u8(n, [0.5, 0.5, 0.5, 1.0]));
        let mr = pack_mr_u8(&metal, &rough);
        ((albedo, mr, normal, height), ms)
    }

    pub fn eval_previews(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        nodes: &[GraphNode],
        links: &[TexLink],
        ids: &[String],
        res: u32,
    ) -> HashMap<String, Vec<u8>> {
        let mut needed = std::collections::HashSet::new();
        for id in ids {
            needed.extend(ancestors_of(links, id));
        }
        let (maps, _) = self.eval_nodes(device, queue, nodes, links, &needed, res, ids);
        maps
    }

    fn eval_nodes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        nodes: &[GraphNode],
        links: &[TexLink],
        needed: &std::collections::HashSet<String>,
        res: u32,
        read_ids: &[String],
    ) -> (HashMap<String, Vec<u8>>, HashMap<String, f32>) {
        let res = res.max(1);
        self.param_i = 0;
        self.pool
            .retain(|(id, r), _| *r != res || needed.contains(id));
        let fps = compute_out_fingerprints(nodes, links, needed, res);
        let order = topo_order(nodes, links, needed);
        let dummy = self.dummy_view.clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("graph_eval"),
        });
        let t0 = Instant::now();
        let mut computed: Vec<String> = Vec::new();
        let mut timings: HashMap<String, f32> = HashMap::new();

        for id in &order {
            let Some(node) = nodes.iter().find(|n| n.id == *id) else {
                continue;
            };
            if matches!(node.kind, NodeKind::Output) {
                continue;
            }
            let fp = fps.get(id).copied().unwrap_or(0);
            if self
                .pool
                .get(&(id.clone(), res))
                .is_some_and(|t| t.fp == fp)
            {
                continue;
            }
            self.ensure_node(device, id, res, fp);

            if node.kind == NodeKind::FloodFill {
                encoder = self.flush(device, queue, encoder);
                let tf = Instant::now();
                let input = match find_link(links, id, "in") {
                    Some(l) => self.read_luma(device, queue, &l.from_node, res),
                    None => vec![0.0; (res * res) as usize],
                };
                let gray = flood_fill_gray(&input, res, &node.flood_fill);
                self.upload_gray(queue, id, res, &gray);
                timings.insert(id.clone(), tf.elapsed().as_secs_f32() * 1000.0);
                computed.push(id.clone());
                continue;
            }

            if node.kind == NodeKind::Blur {
                self.encode_blur(device, queue, &mut encoder, node, links, res, &dummy);
                computed.push(id.clone());
                continue;
            }

            if node.kind == NodeKind::ColorRamp {
                encoder = self.flush(device, queue, encoder);
                self.upload_ramp(queue, &node.color_ramp);
            }

            let (params, views) = self.bind_views(nodes, node, links, res, &dummy);
            self.record(device, queue, &mut encoder, params, id, res, views);
            computed.push(id.clone());
        }

        let _ = self.flush(device, queue, encoder);
        let total = t0.elapsed().as_secs_f32() * 1000.0;
        let n = computed.len().max(1) as f32;
        let share = total / n;
        for id in &computed {
            timings.entry(id.clone()).or_insert(share);
        }

        let mut maps = HashMap::new();
        for rid in read_ids {
            if rid.is_empty() {
                continue;
            }
            if let Some(rgba) = self.read_rgba8(device, queue, rid, res) {
                maps.insert(rid.clone(), rgba);
            }
        }
        (maps, timings)
    }

    fn bind_views(
        &self,
        nodes: &[GraphNode],
        node: &GraphNode,
        links: &[TexLink],
        res: u32,
        dummy: &wgpu::TextureView,
    ) -> (NodeParamsGpu, [wgpu::TextureView; 3]) {
        let view_of = |port: &str| -> (bool, wgpu::TextureView, [f32; 4]) {
            match find_link(links, &node.id, port) {
                Some(l) => {
                    let v = self
                        .pool
                        .get(&(l.from_node.clone(), res))
                        .map(|t| t.view.clone())
                        .unwrap_or_else(|| dummy.clone());
                    (true, v, [0.0; 4])
                }
                None => (false, dummy.clone(), [0.0; 4]),
            }
        };
        let mut params = pack_params(node, res);
        let mut va = dummy.clone();
        let mut vb = dummy.clone();
        let mut vc = dummy.clone();
        match node.kind {
            NodeKind::Blend => {
                let (ha, a, _) = view_of("a");
                let (hb, b, _) = view_of("b");
                let (hc, c, _) = view_of("mask");
                if ha {
                    params.flags |= 2;
                    va = a;
                } else {
                    params.color = node.blend_a;
                }
                if hb {
                    params.flags |= 4;
                    vb = b;
                } else {
                    params.color_b = node.blend_b;
                }
                if hc {
                    params.flags |= 8;
                    vc = c;
                }
            }
            NodeKind::Levels
            | NodeKind::GrayToColor
            | NodeKind::ColorToGray
            | NodeKind::Invert
            | NodeKind::Distort
            | NodeKind::Transform
            | NodeKind::TileSampler => {
                let (ha, a, _) = view_of("in");
                if ha {
                    params.flags |= 2;
                    va = a;
                } else {
                    params.color = if matches!(node.kind, NodeKind::Transform | NodeKind::TileSampler) {
                        [0.0, 0.0, 0.0, 1.0]
                    } else {
                        [0.5, 0.5, 0.5, 1.0]
                    };
                }
                if node.kind == NodeKind::Invert && src_is_color(nodes, links, node, "in") {
                    params.kind = 1;
                }
            }
            NodeKind::ColorRamp => {
                let (ha, a, _) = view_of("fac");
                if ha {
                    params.flags |= 2;
                    va = a;
                } else {
                    params.color = [0.5, 0.5, 0.5, 1.0];
                }
            }
            NodeKind::HeightToNormal | NodeKind::Curvature => {
                let (ha, a, _) = view_of("height");
                if ha {
                    params.flags |= 2;
                    va = a;
                }
            }
            NodeKind::Warp | NodeKind::DirectionalWarp => {
                let (ha, a, _) = view_of("in");
                let (hb, b, _) = view_of("drive");
                if ha {
                    params.flags |= 2;
                    va = a;
                } else {
                    params.color = [0.5, 0.5, 0.5, 1.0];
                }
                if hb {
                    params.flags |= 4;
                    vb = b;
                } else if node.kind == NodeKind::DirectionalWarp {
                    params.color_b = [1.0, 1.0, 1.0, 1.0];
                } else {
                    params.color_b = [0.5, 0.5, 0.5, 1.0];
                }
            }
            NodeKind::SlopeBlur => {
                let (ha, a, _) = view_of("in");
                let (hb, b, _) = view_of("slope");
                if ha {
                    params.flags |= 2;
                    va = a;
                } else {
                    params.color = [0.5, 0.5, 0.5, 1.0];
                }
                if hb {
                    params.flags |= 4;
                    vb = b;
                }
            }
            _ => {}
        }
        (params, [va, vb, vc])
    }

    fn encode_blur(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        node: &GraphNode,
        links: &[TexLink],
        res: u32,
        dummy: &wgpu::TextureView,
    ) {
        self.ensure_scratch(device, res);
        let src_in = find_link(links, &node.id, "in").and_then(|l| {
            self.pool
                .get(&(l.from_node.clone(), res))
                .map(|t| t.view.clone())
        });
        let drive = find_link(links, &node.id, "drive").and_then(|l| {
            self.pool
                .get(&(l.from_node.clone(), res))
                .map(|t| t.view.clone())
        });
        let out_id = node.id.clone();
        let radius = node.blur.radius.clamp(0.0, 32.0);
        let steps = radius.ceil() as u32;
        if src_in.is_none() || steps == 0 {
            let mut p = pack_params(node, res);
            p.op = 21;
            if src_in.is_some() {
                p.flags |= 2;
            } else {
                p.color = [0.5, 0.5, 0.5, 1.0];
            }
            let a = src_in.unwrap_or_else(|| dummy.clone());
            self.record(
                device,
                queue,
                encoder,
                p,
                &out_id,
                res,
                [a, dummy.clone(), dummy.clone()],
            );
            return;
        }
        let original = src_in.unwrap();
        let mut cur = original.clone();
        let last = steps;
        for i in 0..last {
            let mut ph = pack_params(node, res);
            ph.op = 17;
            ph.flags |= 2;
            let s0 = self.scratch[0].view.clone();
            let s1 = self.scratch[1].view.clone();
            self.record_to(
                device,
                queue,
                encoder,
                ph,
                &s0,
                [cur.clone(), dummy.clone(), dummy.clone()],
            );
            let mut pv = pack_params(node, res);
            pv.op = 18;
            pv.flags |= 2;
            let last_pass = i + 1 == last && drive.is_none();
            if last_pass {
                self.record(
                    device,
                    queue,
                    encoder,
                    pv,
                    &out_id,
                    res,
                    [s0, dummy.clone(), dummy.clone()],
                );
            } else {
                self.record_to(
                    device,
                    queue,
                    encoder,
                    pv,
                    &s1,
                    [s0, dummy.clone(), dummy.clone()],
                );
                cur = s1;
            }
        }
        if let Some(drv) = drive {
            let mut pm = pack_params(node, res);
            pm.op = 19;
            pm.flags |= 2 | 4 | 8;
            self.record(
                device,
                queue,
                encoder,
                pm,
                &out_id,
                res,
                [original, cur, drv],
            );
        }
    }

    fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: NodeParamsGpu,
        out_id: &str,
        res: u32,
        ins: [wgpu::TextureView; 3],
    ) {
        let out = self.pool[&(out_id.to_string(), res)].view.clone();
        self.record_to(device, queue, encoder, params, &out, ins);
    }

    fn record_to(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: NodeParamsGpu,
        out: &wgpu::TextureView,
        ins: [wgpu::TextureView; 3],
    ) {
        let i = self.alloc_param(device);
        queue.write_buffer(&self.param_bufs[i], 0, bytemuck::bytes_of(&params));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("graph_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.param_bufs[i].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(out),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ins[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&ins[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ins[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.ramp_view),
                },
            ],
        });
        self.pending_bgs.push(bg);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("graph_node"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.pending_bgs.last().unwrap(), &[]);
            let g = params.res.div_ceil(8);
            pass.dispatch_workgroups(g, g, 1);
        }
    }

    fn flush(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: wgpu::CommandEncoder,
    ) -> wgpu::CommandEncoder {
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.pending_bgs.clear();
        self.param_i = 0;
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("graph_eval"),
        })
    }

    fn alloc_param(&mut self, device: &wgpu::Device) -> usize {
        if self.param_i >= self.param_bufs.len() {
            self.param_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("graph_params"),
                size: 256,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let i = self.param_i;
        self.param_i += 1;
        i
    }

    fn ensure_node(&mut self, device: &wgpu::Device, id: &str, res: u32, fp: u64) {
        let key = (id.to_string(), res);
        if let Some(t) = self.pool.get_mut(&key) {
            t.fp = fp;
            return;
        }
        let t = make_tex(device, res, "graph_node");
        self.pool.insert(
            key,
            GpuTex {
                tex: t.tex,
                view: t.view,
                fp,
                res,
            },
        );
    }

    fn ensure_scratch(&mut self, device: &wgpu::Device, res: u32) {
        while self.scratch.len() < 2 {
            let t = make_tex(device, res, "graph_scratch");
            self.scratch.push(GpuTex {
                tex: t.tex,
                view: t.view,
                fp: 0,
                res,
            });
        }
        for s in &mut self.scratch {
            if s.res != res {
                let t = make_tex(device, res, "graph_scratch");
                s.tex = t.tex;
                s.view = t.view;
                s.res = res;
            }
        }
    }

    fn upload_ramp(&self, queue: &wgpu::Queue, ramp: &ColorRampParams) {
        let mut px = vec![0.0f32; 256 * 4];
        for i in 0..256 {
            let c = sample_gradient(&ramp.colors, &ramp.opacities, i as f32 / 255.0);
            let o = i * 4;
            px[o] = c[0];
            px[o + 1] = c[1];
            px[o + 2] = c[2];
            px[o + 3] = c[3];
        }
        queue.write_texture(
            self.ramp_tex.as_image_copy(),
            bytemuck::cast_slice(&px),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_gray(&self, queue: &wgpu::Queue, id: &str, res: u32, gray: &[f32]) {
        let Some(slot) = self.pool.get(&(id.to_string(), res)) else {
            return;
        };
        let row = padded_row_bytes(res);
        let mut bytes = vec![0u8; row as usize * res as usize];
        for y in 0..res as usize {
            let dst = &mut bytes[y * row as usize..][..res as usize * 16];
            let f: &mut [f32] = bytemuck::cast_slice_mut(dst);
            for x in 0..res as usize {
                let g = gray[y * res as usize + x];
                let o = x * 4;
                f[o] = g;
                f[o + 1] = g;
                f[o + 2] = g;
                f[o + 3] = 1.0;
            }
        }
        queue.write_texture(
            slot.tex.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(res),
            },
            wgpu::Extent3d {
                width: res,
                height: res,
                depth_or_array_layers: 1,
            },
        );
    }

    fn read_luma(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        res: u32,
    ) -> Vec<f32> {
        let Some(rgba) = self.read_f32(device, queue, id, res) else {
            return vec![0.0; (res * res) as usize];
        };
        let n = (res * res) as usize;
        (0..n)
            .map(|i| {
                let o = i * 4;
                0.2126 * rgba[o] + 0.7152 * rgba[o + 1] + 0.0722 * rgba[o + 2]
            })
            .collect()
    }

    fn read_rgba8(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        res: u32,
    ) -> Option<Vec<u8>> {
        let rgba = self.read_f32(device, queue, id, res)?;
        let n = (res * res) as usize;
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            let o = i * 4;
            out.push((rgba[o].clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push((rgba[o + 1].clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push((rgba[o + 2].clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push((rgba[o + 3].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        Some(out)
    }

    fn read_f32(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: &str,
        res: u32,
    ) -> Option<Vec<f32>> {
        let slot = self.pool.get(&(id.to_string(), res))?;
        let row = padded_row_bytes(res);
        let bytes = row as u64 * res as u64;
        if self.staging.as_ref().is_none_or(|_| self.staging_bytes < bytes) {
            self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("graph_staging"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            self.staging_bytes = bytes;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("graph_read"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &slot.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: self.staging.as_ref().unwrap(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(res),
                },
            },
            wgpu::Extent3d {
                width: res,
                height: res,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let staging = self.staging.as_ref().unwrap();
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map_async");
        let mapped = slice.get_mapped_range().expect("mapped");
        let n = (res * res) as usize;
        let mut data = Vec::with_capacity(n * 4);
        for y in 0..res as usize {
            let off = y * row as usize;
            let row_f: &[f32] = bytemuck::cast_slice(&mapped[off..off + row as usize]);
            data.extend_from_slice(&row_f[..res as usize * 4]);
        }
        drop(mapped);
        staging.unmap();
        Some(data)
    }
}

fn pack_params(n: &GraphNode, res: u32) -> NodeParamsGpu {
    let mut g = NodeParamsGpu {
        res,
        op: 0,
        kind: 0,
        octaves: 1,
        flags: 0,
        _p0: 0,
        _p1: 0,
        _p2: 0,
        scale: 1.0,
        seed: 0.0,
        f0: 0.0,
        f1: 0.0,
        f2: 0.0,
        f3: 0.0,
        f4: 0.0,
        f5: 0.0,
        color: [0.0; 4],
        color_b: [0.0; 4],
    };
    match n.kind {
        NodeKind::Noise => {
            g.op = 0;
            g.kind = noise_kind(n.noise.kind);
            g.octaves = n.noise.octaves.clamp(1, 8) as u32;
            if n.noise.tileable {
                g.flags |= 1;
            }
            g.scale = n.noise.scale;
            g.seed = n.noise.seed;
            g.f0 = n.noise.angle.to_radians();
            g.f1 = n.noise.stretch.max(1.0);
        }
        NodeKind::Color => {
            g.op = 1;
            g.color = n.color;
        }
        NodeKind::Gradient => {
            g.op = 2;
            g.kind = match n.gradient_mode {
                GradientMode::Linear => 0,
                GradientMode::Radial => 1,
            };
        }
        NodeKind::Lines => {
            let ln = &n.lines;
            g.op = 3;
            g.f0 = ln.width.clamp(0.0, 1.0);
            g.scale = ln.count.clamp(1, 64) as f32;
            g.f2 = ln.rotation.to_radians();
            g.f3 = ln.intensity.clamp(0.0, 1.0);
            g.f4 = ln.bg_intensity.clamp(0.0, 1.0);
        }
        NodeKind::Checker => {
            let c = &n.checker;
            g.op = 4;
            g.scale = c.scale.round().max(1.0);
            g.f0 = c.intensity_a.clamp(0.0, 1.0);
            g.f1 = c.intensity_b.clamp(0.0, 1.0);
        }
        NodeKind::Tile => {
            let t = &n.tile;
            g.op = 5;
            g.octaves = t.x_amount.clamp(1, 32) as u32;
            g.kind = t.y_amount.clamp(1, 32) as u32;
            g.f0 = t.gap.clamp(0.0, 0.4);
            g.f1 = t.size_rand.clamp(0.0, 1.0);
            g.f2 = t.offset.clamp(0.0, 1.0);
            g.f3 = t.roundness.clamp(0.0, 1.0);
            g.seed = t.seed;
        }
        NodeKind::Bricks => {
            let b = &n.bricks;
            g.op = 6;
            g.scale = b.x_amount.clamp(1, 64) as f32;
            g.f5 = b.y_amount.clamp(1, 64) as f32;
            g.f0 = b.gap.clamp(0.0, 0.4);
            g.f1 = b.offset.clamp(0.0, 1.0);
            g.f2 = b.roundness.clamp(0.0, 1.0);
            g.f3 = b.bevel.clamp(0.0, 1.0);
        }
        NodeKind::Invert => {
            g.op = 7;
        }
        NodeKind::Levels => {
            let l = &n.levels;
            g.op = 8;
            g.f0 = l.in_black;
            g.f1 = l.in_white;
            g.f2 = l.gamma;
            g.f3 = l.out_black;
            g.f4 = l.out_white;
        }
        NodeKind::GrayToColor => g.op = 9,
        NodeKind::ColorToGray => g.op = 10,
        NodeKind::ColorRamp => g.op = 11,
        NodeKind::Blend => {
            g.op = 12;
            g.kind = blend_kind(n.blend_mode);
            g.f0 = n.mix.clamp(0.0, 1.0);
            g.color = n.blend_a;
            g.color_b = n.blend_b;
        }
        NodeKind::HeightToNormal => {
            g.op = 13;
            g.f0 = n.normal_strength.max(0.0);
        }
        NodeKind::Curvature => {
            g.op = 22;
            g.f0 = n.curvature.intensity.max(0.0);
            g.f1 = n.curvature.radius.max(1) as f32;
        }
        NodeKind::Distort => {
            g.op = 14;
            g.f0 = n.distort.strength.clamp(0.0, 1.0);
            g.scale = n.distort.scale.max(0.01);
            g.seed = n.distort.seed;
        }
        NodeKind::Warp => {
            g.op = 15;
            g.f0 = n.warp.strength.clamp(0.0, 2.0);
        }
        NodeKind::DirectionalWarp => {
            g.op = 16;
            g.f0 = n.dir_warp.intensity.clamp(0.0, 2.0);
            g.f1 = n.dir_warp.angle.to_radians();
        }
        NodeKind::Blur => {
            g.op = 17;
            g.f0 = n.blur.radius;
        }
        NodeKind::SlopeBlur => {
            g.op = 20;
            g.f0 = n.slope_blur.intensity.clamp(0.0, 2.0);
            g.octaves = n.slope_blur.samples.clamp(1, 32) as u32;
            g.kind = match n.slope_blur.mode {
                SlopeBlurMode::Blur => 0,
                SlopeBlurMode::Min => 1,
                SlopeBlurMode::Max => 2,
            };
        }
        NodeKind::Shape => {
            g.op = 23;
            g.kind = match n.shape.kind {
                ShapeKind::Rectangle => 0,
                ShapeKind::Circle => 1,
                ShapeKind::Triangle => 2,
                ShapeKind::NGon => 3,
            };
            g.f0 = n.shape.size_x.max(0.0);
            g.f1 = n.shape.size_y.max(0.0);
            g.octaves = n.shape.sides.clamp(3, 16) as u32;
        }
        NodeKind::Transform => {
            g.op = 24;
            g.f0 = n.transform.offset_x;
            g.f1 = n.transform.offset_y;
            g.f2 = n.transform.scale_x.max(1e-6);
            g.f3 = n.transform.scale_y.max(1e-6);
            g.f4 = n.transform.rotation.to_radians();
            if n.transform.tileable {
                g.flags |= 1;
            }
        }
        NodeKind::TileSampler => {
            let t = &n.tile_sampler;
            g.op = 25;
            g.octaves = t.x_amount.clamp(1, 32) as u32;
            g.kind = t.y_amount.clamp(1, 32) as u32;
            g.f0 = t.offset_rand.clamp(0.0, 1.0);
            g.f1 = t.rotation_rand.clamp(0.0, 1.0);
            g.f2 = t.scale_rand.clamp(0.0, 1.0);
            g.seed = t.seed;
        }
        NodeKind::FloodFill | NodeKind::Output => {}
    }
    g
}

fn src_is_color(nodes: &[GraphNode], links: &[TexLink], node: &GraphNode, port: &str) -> bool {
    find_link(links, &node.id, port)
        .and_then(|l| nodes.iter().find(|n| n.id == l.from_node))
        .is_some_and(|n| {
            matches!(
                n.kind,
                NodeKind::Color
                    | NodeKind::Blend
                    | NodeKind::HeightToNormal
                    | NodeKind::GrayToColor
                    | NodeKind::ColorRamp
            )
        })
}

fn src_id(links: &[TexLink], to: &str, port: &str) -> Option<String> {
    find_link(links, to, port).map(|l| l.from_node.clone())
}

fn noise_kind(k: NoiseType) -> u32 {
    match k {
        NoiseType::Value => 0,
        NoiseType::Perlin => 1,
        NoiseType::Voronoi => 2,
        NoiseType::VoronoiEdge => 3,
        NoiseType::Gauss => 4,
        NoiseType::Cloud => 5,
        NoiseType::Anisotropic => 6,
    }
}

fn blend_kind(m: BlendMode) -> u32 {
    match m {
        BlendMode::Mix => 0,
        BlendMode::Multiply => 1,
        BlendMode::Add => 2,
        BlendMode::Overlay => 3,
        BlendMode::Screen => 4,
        BlendMode::Divide => 5,
        BlendMode::Subtract => 6,
        BlendMode::Difference => 7,
        BlendMode::Darken => 8,
        BlendMode::Lighten => 9,
    }
}

struct TexPair {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

fn make_tex(device: &wgpu::Device, res: u32, label: &str) -> TexPair {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: res.max(1),
            height: res.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    TexPair { tex, view }
}

fn padded_row_bytes(res: u32) -> u32 {
    let raw = res * 16;
    raw.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

fn buf_entry(binding: u32, vis: wgpu::ShaderStages, uniform: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: if uniform {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only: true }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn store_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn solid_u8(n: usize, c: [f32; 4]) -> Vec<u8> {
    let mut o = Vec::with_capacity(n * 4);
    for _ in 0..n {
        o.push((c[0] * 255.0).round() as u8);
        o.push((c[1] * 255.0).round() as u8);
        o.push((c[2] * 255.0).round() as u8);
        o.push((c[3] * 255.0).round() as u8);
    }
    o
}

fn pack_mr_u8(metal: &[u8], rough: &[u8]) -> Vec<u8> {
    let n = metal.len() / 4;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let o = i * 4;
        let m = luma8(metal[o], metal[o + 1], metal[o + 2]);
        let r = luma8(rough[o], rough[o + 1], rough[o + 2]);
        out.push(255);
        out.push(r);
        out.push(m);
        out.push(255);
    }
    out
}

fn luma8(r: u8, g: u8, b: u8) -> u8 {
    ((0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32).round() as u8).min(255)
}
