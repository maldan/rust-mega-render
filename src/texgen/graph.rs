//! Graph wiring + content hashes + flood-fill (CPU).

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use glam::Vec2;

use super::node::{
    FloodFillParams, GraphNode, NodeKind,
};

/// Directed edge between node ports. No UI identity.
#[derive(Clone, Debug)]
pub struct TexLink {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

impl TexLink {
    pub fn new(
        from_node: impl Into<String>,
        from_port: impl Into<String>,
        to_node: impl Into<String>,
        to_port: impl Into<String>,
    ) -> Self {
        Self {
            from_node: from_node.into(),
            from_port: from_port.into(),
            to_node: to_node.into(),
            to_port: to_port.into(),
        }
    }
}

/// Nodes + links + output. Enough to bake a PBR material.
#[derive(Clone, Debug)]
pub struct TexGraph {
    pub nodes: Vec<GraphNode>,
    pub links: Vec<TexLink>,
    pub output_id: String,
    next: u64,
}

impl Default for TexGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TexGraph {
    pub fn new() -> Self {
        let mut g = Self {
            nodes: Vec::new(),
            links: Vec::new(),
            output_id: String::new(),
            next: 1,
        };
        let id = g.add(NodeKind::Output);
        g.output_id = id;
        g
    }

    pub fn add(&mut self, kind: NodeKind) -> String {
        let id = format!("n{}", self.next);
        self.next += 1;
        self.nodes
            .push(GraphNode::new(id.clone(), kind, Vec2::ZERO));
        id
    }

    pub fn connect(&mut self, from: &str, from_port: &str, to: &str, to_port: &str) {
        self.links
            .retain(|l| !(l.to_node == to && l.to_port == to_port));
        self.links.push(TexLink::new(from, from_port, to, to_port));
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn output(&self) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == self.output_id)
    }
}

pub fn find_link<'a>(links: &'a [TexLink], to_node: &str, to_port: &str) -> Option<&'a TexLink> {
    links
        .iter()
        .find(|l| l.to_node == to_node && l.to_port == to_port)
}

pub fn ancestors_of(links: &[TexLink], root: &str) -> HashSet<String> {
    let mut needed = HashSet::new();
    let mut stack = vec![root.to_string()];
    while let Some(id) = stack.pop() {
        if !needed.insert(id.clone()) {
            continue;
        }
        for l in links {
            if l.to_node == id {
                stack.push(l.from_node.clone());
            }
        }
    }
    needed
}

pub fn topo_order(
    nodes: &[GraphNode],
    links: &[TexLink],
    needed: &HashSet<String>,
) -> Vec<String> {
    let ids: Vec<String> = nodes
        .iter()
        .filter(|n| needed.contains(&n.id))
        .map(|n| n.id.clone())
        .collect();
    let mut indeg: HashMap<String, usize> = ids.iter().map(|id| (id.clone(), 0)).collect();
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for link in links {
        if !needed.contains(&link.from_node) || !needed.contains(&link.to_node) {
            continue;
        }
        adj.entry(link.from_node.clone())
            .or_default()
            .push(link.to_node.clone());
        *indeg.entry(link.to_node.clone()).or_default() += 1;
    }
    let mut q: VecDeque<String> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    while let Some(id) = q.pop_front() {
        if !seen.insert(id.clone()) {
            continue;
        }
        out.push(id.clone());
        if let Some(next) = adj.get(&id) {
            for n in next {
                if let Some(d) = indeg.get_mut(n) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        q.push_back(n.clone());
                    }
                }
            }
        }
    }
    for id in ids {
        if !seen.contains(&id) {
            out.push(id);
        }
    }
    out
}

/// Content hash of each node's `out` (params + upstream + resolution).
pub fn compute_out_fingerprints(
    nodes: &[GraphNode],
    links: &[TexLink],
    needed: &HashSet<String>,
    res: u32,
) -> HashMap<String, u64> {
    let order = topo_order(nodes, links, needed);
    let mut ports: HashMap<(String, String), u64> = HashMap::new();
    let mut out = HashMap::new();
    for id in order {
        let Some(node) = nodes.iter().find(|n| n.id == id) else {
            continue;
        };
        if matches!(node.kind, NodeKind::Output) {
            continue;
        }
        let up = |port: &str| -> u64 {
            find_link(links, &id, port)
                .and_then(|l| {
                    ports
                        .get(&(l.from_node.clone(), l.from_port.clone()))
                        .copied()
                })
                .unwrap_or(0)
        };
        let mut fp = node_param_hash(node);
        match node.kind {
            NodeKind::Blend => {
                fp = hash_combine(fp, up("a"));
                fp = hash_combine(fp, up("b"));
                fp = hash_combine(fp, up("mask"));
            }
            NodeKind::Levels
            | NodeKind::GrayToColor
            | NodeKind::ColorToGray
            | NodeKind::Invert
            | NodeKind::FloodFill => {
                fp = hash_combine(fp, up("in"));
            }
            NodeKind::HeightToNormal | NodeKind::Curvature => {
                fp = hash_combine(fp, up("height"));
            }
            NodeKind::ColorRamp => {
                fp = hash_combine(fp, up("fac"));
            }
            NodeKind::Distort => {
                fp = hash_combine(fp, up("in"));
            }
            NodeKind::Warp | NodeKind::DirectionalWarp | NodeKind::Blur => {
                fp = hash_combine(fp, up("in"));
                fp = hash_combine(fp, up("drive"));
            }
            NodeKind::SlopeBlur => {
                fp = hash_combine(fp, up("in"));
                fp = hash_combine(fp, up("slope"));
            }
            NodeKind::Transform => {
                fp = hash_combine(fp, up("in"));
            }
            NodeKind::TileSampler => {
                fp = hash_combine(fp, up("in"));
            }
            _ => {}
        }
        fp = hash_combine(fp, res as u64);
        ports.insert((id.clone(), "out".into()), fp);
        out.insert(id, fp);
    }
    out
}

fn hash_combine(a: u64, b: u64) -> u64 {
    a ^ b.wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(a << 6)
        .wrapping_add(a >> 2)
}

fn node_param_hash(node: &GraphNode) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::mem::discriminant(&node.kind).hash(&mut h);
    match node.kind {
        NodeKind::Color => {
            for c in node.color {
                c.to_bits().hash(&mut h);
            }
        }
        NodeKind::Noise => {
            std::mem::discriminant(&node.noise.kind).hash(&mut h);
            node.noise.scale.to_bits().hash(&mut h);
            node.noise.octaves.hash(&mut h);
            node.noise.seed.to_bits().hash(&mut h);
            node.noise.tileable.hash(&mut h);
            node.noise.angle.to_bits().hash(&mut h);
            node.noise.stretch.to_bits().hash(&mut h);
        }
        NodeKind::Gradient => {
            std::mem::discriminant(&node.gradient_mode).hash(&mut h);
        }
        NodeKind::Lines => {
            node.lines.width.to_bits().hash(&mut h);
            node.lines.count.hash(&mut h);
            node.lines.rotation.to_bits().hash(&mut h);
            node.lines.intensity.to_bits().hash(&mut h);
            node.lines.bg_intensity.to_bits().hash(&mut h);
        }
        NodeKind::Distort => {
            node.distort.strength.to_bits().hash(&mut h);
            node.distort.scale.to_bits().hash(&mut h);
            node.distort.seed.to_bits().hash(&mut h);
        }
        NodeKind::Warp => {
            node.warp.strength.to_bits().hash(&mut h);
        }
        NodeKind::DirectionalWarp => {
            node.dir_warp.intensity.to_bits().hash(&mut h);
            node.dir_warp.angle.to_bits().hash(&mut h);
        }
        NodeKind::Blur => {
            node.blur.radius.to_bits().hash(&mut h);
        }
        NodeKind::SlopeBlur => {
            node.slope_blur.intensity.to_bits().hash(&mut h);
            node.slope_blur.samples.hash(&mut h);
            std::mem::discriminant(&node.slope_blur.mode).hash(&mut h);
        }
        NodeKind::Checker => {
            node.checker.intensity_a.to_bits().hash(&mut h);
            node.checker.intensity_b.to_bits().hash(&mut h);
            node.checker.scale.to_bits().hash(&mut h);
        }
        NodeKind::Tile => {
            node.tile.x_amount.hash(&mut h);
            node.tile.y_amount.hash(&mut h);
            node.tile.gap.to_bits().hash(&mut h);
            node.tile.size_rand.to_bits().hash(&mut h);
            node.tile.offset.to_bits().hash(&mut h);
            node.tile.roundness.to_bits().hash(&mut h);
            node.tile.seed.to_bits().hash(&mut h);
        }
        NodeKind::Bricks => {
            node.bricks.x_amount.hash(&mut h);
            node.bricks.y_amount.hash(&mut h);
            node.bricks.gap.to_bits().hash(&mut h);
            node.bricks.offset.to_bits().hash(&mut h);
            node.bricks.roundness.to_bits().hash(&mut h);
            node.bricks.bevel.to_bits().hash(&mut h);
        }
        NodeKind::FloodFill => {
            node.flood_fill.seed.to_bits().hash(&mut h);
            node.flood_fill.threshold.to_bits().hash(&mut h);
            node.flood_fill.luma_min.to_bits().hash(&mut h);
            node.flood_fill.luma_max.to_bits().hash(&mut h);
        }
        NodeKind::Blend => {
            std::mem::discriminant(&node.blend_mode).hash(&mut h);
            node.mix.to_bits().hash(&mut h);
            for c in node.blend_a {
                c.to_bits().hash(&mut h);
            }
            for c in node.blend_b {
                c.to_bits().hash(&mut h);
            }
        }
        NodeKind::Levels => {
            node.levels.in_black.to_bits().hash(&mut h);
            node.levels.in_white.to_bits().hash(&mut h);
            node.levels.gamma.to_bits().hash(&mut h);
            node.levels.out_black.to_bits().hash(&mut h);
            node.levels.out_white.to_bits().hash(&mut h);
        }
        NodeKind::HeightToNormal => {
            node.normal_strength.to_bits().hash(&mut h);
        }
        NodeKind::Curvature => {
            node.curvature.intensity.to_bits().hash(&mut h);
            node.curvature.radius.hash(&mut h);
        }
        NodeKind::ColorRamp => {
            node.color_ramp.colors.len().hash(&mut h);
            for s in &node.color_ramp.colors {
                s.t.to_bits().hash(&mut h);
                for c in s.color {
                    c.to_bits().hash(&mut h);
                }
            }
            node.color_ramp.opacities.len().hash(&mut h);
            for s in &node.color_ramp.opacities {
                s.t.to_bits().hash(&mut h);
                s.alpha.to_bits().hash(&mut h);
            }
        }
        NodeKind::GrayToColor | NodeKind::ColorToGray | NodeKind::Invert | NodeKind::Output => {}
        NodeKind::Shape => {
            std::mem::discriminant(&node.shape.kind).hash(&mut h);
            node.shape.size_x.to_bits().hash(&mut h);
            node.shape.size_y.to_bits().hash(&mut h);
            node.shape.sides.hash(&mut h);
        }
        NodeKind::Transform => {
            node.transform.offset_x.to_bits().hash(&mut h);
            node.transform.offset_y.to_bits().hash(&mut h);
            node.transform.scale_x.to_bits().hash(&mut h);
            node.transform.scale_y.to_bits().hash(&mut h);
            node.transform.rotation.to_bits().hash(&mut h);
            node.transform.tileable.hash(&mut h);
        }
        NodeKind::TileSampler => {
            node.tile_sampler.x_amount.hash(&mut h);
            node.tile_sampler.y_amount.hash(&mut h);
            node.tile_sampler.offset_rand.to_bits().hash(&mut h);
            node.tile_sampler.rotation_rand.to_bits().hash(&mut h);
            node.tile_sampler.scale_rand.to_bits().hash(&mut h);
            node.tile_sampler.seed.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Union-find flood fill → random gray per island. `gray` is luma, length `res*res`.
pub fn flood_fill_gray(gray: &[f32], res: u32, params: &FloodFillParams) -> Vec<f32> {
    let res = res.max(1);
    let w = res as i32;
    let n = (res * res) as usize;
    let thresh = params.threshold.clamp(0.0, 1.0);
    let lo = params.luma_min.clamp(0.0, 1.0);
    let hi = params.luma_max.clamp(0.0, 1.0);
    let mut parent: Vec<u32> = (0..n as u32).collect();

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    fn union(parent: &mut [u32], a: u32, b: u32) {
        let mut ra = find(parent, a);
        let mut rb = find(parent, b);
        if ra == rb {
            return;
        }
        if ra > rb {
            std::mem::swap(&mut ra, &mut rb);
        }
        parent[rb as usize] = ra;
    }

    let fg = |i: usize| gray.get(i).copied().unwrap_or(0.0) > thresh;
    for y in 0..w {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if !fg(i) {
                continue;
            }
            let ir = (y * w + (x + 1) % w) as usize;
            if fg(ir) {
                union(&mut parent, i as u32, ir as u32);
            }
            let idn = (((y + 1) % w) * w + x) as usize;
            if fg(idn) {
                union(&mut parent, i as u32, idn as u32);
            }
        }
    }

    let mut data = vec![0.0; n];
    for i in 0..n {
        if !fg(i) {
            continue;
        }
        let root = find(&mut parent, i as u32);
        data[i] = lo + (hi - lo) * hash1(root as i32, params.seed);
    }
    data
}

fn hash1(i: i32, seed: f32) -> f32 {
    let mut n = (i as u32)
        .wrapping_mul(374761393)
        ^ seed.to_bits().wrapping_mul(2246822519);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    n ^= n >> 16;
    (n as f32) / (u32::MAX as f32)
}
