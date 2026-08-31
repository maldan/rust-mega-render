//! Binary IR for [`super::TexGraph`] (TEXG payload).

use glam::Vec2;

use crate::HeightMode;

use super::graph::{TexGraph, TexLink};
use super::node::{
    BlendMode, BlurParams, BricksParams, CheckerParams, CurvatureParams, DistortParams,
    DirWarpParams, FloodFillParams, GradientMode, GradientStop, GraphNode, LevelsParams,
    LinesParams, NodeKind, NoiseParams, NoiseType, OpacityStop, ShapeKind, ShapeParams,
    SlopeBlurMode, SlopeBlurParams, TileParams, TileSamplerParams, TransformParams, WarpParams,
};

const VERSION: u32 = 1;

/// Bake recipe: graph + resolution. Lives on [`crate::MaterialFile`], not on `Material`.
#[derive(Clone, Debug)]
pub struct TexGraphFile {
    pub resolution: u32,
    pub graph: TexGraph,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TexGraphBytesError {
    Truncated,
    UnsupportedVersion(u32),
    BadUtf8,
    UnknownNodeKind(u8),
    NoOutput,
}

impl std::fmt::Display for TexGraphBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated TEXG payload"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported TEXG IR version {v}"),
            Self::BadUtf8 => write!(f, "TEXG string is not UTF-8"),
            Self::UnknownNodeKind(k) => write!(f, "unknown TEXG node kind {k}"),
            Self::NoOutput => write!(f, "TEXG graph has no Output node"),
        }
    }
}

impl std::error::Error for TexGraphBytesError {}

impl TexGraphFile {
    pub fn to_bytes(&self) -> Vec<u8> {
        to_bytes(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TexGraphBytesError> {
        from_bytes(bytes)
    }
}

pub fn to_bytes(file: &TexGraphFile) -> Vec<u8> {
    let mut o = Vec::new();
    write_u32(&mut o, VERSION);
    write_u32(&mut o, file.resolution.max(1));
    write_u64(&mut o, file.graph.next_serial());
    write_str(&mut o, &file.graph.output_id);
    write_u32(&mut o, file.graph.nodes.len() as u32);
    for n in &file.graph.nodes {
        write_node(&mut o, n);
    }
    write_u32(&mut o, file.graph.links.len() as u32);
    for l in &file.graph.links {
        write_str(&mut o, &l.from_node);
        write_str(&mut o, &l.from_port);
        write_str(&mut o, &l.to_node);
        write_str(&mut o, &l.to_port);
    }
    o
}

pub fn from_bytes(bytes: &[u8]) -> Result<TexGraphFile, TexGraphBytesError> {
    let mut r = Reader { bytes, pos: 0 };
    let version = r.u32()?;
    if version != VERSION {
        return Err(TexGraphBytesError::UnsupportedVersion(version));
    }
    let resolution = r.u32()?.clamp(64, 2048);
    let next = r.u64()?;
    let output_id = r.str()?;
    let n_nodes = r.u32()? as usize;
    let mut nodes = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        nodes.push(read_node(&mut r)?);
    }
    let n_links = r.u32()? as usize;
    let mut links = Vec::with_capacity(n_links);
    for _ in 0..n_links {
        links.push(TexLink::new(r.str()?, r.str()?, r.str()?, r.str()?));
    }
    if r.pos != bytes.len() {
        return Err(TexGraphBytesError::Truncated);
    }
    let graph = TexGraph::from_decoded(nodes, links, output_id, next)?;
    Ok(TexGraphFile {
        resolution,
        graph,
    })
}

fn write_node(o: &mut Vec<u8>, n: &GraphNode) {
    o.push(n.kind.to_u8());
    write_str(o, &n.id);
    write_f32s(o, &[n.pos.x, n.pos.y]);
    write_f32s(o, &n.color);
    o.push(noise_u8(n.noise.kind));
    write_f32s(o, &[n.noise.scale]);
    write_i32(o, n.noise.octaves);
    write_f32s(o, &[n.noise.seed]);
    o.push(u8::from(n.noise.tileable));
    write_f32s(o, &[n.noise.angle, n.noise.stretch]);
    o.push(blend_u8(n.blend_mode));
    write_f32s(o, &[n.mix]);
    write_f32s(o, &n.blend_a);
    write_f32s(o, &n.blend_b);
    o.push(grad_u8(n.gradient_mode));
    write_f32s(
        o,
        &[
            n.levels.in_black,
            n.levels.in_white,
            n.levels.gamma,
            n.levels.out_black,
            n.levels.out_white,
        ],
    );
    write_f32s(o, &[n.normal_strength, n.curvature.intensity]);
    write_i32(o, n.curvature.radius);
    write_u16(o, n.color_ramp.colors.len().min(u16::MAX as usize) as u16);
    for s in &n.color_ramp.colors {
        write_f32s(o, &[s.t]);
        write_f32s(o, &s.color);
    }
    write_u16(
        o,
        n.color_ramp.opacities.len().min(u16::MAX as usize) as u16,
    );
    for s in &n.color_ramp.opacities {
        write_f32s(o, &[s.t, s.alpha]);
    }
    write_f32s(o, &[n.lines.width]);
    write_i32(o, n.lines.count);
    write_f32s(
        o,
        &[
            n.lines.rotation,
            n.lines.intensity,
            n.lines.bg_intensity,
            n.distort.strength,
            n.distort.scale,
            n.distort.seed,
            n.warp.strength,
            n.dir_warp.intensity,
            n.dir_warp.angle,
            n.blur.radius,
            n.slope_blur.intensity,
        ],
    );
    write_i32(o, n.slope_blur.samples);
    o.push(slope_u8(n.slope_blur.mode));
    write_f32s(
        o,
        &[
            n.checker.intensity_a,
            n.checker.intensity_b,
            n.checker.scale,
        ],
    );
    write_i32(o, n.tile.x_amount);
    write_i32(o, n.tile.y_amount);
    write_f32s(
        o,
        &[
            n.tile.gap,
            n.tile.size_rand_x,
            n.tile.size_rand_y,
            n.tile.offset,
            n.tile.roundness,
            n.tile.seed,
        ],
    );
    write_i32(o, n.bricks.x_amount);
    write_i32(o, n.bricks.y_amount);
    write_f32s(
        o,
        &[
            n.bricks.gap,
            n.bricks.offset,
            n.bricks.roundness,
            n.bricks.bevel,
        ],
    );
    write_f32s(
        o,
        &[
            n.flood_fill.seed,
            n.flood_fill.threshold,
            n.flood_fill.luma_min,
            n.flood_fill.luma_max,
        ],
    );
    o.push(shape_u8(n.shape.kind));
    write_f32s(o, &[n.shape.size_x, n.shape.size_y]);
    write_i32(o, n.shape.sides);
    write_f32s(
        o,
        &[
            n.transform.offset_x,
            n.transform.offset_y,
            n.transform.scale_x,
            n.transform.scale_y,
            n.transform.rotation,
        ],
    );
    o.push(u8::from(n.transform.tileable));
    write_i32(o, n.tile_sampler.x_amount);
    write_i32(o, n.tile_sampler.y_amount);
    write_f32s(
        o,
        &[
            n.tile_sampler.offset_rand,
            n.tile_sampler.rotation_rand,
            n.tile_sampler.scale_rand,
            n.tile_sampler.seed,
        ],
    );
    write_i32(o, n.tess_factor);
    write_f32s(o, &[n.displacement]);
    o.push(height_u8(n.height_mode));
}

fn read_node(r: &mut Reader) -> Result<GraphNode, TexGraphBytesError> {
    let kind_id = r.u8()?;
    let kind = NodeKind::from_u8(kind_id).ok_or(TexGraphBytesError::UnknownNodeKind(kind_id))?;
    let id = r.str()?;
    let pos = r.f32s::<2>()?;
    let mut n = GraphNode::new(id, kind, Vec2::new(pos[0], pos[1]));
    n.color = r.f32s::<4>()?;
    n.noise = NoiseParams {
        kind: noise_from(r.u8()?),
        scale: r.f32()?,
        octaves: r.i32()?,
        seed: r.f32()?,
        tileable: r.u8()? != 0,
        angle: r.f32()?,
        stretch: r.f32()?,
    };
    n.blend_mode = blend_from(r.u8()?);
    n.mix = r.f32()?;
    n.blend_a = r.f32s::<4>()?;
    n.blend_b = r.f32s::<4>()?;
    n.gradient_mode = if r.u8()? == 1 {
        GradientMode::Radial
    } else {
        GradientMode::Linear
    };
    n.levels = LevelsParams {
        in_black: r.f32()?,
        in_white: r.f32()?,
        gamma: r.f32()?,
        out_black: r.f32()?,
        out_white: r.f32()?,
    };
    n.normal_strength = r.f32()?;
    n.curvature = CurvatureParams {
        intensity: r.f32()?,
        radius: r.i32()?,
    };
    let n_col = r.u16()? as usize;
    n.color_ramp.colors = Vec::with_capacity(n_col);
    for _ in 0..n_col {
        let t = r.f32()?;
        let color = r.f32s::<4>()?;
        n.color_ramp.colors.push(GradientStop { t, color });
    }
    let n_op = r.u16()? as usize;
    n.color_ramp.opacities = Vec::with_capacity(n_op);
    for _ in 0..n_op {
        n.color_ramp.opacities.push(OpacityStop {
            t: r.f32()?,
            alpha: r.f32()?,
        });
    }
    n.lines = LinesParams {
        width: r.f32()?,
        count: r.i32()?,
        rotation: r.f32()?,
        intensity: r.f32()?,
        bg_intensity: r.f32()?,
    };
    n.distort = DistortParams {
        strength: r.f32()?,
        scale: r.f32()?,
        seed: r.f32()?,
    };
    n.warp = WarpParams {
        strength: r.f32()?,
    };
    n.dir_warp = DirWarpParams {
        intensity: r.f32()?,
        angle: r.f32()?,
    };
    n.blur = BlurParams { radius: r.f32()? };
    n.slope_blur = SlopeBlurParams {
        intensity: r.f32()?,
        samples: r.i32()?,
        mode: slope_from(r.u8()?),
    };
    n.checker = CheckerParams {
        intensity_a: r.f32()?,
        intensity_b: r.f32()?,
        scale: r.f32()?,
    };
    n.tile = TileParams {
        x_amount: r.i32()?,
        y_amount: r.i32()?,
        gap: r.f32()?,
        size_rand_x: r.f32()?,
        size_rand_y: r.f32()?,
        offset: r.f32()?,
        roundness: r.f32()?,
        seed: r.f32()?,
    };
    n.bricks = BricksParams {
        x_amount: r.i32()?,
        y_amount: r.i32()?,
        gap: r.f32()?,
        offset: r.f32()?,
        roundness: r.f32()?,
        bevel: r.f32()?,
    };
    n.flood_fill = FloodFillParams {
        seed: r.f32()?,
        threshold: r.f32()?,
        luma_min: r.f32()?,
        luma_max: r.f32()?,
    };
    n.shape = ShapeParams {
        kind: shape_from(r.u8()?),
        size_x: r.f32()?,
        size_y: r.f32()?,
        sides: r.i32()?,
    };
    n.transform = TransformParams {
        offset_x: r.f32()?,
        offset_y: r.f32()?,
        scale_x: r.f32()?,
        scale_y: r.f32()?,
        rotation: r.f32()?,
        tileable: r.u8()? != 0,
    };
    n.tile_sampler = TileSamplerParams {
        x_amount: r.i32()?,
        y_amount: r.i32()?,
        offset_rand: r.f32()?,
        rotation_rand: r.f32()?,
        scale_rand: r.f32()?,
        seed: r.f32()?,
    };
    n.tess_factor = r.i32()?;
    n.displacement = r.f32()?;
    n.height_mode = height_from(r.u8()?);
    n.preview_dirty = kind.has_preview();
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::NodeKind;

    #[test]
    fn graph_roundtrip() {
        let mut g = TexGraph::new();
        let out = g.output_id.clone();
        let noise = g.add(NodeKind::Noise);
        let bricks = g.add(NodeKind::Bricks);
        if let Some(n) = g.node_mut(&noise) {
            n.noise.scale = 7.5;
            n.pos = Vec2::new(12.0, 34.0);
        }
        if let Some(n) = g.node_mut(&bricks) {
            n.bricks.x_amount = 6;
        }
        g.connect(&bricks, "out", &out, "height");
        g.connect(&noise, "out", &out, "roughness");
        let file = TexGraphFile {
            resolution: 512,
            graph: g,
        };
        let back = TexGraphFile::from_bytes(&file.to_bytes()).unwrap();
        assert_eq!(back.resolution, 512);
        assert_eq!(back.graph.nodes.len(), 3);
        assert_eq!(back.graph.links.len(), 2);
        let n = back
            .graph
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Noise)
            .unwrap();
        assert_eq!(n.noise.scale, 7.5);
        assert_eq!(n.pos, Vec2::new(12.0, 34.0));
        let b = back
            .graph
            .nodes
            .iter()
            .find(|n| n.kind == NodeKind::Bricks)
            .unwrap();
        assert_eq!(b.bricks.x_amount, 6);
    }
}

fn noise_u8(t: NoiseType) -> u8 {
    match t {
        NoiseType::Value => 0,
        NoiseType::Perlin => 1,
        NoiseType::Voronoi => 2,
        NoiseType::VoronoiEdge => 3,
        NoiseType::Gauss => 4,
        NoiseType::Cloud => 5,
        NoiseType::Anisotropic => 6,
    }
}

fn noise_from(id: u8) -> NoiseType {
    match id {
        1 => NoiseType::Perlin,
        2 => NoiseType::Voronoi,
        3 => NoiseType::VoronoiEdge,
        4 => NoiseType::Gauss,
        5 => NoiseType::Cloud,
        6 => NoiseType::Anisotropic,
        _ => NoiseType::Value,
    }
}

fn blend_u8(m: BlendMode) -> u8 {
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

fn blend_from(id: u8) -> BlendMode {
    match id {
        1 => BlendMode::Multiply,
        2 => BlendMode::Add,
        3 => BlendMode::Overlay,
        4 => BlendMode::Screen,
        5 => BlendMode::Divide,
        6 => BlendMode::Subtract,
        7 => BlendMode::Difference,
        8 => BlendMode::Darken,
        9 => BlendMode::Lighten,
        _ => BlendMode::Mix,
    }
}

fn grad_u8(m: GradientMode) -> u8 {
    match m {
        GradientMode::Linear => 0,
        GradientMode::Radial => 1,
    }
}

fn slope_u8(m: SlopeBlurMode) -> u8 {
    match m {
        SlopeBlurMode::Blur => 0,
        SlopeBlurMode::Min => 1,
        SlopeBlurMode::Max => 2,
    }
}

fn slope_from(id: u8) -> SlopeBlurMode {
    match id {
        1 => SlopeBlurMode::Min,
        2 => SlopeBlurMode::Max,
        _ => SlopeBlurMode::Blur,
    }
}

fn shape_u8(k: ShapeKind) -> u8 {
    match k {
        ShapeKind::Rectangle => 0,
        ShapeKind::Circle => 1,
        ShapeKind::Triangle => 2,
        ShapeKind::NGon => 3,
    }
}

fn shape_from(id: u8) -> ShapeKind {
    match id {
        1 => ShapeKind::Circle,
        2 => ShapeKind::Triangle,
        3 => ShapeKind::NGon,
        _ => ShapeKind::Rectangle,
    }
}

fn height_u8(m: HeightMode) -> u8 {
    match m {
        HeightMode::Tessellate => 0,
        HeightMode::Parallax => 1,
    }
}

fn height_from(id: u8) -> HeightMode {
    match id {
        1 => HeightMode::Parallax,
        _ => HeightMode::Tessellate,
    }
}

fn write_str(o: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    let n = u16::try_from(b.len()).unwrap_or(u16::MAX);
    let b = &b[..n as usize];
    write_u16(o, n);
    o.extend_from_slice(b);
}

fn write_u16(o: &mut Vec<u8>, v: u16) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn write_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn write_i32(o: &mut Vec<u8>, v: i32) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn write_f32s(o: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], TexGraphBytesError> {
        if self.remaining() < n {
            return Err(TexGraphBytesError::Truncated);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, TexGraphBytesError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, TexGraphBytesError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, TexGraphBytesError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, TexGraphBytesError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, TexGraphBytesError> {
        Ok(self.u32()? as i32)
    }

    fn f32(&mut self) -> Result<f32, TexGraphBytesError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32s<const N: usize>(&mut self) -> Result<[f32; N], TexGraphBytesError> {
        let mut a = [0.0; N];
        for x in &mut a {
            *x = self.f32()?;
        }
        Ok(a)
    }

    fn str(&mut self) -> Result<String, TexGraphBytesError> {
        let n = self.u16()? as usize;
        let raw = self.take(n)?;
        std::str::from_utf8(raw)
            .map(|s| s.to_string())
            .map_err(|_| TexGraphBytesError::BadUtf8)
    }
}
