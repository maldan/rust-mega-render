//! Node kinds and per-node parameters.

use glam::Vec2;

use crate::HeightMode;

/// Custom port type: grayscale / height / mask map.
pub mod port_tex {
    pub const GRAY: u16 = 100;
    pub const COLOR: u16 = 101;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Output,
    Color,
    Noise,
    Blend,
    Gradient,
    Levels,
    HeightToNormal,
    Curvature,
    GrayToColor,
    ColorToGray,
    ColorRamp,
    Lines,
    Distort,
    Checker,
    Tile,
    Bricks,
    FloodFill,
    Invert,
    Warp,
    DirectionalWarp,
    Blur,
    SlopeBlur,
    Shape,
    Transform,
    TileSampler,
}

impl NodeKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::Output => "Output",
            Self::Color => "Color",
            Self::Noise => "Noise",
            Self::Blend => "Blend",
            Self::Gradient => "Gradient",
            Self::Levels => "Levels",
            Self::HeightToNormal => "Height → Normal",
            Self::Curvature => "Curvature",
            Self::GrayToColor => "Gray → Color",
            Self::ColorToGray => "Color → Gray",
            Self::ColorRamp => "Color Ramp",
            Self::Lines => "Lines",
            Self::Distort => "Distort",
            Self::Checker => "Checker",
            Self::Tile => "Tile Generator",
            Self::Bricks => "Bricks",
            Self::FloodFill => "Flood Fill",
            Self::Invert => "Invert",
            Self::Warp => "Warp",
            Self::DirectionalWarp => "Directional Warp",
            Self::Blur => "Blur",
            Self::SlopeBlur => "Slope Blur",
            Self::Shape => "Shape",
            Self::Transform => "Transform",
            Self::TileSampler => "Tile Sampler",
        }
    }

    pub fn can_delete(self) -> bool {
        !matches!(self, Self::Output)
    }

    pub fn has_preview(self) -> bool {
        !matches!(self, Self::Output)
    }

    pub fn has_inspector(self) -> bool {
        !matches!(
            self,
            Self::GrayToColor | Self::ColorToGray | Self::Invert
        )
    }

    pub(crate) fn to_u8(self) -> u8 {
        match self {
            Self::Output => 0,
            Self::Color => 1,
            Self::Noise => 2,
            Self::Blend => 3,
            Self::Gradient => 4,
            Self::Levels => 5,
            Self::HeightToNormal => 6,
            Self::Curvature => 7,
            Self::GrayToColor => 8,
            Self::ColorToGray => 9,
            Self::ColorRamp => 10,
            Self::Lines => 11,
            Self::Distort => 12,
            Self::Checker => 13,
            Self::Tile => 14,
            Self::Bricks => 15,
            Self::FloodFill => 16,
            Self::Invert => 17,
            Self::Warp => 18,
            Self::DirectionalWarp => 19,
            Self::Blur => 20,
            Self::SlopeBlur => 21,
            Self::Shape => 22,
            Self::Transform => 23,
            Self::TileSampler => 24,
        }
    }

    pub(crate) fn from_u8(id: u8) -> Option<Self> {
        Some(match id {
            0 => Self::Output,
            1 => Self::Color,
            2 => Self::Noise,
            3 => Self::Blend,
            4 => Self::Gradient,
            5 => Self::Levels,
            6 => Self::HeightToNormal,
            7 => Self::Curvature,
            8 => Self::GrayToColor,
            9 => Self::ColorToGray,
            10 => Self::ColorRamp,
            11 => Self::Lines,
            12 => Self::Distort,
            13 => Self::Checker,
            14 => Self::Tile,
            15 => Self::Bricks,
            16 => Self::FloodFill,
            17 => Self::Invert,
            18 => Self::Warp,
            19 => Self::DirectionalWarp,
            20 => Self::Blur,
            21 => Self::SlopeBlur,
            22 => Self::Shape,
            23 => Self::Transform,
            24 => Self::TileSampler,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Mix,
    Multiply,
    Add,
    Overlay,
    Screen,
    Divide,
    Subtract,
    Difference,
    Darken,
    Lighten,
}

impl BlendMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Mix => "Mix",
            Self::Multiply => "Multiply",
            Self::Add => "Add",
            Self::Overlay => "Overlay",
            Self::Screen => "Screen",
            Self::Divide => "Divide",
            Self::Subtract => "Subtract",
            Self::Difference => "Difference",
            Self::Darken => "Darken",
            Self::Lighten => "Lighten",
        }
    }

    pub const ALL: [BlendMode; 10] = [
        Self::Mix,
        Self::Multiply,
        Self::Add,
        Self::Overlay,
        Self::Screen,
        Self::Divide,
        Self::Subtract,
        Self::Difference,
        Self::Darken,
        Self::Lighten,
    ];
    pub const LABELS: [&'static str; 10] = [
        "Mix",
        "Multiply",
        "Add",
        "Overlay",
        "Screen",
        "Divide",
        "Subtract",
        "Difference",
        "Darken",
        "Lighten",
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientMode {
    Linear,
    Radial,
}

impl GradientMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Radial => "Radial",
        }
    }

    pub const ALL: [GradientMode; 2] = [Self::Linear, Self::Radial];
    pub const LABELS: [&'static str; 2] = ["Linear", "Radial"];
}

/// Color stop (RGB; alpha in `color[3]` is ignored — use [`OpacityStop`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    pub t: f32,
    pub color: [f32; 4],
}

/// Opacity stop (alpha only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpacityStop {
    pub t: f32,
    pub alpha: f32,
}

/// Sample RGB from color stops and alpha from opacity stops.
pub fn sample_gradient(colors: &[GradientStop], opacities: &[OpacityStop], t: f32) -> [f32; 4] {
    let rgb = sample_rgb(colors, t);
    let a = sample_alpha(opacities, t);
    [rgb[0], rgb[1], rgb[2], a]
}

fn sample_rgb(stops: &[GradientStop], t: f32) -> [f32; 3] {
    if stops.is_empty() {
        return [1.0, 1.0, 1.0];
    }
    if stops.len() == 1 {
        let c = stops[0].color;
        return [c[0], c[1], c[2]];
    }
    let t = t.clamp(0.0, 1.0);
    let mut ordered: Vec<&GradientStop> = stops.iter().collect();
    ordered.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    if t <= ordered[0].t {
        let c = ordered[0].color;
        return [c[0], c[1], c[2]];
    }
    let last = *ordered.last().unwrap();
    if t >= last.t {
        return [last.color[0], last.color[1], last.color[2]];
    }
    for w in ordered.windows(2) {
        let a = w[0];
        let b = w[1];
        if t >= a.t && t <= b.t {
            let span = (b.t - a.t).max(1e-5);
            let u = (t - a.t) / span;
            return [
                a.color[0] + (b.color[0] - a.color[0]) * u,
                a.color[1] + (b.color[1] - a.color[1]) * u,
                a.color[2] + (b.color[2] - a.color[2]) * u,
            ];
        }
    }
    [last.color[0], last.color[1], last.color[2]]
}

fn sample_alpha(stops: &[OpacityStop], t: f32) -> f32 {
    if stops.is_empty() {
        return 1.0;
    }
    if stops.len() == 1 {
        return stops[0].alpha.clamp(0.0, 1.0);
    }
    let t = t.clamp(0.0, 1.0);
    let mut ordered: Vec<&OpacityStop> = stops.iter().collect();
    ordered.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    if t <= ordered[0].t {
        return ordered[0].alpha.clamp(0.0, 1.0);
    }
    let last = *ordered.last().unwrap();
    if t >= last.t {
        return last.alpha.clamp(0.0, 1.0);
    }
    for w in ordered.windows(2) {
        let a = w[0];
        let b = w[1];
        if t >= a.t && t <= b.t {
            let span = (b.t - a.t).max(1e-5);
            let u = (t - a.t) / span;
            return (a.alpha + (b.alpha - a.alpha) * u).clamp(0.0, 1.0);
        }
    }
    last.alpha.clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseType {
    Value,
    Perlin,
    Voronoi,
    VoronoiEdge,
    Gauss,
    Cloud,
    Anisotropic,
}

impl NoiseType {
    pub const ALL: [NoiseType; 7] = [
        Self::Value,
        Self::Perlin,
        Self::Voronoi,
        Self::VoronoiEdge,
        Self::Gauss,
        Self::Cloud,
        Self::Anisotropic,
    ];
    pub const LABELS: [&'static str; 7] = [
        "Value",
        "Perlin",
        "Voronoi",
        "Voronoi Edge",
        "Gauss",
        "Cloud",
        "Anisotropic",
    ];
}

#[derive(Clone, Debug)]
pub struct NoiseParams {
    pub kind: NoiseType,
    pub scale: f32,
    pub octaves: i32,
    pub seed: f32,
    pub tileable: bool,
    pub angle: f32,
    pub stretch: f32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            kind: NoiseType::Value,
            scale: 4.0,
            octaves: 4,
            seed: 1.0,
            tileable: true,
            angle: 0.0,
            stretch: 16.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LevelsParams {
    pub in_black: f32,
    pub in_white: f32,
    pub gamma: f32,
    pub out_black: f32,
    pub out_white: f32,
}

impl Default for LevelsParams {
    fn default() -> Self {
        Self {
            in_black: 0.0,
            in_white: 1.0,
            gamma: 1.0,
            out_black: 0.0,
            out_white: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColorRampParams {
    pub colors: Vec<GradientStop>,
    pub opacities: Vec<OpacityStop>,
}

impl Default for ColorRampParams {
    fn default() -> Self {
        Self {
            colors: vec![
                GradientStop {
                    t: 0.0,
                    color: [0.0, 0.0, 0.0, 1.0],
                },
                GradientStop {
                    t: 1.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            ],
            opacities: vec![
                OpacityStop { t: 0.0, alpha: 1.0 },
                OpacityStop { t: 1.0, alpha: 1.0 },
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinesParams {
    /// Line thickness as fraction of UV (0..1).
    pub width: f32,
    pub count: i32,
    /// Degrees.
    pub rotation: f32,
    /// Line value (1 = white, 0 = black).
    pub intensity: f32,
    /// Background value (0 = black, 1 = white).
    pub bg_intensity: f32,
}

impl Default for LinesParams {
    fn default() -> Self {
        Self {
            width: 0.05,
            count: 1,
            rotation: 0.0,
            intensity: 1.0,
            bg_intensity: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DistortParams {
    /// UV warp amount (0..1 typical).
    pub strength: f32,
    /// Warp-noise frequency.
    pub scale: f32,
    pub seed: f32,
}

impl Default for DistortParams {
    fn default() -> Self {
        Self {
            strength: 0.15,
            scale: 3.0,
            seed: 1.0,
        }
    }
}

/// Height-field curvature (discrete Laplacian). 0.5 = flat, >0.5 convex.
#[derive(Clone, Debug)]
pub struct CurvatureParams {
    pub intensity: f32,
    /// Neighbor offset in pixels.
    pub radius: i32,
}

impl Default for CurvatureParams {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            radius: 1,
        }
    }
}

/// Substance-style Warp: displace by gradient of a drive (intensity) map.
#[derive(Clone, Debug)]
pub struct WarpParams {
    pub strength: f32,
}

impl Default for WarpParams {
    fn default() -> Self {
        Self { strength: 0.02 }
    }
}

/// Substance Directional Warp: shift UV along a fixed angle, scaled by a drive map.
#[derive(Clone, Debug)]
pub struct DirWarpParams {
    pub intensity: f32,
    /// Degrees. 0 = +U.
    pub angle: f32,
}

impl Default for DirWarpParams {
    fn default() -> Self {
        Self {
            intensity: 0.1,
            angle: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlurParams {
    /// Blur radius in pixels. Drive map (0..1) scales this per texel.
    pub radius: f32,
}

impl Default for BlurParams {
    fn default() -> Self {
        Self { radius: 5.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlopeBlurMode {
    Blur,
    Min,
    Max,
}

impl SlopeBlurMode {
    pub const ALL: [SlopeBlurMode; 3] = [Self::Blur, Self::Min, Self::Max];
    pub const LABELS: [&'static str; 3] = ["Blur", "Min", "Max"];
}

/// Substance Slope Blur: sample `in` along the gradient of a slope map.
#[derive(Clone, Debug)]
pub struct SlopeBlurParams {
    pub intensity: f32,
    pub samples: i32,
    pub mode: SlopeBlurMode,
}

impl Default for SlopeBlurParams {
    fn default() -> Self {
        Self {
            intensity: 0.05,
            samples: 8,
            mode: SlopeBlurMode::Blur,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CheckerParams {
    pub intensity_a: f32,
    pub intensity_b: f32,
    /// Squares along each axis.
    pub scale: f32,
}

impl Default for CheckerParams {
    fn default() -> Self {
        Self {
            intensity_a: 0.0,
            intensity_b: 1.0,
            scale: 4.0,
        }
    }
}

/// Substance-style Tile Random: irregular masonry mask (seamless).
#[derive(Clone, Debug)]
pub struct TileParams {
    pub x_amount: i32,
    pub y_amount: i32,
    /// Mortar width as fraction of the smaller average cell (same in U and V).
    pub gap: f32,
    /// How uneven column widths are (0 = regular, 1 = wild).
    pub size_rand_x: f32,
    /// How uneven row heights are (0 = regular, 1 = wild).
    pub size_rand_y: f32,
    /// Horizontal row stagger (brick bond), 0..1.
    pub offset: f32,
    /// Corner roundness 0..1.
    pub roundness: f32,
    pub seed: f32,
}

impl Default for TileParams {
    fn default() -> Self {
        Self {
            x_amount: 4,
            y_amount: 4,
            gap: 0.12,
            size_rand_x: 0.65,
            size_rand_y: 0.65,
            offset: 0.35,
            roundness: 0.35,
            seed: 1.0,
        }
    }
}

/// Regular running-bond brick mask / height (seamless).
#[derive(Clone, Debug)]
pub struct BricksParams {
    pub x_amount: i32,
    pub y_amount: i32,
    /// Mortar width as fraction of cell (0..0.4).
    pub gap: f32,
    /// Horizontal row stagger (0.5 = running bond).
    pub offset: f32,
    /// Corner roundness 0..1.
    pub roundness: f32,
    /// Edge falloff 0..1 (0 = hard mask, >0 = height bevel).
    pub bevel: f32,
}

impl Default for BricksParams {
    fn default() -> Self {
        Self {
            x_amount: 4,
            y_amount: 8,
            gap: 0.08,
            offset: 0.5,
            roundness: 0.12,
            bevel: 0.12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    Rectangle,
    Circle,
    Triangle,
    NGon,
}

impl ShapeKind {
    pub const ALL: [ShapeKind; 4] = [
        Self::Rectangle,
        Self::Circle,
        Self::Triangle,
        Self::NGon,
    ];
    pub const LABELS: [&'static str; 4] = ["Rectangle", "Circle", "Triangle", "N-Gon"];
}

#[derive(Clone, Debug)]
pub struct ShapeParams {
    pub kind: ShapeKind,
    pub size_x: f32,
    pub size_y: f32,
    pub sides: i32,
}

impl Default for ShapeParams {
    fn default() -> Self {
        Self {
            kind: ShapeKind::Rectangle,
            size_x: 0.5,
            size_y: 0.5,
            sides: 6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransformParams {
    pub offset_x: f32,
    pub offset_y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotation: f32,
    pub tileable: bool,
}

impl Default for TransformParams {
    fn default() -> Self {
        Self {
            offset_x: 0.0,
            offset_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            rotation: 0.0,
            tileable: false,
        }
    }
}

/// Stamp an input pattern into an M×N grid with per-cell random transform.
#[derive(Clone, Debug)]
pub struct TileSamplerParams {
    pub x_amount: i32,
    pub y_amount: i32,
    pub offset_rand: f32,
    pub rotation_rand: f32,
    pub scale_rand: f32,
    pub seed: f32,
}

impl Default for TileSamplerParams {
    fn default() -> Self {
        Self {
            x_amount: 4,
            y_amount: 4,
            offset_rand: 0.0,
            rotation_rand: 0.0,
            scale_rand: 0.0,
            seed: 1.0,
        }
    }
}

/// Substance-style flood fill → random grayscale per island.
#[derive(Clone, Debug)]
pub struct FloodFillParams {
    pub seed: f32,
    /// Pixels above this are shapes; below is mortar/barrier.
    pub threshold: f32,
    /// Random gray range per island (0..1). Can be inverted.
    pub luma_min: f32,
    pub luma_max: f32,
}

impl Default for FloodFillParams {
    fn default() -> Self {
        Self {
            seed: 1.0,
            threshold: 0.5,
            luma_min: 0.35,
            luma_max: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: String,
    pub kind: NodeKind,
    pub pos: Vec2,
    /// Uniform color (Color node) — linear 0..1 RGBA.
    pub color: [f32; 4],
    pub noise: NoiseParams,
    pub blend_mode: BlendMode,
    /// Blend opacity / mix factor.
    pub mix: f32,
    /// Used when Blend `a` is unconnected.
    pub blend_a: [f32; 4],
    /// Used when Blend `b` is unconnected.
    pub blend_b: [f32; 4],
    pub gradient_mode: GradientMode,
    pub levels: LevelsParams,
    /// Height→Normal strength.
    pub normal_strength: f32,
    pub curvature: CurvatureParams,
    pub color_ramp: ColorRampParams,
    pub lines: LinesParams,
    pub distort: DistortParams,
    pub warp: WarpParams,
    pub dir_warp: DirWarpParams,
    pub blur: BlurParams,
    pub slope_blur: SlopeBlurParams,
    pub checker: CheckerParams,
    pub tile: TileParams,
    pub bricks: BricksParams,
    pub flood_fill: FloodFillParams,
    pub shape: ShapeParams,
    pub transform: TransformParams,
    pub tile_sampler: TileSamplerParams,
    /// Output: tessellation cap (1..=32), or parallax ray-march steps.
    pub tess_factor: i32,
    /// Output: world displacement at height=1.
    pub displacement: f32,
    /// Output: tessellate vertices vs parallax UV offset.
    pub height_mode: HeightMode,
    /// UI texture slot for in-node preview (`None` until assigned).
    pub preview_slot: Option<u32>,
    pub preview_rgba: Vec<u8>,
    pub preview_res: u32,
    pub preview_version: u64,
    pub preview_uploaded: u64,
    pub preview_dirty: bool,
}

impl GraphNode {
    pub fn new(id: String, kind: NodeKind, pos: Vec2) -> Self {
        Self {
            id,
            kind,
            pos,
            color: [0.75, 0.75, 0.8, 1.0],
            noise: NoiseParams::default(),
            blend_mode: BlendMode::Mix,
            mix: 0.5,
            blend_a: [0.0, 0.0, 0.0, 1.0],
            blend_b: [1.0, 1.0, 1.0, 1.0],
            gradient_mode: GradientMode::Linear,
            levels: LevelsParams::default(),
            normal_strength: 1.0,
            curvature: CurvatureParams::default(),
            color_ramp: ColorRampParams::default(),
            lines: LinesParams::default(),
            distort: DistortParams::default(),
            warp: WarpParams::default(),
            dir_warp: DirWarpParams::default(),
            blur: BlurParams::default(),
            slope_blur: SlopeBlurParams::default(),
            checker: CheckerParams::default(),
            tile: TileParams::default(),
            bricks: BricksParams::default(),
            flood_fill: FloodFillParams::default(),
            shape: ShapeParams::default(),
            transform: TransformParams::default(),
            tile_sampler: TileSamplerParams::default(),
            tess_factor: 32,
            displacement: 0.12,
            height_mode: HeightMode::Tessellate,
            preview_slot: None,
            preview_rgba: Vec::new(),
            preview_res: 0,
            preview_version: 0,
            preview_uploaded: 0,
            preview_dirty: kind.has_preview(),
        }
    }
}
