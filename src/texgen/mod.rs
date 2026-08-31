//! Procedural texture graph → PBR maps / material.

mod bake;
mod gpu;
mod graph;
mod node;
mod ser;

pub use bake::insert_maps;
pub use gpu::GpuEval;
pub use graph::{
    ancestors_of, compute_out_fingerprints, find_link, flood_fill_gray, topo_order, TexGraph,
    TexLink,
};
pub use ser::{TexGraphBytesError, TexGraphFile};
pub use node::{
    port_tex, sample_gradient, BlendMode, BlurParams, BricksParams, CheckerParams, ColorRampParams,
    CurvatureParams, DistortParams, DirWarpParams, FloodFillParams, GradientMode, GradientStop,
    GraphNode, LevelsParams, LinesParams, NodeKind, NoiseParams, NoiseType, OpacityStop, ShapeKind,
    ShapeParams, SlopeBlurMode, SlopeBlurParams, TileParams, TileSamplerParams, TransformParams,
    WarpParams,
};
