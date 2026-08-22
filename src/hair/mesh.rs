//! Strand interpolation and ribbon meshing.

use crate::mesh::Mesh;
use glam::{Vec2, Vec3};

use super::curve::{sample_curve, HairCurve};
use super::params::{HairGuide, HairParams, HairShape, HairStyle, LayerRandom, RandRange};
use super::MAX_HAIR_BONES;

/// Per-strand brightness variation range (vertex `colors[0].y`).
const TINT_MIN: f32 = 0.72;
const TINT_MAX: f32 = 1.28;

#[derive(Clone, Copy, Default)]
struct SkinBind {
    bone_base: u16,
    n_bones: u16,
    weight: f32,
}

#[derive(Clone)]
pub struct Strand {
    pub pts: Vec<Vec3>,
    pub nrms: Vec<Vec3>,
    pub widths: Vec<f32>,
    /// Multiplier on albedo RGB (vertex `colors[0].y`).
    pub shade: f32,
    /// Per-layer opacity (vertex `colors[0].z`).
    pub alpha: f32,
    binds: [SkinBind; 2],
}

pub fn generate_hair_mesh(
    guides: &[HairGuide],
    fills: &[(usize, usize)],
    params: &HairParams,
    auto_idx: u32,
) -> (HairMeshBuffers, Option<HairMeshBuffers>) {
    let strands = build_strands(guides, fills, params, auto_idx);
    mesh_from_strands(&strands, params)
}

pub fn build_strands(
    guides: &[HairGuide],
    fills: &[(usize, usize)],
    params: &HairParams,
    auto_idx: u32,
) -> Vec<Strand> {
    let expanded = expand_guides(guides);
    let alpha = params.layer_alpha.first().copied().unwrap_or(1.0);
    let dist = auto_idx as f32 * params.layer_gap.max(0.0);

    // Sample each guide, then push this whole auto-stack along the root
    // normal (away from the scalp). Per-point normals would fold the strand;
    // a rigid root-normal offset keeps stacks clearly separated for depth.
    let layout = skin_layout(&expanded);

    let mut bases: Vec<Option<Strand>> = vec![None; expanded.len()];
    for (i, e) in expanded.iter().enumerate() {
        if e.guide.points.len() < 2 {
            continue;
        }
        let mut s = sample_guide(&e.guide, params);
        s = offset_along_root_normal(&s, dist);
        s.alpha = alpha;
        s.binds = [skin_bind(layout[i], 1.0), SkinBind::default()];
        bases[i] = Some(s);
    }

    let mut strands = Vec::new();
    let mut ids = Vec::new();

    // Guide strands
    for (i, e) in expanded.iter().enumerate() {
        if let Some(s) = bases[i].as_ref() {
            strands.push(s.clone());
            ids.push(strand_key(
                e.src as u32,
                e.mirrored as u32,
                auto_idx,
                0,
                0,
            ));
        }
    }

    // Density fills between guide pairs (same auto-stack)
    if params.density > 0 && expanded.len() >= 2 {
        let extra = params.density;
        for (pair, (i, j)) in fill_pairs(&expanded, fills).into_iter().enumerate() {
            let Some(ga) = bases[i].as_ref() else {
                continue;
            };
            let Some(gb) = bases[j].as_ref() else {
                continue;
            };
            for k in 1..=extra {
                let t = k as f32 / (extra + 1) as f32;
                let mut s = lerp_strand(ga, gb, t, params.fill_curve);
                s.alpha = alpha;
                s.binds = [
                    skin_bind(layout[i], 1.0 - t),
                    skin_bind(layout[j], t),
                ];
                strands.push(s);
                ids.push(strand_key(
                    expanded[i].src as u32,
                    expanded[i].mirrored as u32,
                    auto_idx,
                    k,
                    pair as u32 + 1,
                ));
            }
        }
    }

    // Multiply: extra copies of every strand on this auto-stack, each with a
    // unique seed so the shared randomizer spreads them.
    let multiply = params.multiply.max(1);
    if multiply > 1 {
        let n = strands.len();
        for m in 1..multiply {
            for i in 0..n {
                strands.push(strands[i].clone());
                ids.push(ids[i].wrapping_add(m.wrapping_mul(0x85EBCA77)));
            }
        }
    }

    let lr = params.layer_rand.first().copied().unwrap_or_default();
    for (s, id) in strands.iter_mut().zip(ids.iter()) {
        vary_strand(s, *id, params, &lr);
        refit_frames(s);
    }

    strands
}

fn offset_along_root_normal(s: &Strand, dist: f32) -> Strand {
    if dist.abs() < 1e-8 {
        return s.clone();
    }
    let dir = s
        .nrms
        .first()
        .copied()
        .unwrap_or(Vec3::Y)
        .normalize_or_zero();
    let mut o = s.clone();
    for p in &mut o.pts {
        *p += dir * dist;
    }
    o
}

fn strand_key(src: u32, mirrored: u32, copy: u32, fill_k: u32, pair: u32) -> u32 {
    src.wrapping_mul(0x9E3779B9)
        .wrapping_add(mirrored.wrapping_mul(0x85EBCA77))
        .wrapping_add(copy.wrapping_mul(0xC2B2AE3D))
        .wrapping_add(fill_k.wrapping_mul(0x27D4EB2F))
        .wrapping_add(pair.wrapping_mul(0x165667B1))
}

struct Rng(u32);

impl Rng {
    fn new(seed: u32, id: u32) -> Self {
        let mut x = seed.wrapping_add(1).wrapping_mul(0xA24BAED5).wrapping_add(id);
        x ^= x >> 16;
        x = x.wrapping_mul(0x7FEB352D);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846CA68B);
        x ^= x >> 16;
        Self(x | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn f01(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    fn signed(&mut self) -> f32 {
        self.f01() * 2.0 - 1.0
    }

    /// Uniform sample in `[r.min, r.max]` (works even if `min > max`).
    fn range(&mut self, r: RandRange) -> f32 {
        r.min + self.f01() * (r.max - r.min)
    }
}

fn vary_strand(s: &mut Strand, id: u32, params: &HairParams, lr: &LayerRandom) {
    let r_len = lr.length.is_active();
    let r_wid = lr.width.is_active();
    let r_roll = lr.roll.is_active();
    let r_off = lr.offset.length_squared() > 1e-10;
    let r_rot = lr.rotate.length_squared() > 1e-10;
    let mut rng = Rng::new(params.seed, id);

    // Rigid whole-strand tilt: independent random rotation around the root's
    // side/normal/tangent axes, giving strands slightly different aim
    // directions instead of all growing perfectly parallel (like a real
    // scalp, where no two hairs point in exactly the same direction).
    if r_rot && s.pts.len() >= 2 {
        let tan0 = strand_tangents(&s.pts)[0];
        let nrm0 = s.nrms[0];
        let side0 = tan0.cross(nrm0).normalize_or_zero();
        let ax = rng.signed() * lr.rotate.x;
        let ay = rng.signed() * lr.rotate.y;
        let az = rng.signed() * lr.rotate.z;
        let root = s.pts[0];
        for i in 0..s.pts.len() {
            let mut v = s.pts[i] - root;
            v = rotate_around(v, side0, ax);
            v = rotate_around(v, nrm0, ay);
            v = rotate_around(v, tan0, az);
            s.pts[i] = root + v;
            let mut nv = s.nrms[i];
            nv = rotate_around(nv, side0, ax);
            nv = rotate_around(nv, nrm0, ay);
            nv = rotate_around(nv, tan0, az);
            s.nrms[i] = nv.normalize_or_zero();
        }
    }

    let curl_amt = if r_roll {
        (params.curl * (1.0 + rng.range(lr.roll) * 0.55)).max(0.0)
    } else {
        params.curl
    };
    let curl_start = if r_roll {
        (params.curl_start + rng.range(lr.roll) * 0.1).clamp(0.0, 0.95)
    } else {
        params.curl_start
    };
    let curl_radius = if r_roll {
        (params.curl_radius * (1.0 + rng.range(lr.roll) * 0.35)).max(0.0)
    } else {
        params.curl_radius
    };
    let roll_amt = if r_roll {
        // No `.max(0.0)`: roll may be negative (reverse winding), and jitter
        // should scale the signed amount rather than clamp it toward zero.
        params.roll * (1.0 + rng.range(lr.roll) * 0.55)
    } else {
        params.roll
    };
    let roll_start = if r_roll {
        (params.roll_start + rng.range(lr.roll) * 0.1).clamp(0.0, 0.95)
    } else {
        params.roll_start
    };
    let wave_amp = if r_roll {
        (params.wave_amp * (1.0 + rng.range(lr.roll) * 0.4)).max(0.0)
    } else {
        params.wave_amp
    };
    let wave_freq = if r_roll {
        (params.wave_freq * (1.0 + rng.range(lr.roll) * 0.2)).max(0.0)
    } else {
        params.wave_freq
    };
    let wave_start = if r_roll {
        (params.wave_start + rng.range(lr.roll) * 0.1).clamp(0.0, 0.95)
    } else {
        params.wave_start
    };
    let crimp_amp = if r_roll {
        (params.crimp_amp * (1.0 + rng.range(lr.roll) * 0.4)).max(0.0)
    } else {
        params.crimp_amp
    };
    let crimp_freq = if r_roll {
        (params.crimp_freq * (1.0 + rng.range(lr.roll) * 0.2)).max(0.0)
    } else {
        params.crimp_freq
    };
    let crimp_start = if r_roll {
        (params.crimp_start + rng.range(lr.roll) * 0.08).clamp(0.0, 0.95)
    } else {
        params.crimp_start
    };
    let coil_turns = if r_roll {
        (params.coil_turns * (1.0 + rng.range(lr.roll) * 0.35)).max(0.0)
    } else {
        params.coil_turns
    };
    let coil_start = if r_roll {
        (params.coil_start + rng.range(lr.roll) * 0.1).clamp(0.0, 0.95)
    } else {
        params.coil_start
    };
    let coil_radius = if r_roll {
        (params.coil_radius * (1.0 + rng.range(lr.roll) * 0.3)).max(0.0)
    } else {
        params.coil_radius
    };

    let any_pos_rand = r_len || r_off || r_wid;
    if any_pos_rand && s.pts.len() >= 2 {
        let len_s = if r_len {
            (1.0 + rng.range(lr.length) * 0.32).max(0.35)
        } else {
            1.0
        };
        let w_s = if r_wid {
            (1.0 + rng.range(lr.width) * 0.28).max(0.25)
        } else {
            1.0
        };
        s.shade = if r_off {
            (1.0 + rng.signed() * lr.offset.length() * 0.22).clamp(TINT_MIN, TINT_MAX)
        } else {
            1.0
        };
        let root = s.pts[0];
        let n = s.pts.len();
        // Root-only offset: the whole strand is rigidly translated by a single
        // random vector (derived at the root), so the shape/curvature isn't
        // distorted segment-by-segment. Each axis (side/normal/along-strand)
        // has its own independent magnitude via `lr.offset`.
        let root_shift = if r_off {
            let tan0 = strand_tangents(&s.pts)[0];
            let nrm0 = s.nrms[0];
            let side0 = tan0.cross(nrm0).normalize_or_zero();
            let base_amp = 0.006 + s.widths[0] * 1.1;
            let ax = rng.signed() * lr.offset.x * base_amp;
            let ay = rng.signed() * lr.offset.y * base_amp;
            let az = rng.signed() * lr.offset.z * base_amp * 0.5;
            nrm0 * ay + side0 * ax + tan0 * az
        } else {
            Vec3::ZERO
        };
        for i in 0..n {
            if r_len {
                s.pts[i] = root + (s.pts[i] - root) * len_s;
            }
            if r_wid {
                if let Some(w) = s.widths.get_mut(i) {
                    *w = (*w * w_s).max(0.0005);
                }
            }
            if r_off {
                s.pts[i] += root_shift;
            }
        }
    }
    match params.style {
        HairStyle::Straight => {}
        HairStyle::Roll => apply_roll(s, roll_amt, roll_start),
        HairStyle::Curl => apply_curl(s, curl_amt, curl_start, curl_radius),
        HairStyle::Wave => apply_wave(s, wave_amp, wave_freq, wave_start),
        HairStyle::Crimp => apply_crimp(s, crimp_amp, crimp_freq, crimp_start),
        HairStyle::Coil => {
            apply_coil(s, coil_turns, coil_start, coil_radius, params.coil_taper)
        }
    }
}

pub(crate) struct Expanded {
    pub(crate) guide: HairGuide,
    pub(crate) src: usize,
    pub(crate) mirrored: bool,
}

/// Per expanded guide: (bone_base, n_bones) in this layer's joint space.
pub(crate) fn skin_layout(expanded: &[Expanded]) -> Vec<Option<(u16, u16)>> {
    let mut bone_base = 0u16;
    let mut out = Vec::with_capacity(expanded.len());
    for e in expanded {
        if e.guide.is_static || e.guide.points.len() < 2 {
            out.push(None);
            continue;
        }
        let n_bones = (e.guide.points.len() - 1) as u16;
        if bone_base >= MAX_HAIR_BONES || bone_base.saturating_add(n_bones) > MAX_HAIR_BONES {
            out.push(None);
            continue;
        }
        out.push(Some((bone_base, n_bones)));
        bone_base = bone_base.saturating_add(n_bones);
    }
    out
}

fn skin_bind(layout: Option<(u16, u16)>, weight: f32) -> SkinBind {
    let Some((bone_base, n_bones)) = layout else {
        return SkinBind::default();
    };
    if n_bones == 0 || weight.abs() < 1e-8 {
        return SkinBind::default();
    }
    SkinBind {
        bone_base,
        n_bones,
        weight,
    }
}

fn add_influence(joints: &mut [u16; 4], weights: &mut [f32; 4], n: &mut usize, j: u16, w: f32) {
    if w <= 1e-8 {
        return;
    }
    for i in 0..*n {
        if joints[i] == j {
            weights[i] += w;
            return;
        }
    }
    if *n < 4 {
        joints[*n] = j;
        weights[*n] = w;
        *n += 1;
        return;
    }
    let mut m = 0;
    for i in 1..4 {
        if weights[i] < weights[m] {
            m = i;
        }
    }
    if w > weights[m] {
        joints[m] = j;
        weights[m] = w;
    }
}

/// `t` 0 at root, 1 at tip. Dual-bind neighboring bones of each parent chain.
fn skin_vertex(t: f32, binds: &[SkinBind; 2]) -> ([u16; 4], [f32; 4]) {
    let mut joints = [0u16; 4];
    let mut weights = [0.0f32; 4];
    let mut n = 0usize;
    let t = t.clamp(0.0, 1.0);
    for b in binds {
        if b.n_bones == 0 || b.weight.abs() < 1e-8 {
            continue;
        }
        let f = t * b.n_bones as f32;
        let i = f.floor() as u16;
        let frac = f - i as f32;
        if i >= b.n_bones {
            add_influence(
                &mut joints,
                &mut weights,
                &mut n,
                b.bone_base.saturating_add(b.n_bones - 1),
                b.weight,
            );
        } else if frac <= 1e-5 || i + 1 >= b.n_bones {
            add_influence(
                &mut joints,
                &mut weights,
                &mut n,
                b.bone_base.saturating_add(i.min(b.n_bones - 1)),
                b.weight,
            );
        } else {
            add_influence(
                &mut joints,
                &mut weights,
                &mut n,
                b.bone_base + i,
                b.weight * (1.0 - frac),
            );
            add_influence(
                &mut joints,
                &mut weights,
                &mut n,
                b.bone_base + i + 1,
                b.weight * frac,
            );
        }
    }
    let sum: f32 = weights.iter().sum();
    if sum > 1e-8 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    (joints, weights)
}

pub(crate) fn expand_guides(guides: &[HairGuide]) -> Vec<Expanded> {
    let mut out = Vec::with_capacity(guides.len() * 2);
    for (i, g) in guides.iter().enumerate() {
        let mut base = g.clone();
        base.mirror_x = false;
        out.push(Expanded {
            guide: base,
            src: i,
            mirrored: false,
        });
        if g.mirror_x && g.points.iter().any(|p| p.pos.x.abs() > 1e-3) {
            out.push(Expanded {
                guide: g.mirrored_x(),
                src: i,
                mirrored: true,
            });
        }
    }
    out
}

fn fill_pairs(expanded: &[Expanded], fills: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let find = |src: usize, mirrored: bool| {
        expanded
            .iter()
            .position(|e| e.src == src && e.mirrored == mirrored)
    };
    let mut pairs = Vec::new();
    for &(a, b) in fills {
        if let (Some(i), Some(j)) = (find(a, false), find(b, false)) {
            pairs.push((i.min(j), i.max(j)));
        }
        if let (Some(i), Some(j)) = (find(a, true), find(b, true)) {
            pairs.push((i.min(j), i.max(j)));
        }
    }
    pairs
}

fn sample_guide(g: &HairGuide, params: &HairParams) -> Strand {
    let segs = params.segments.max(2) as usize;
    let (start, tip_dens) = match params.style {
        HairStyle::Straight => (1.0, 1.0),
        // Root keeps normal segment spacing; tip adds dens × more verts.
        HairStyle::Roll => (
            params.roll_start.clamp(0.0, 0.95),
            params.tip_density.max(1.0),
        ),
        HairStyle::Curl => (
            params.curl_start.clamp(0.0, 0.95),
            params.tip_density.max(1.0),
        ),
        HairStyle::Wave => (
            params.wave_start.clamp(0.0, 0.95),
            params.tip_density.max(1.0),
        ),
        HairStyle::Crimp => (
            params.crimp_start.clamp(0.0, 0.95),
            params.tip_density.max(1.0),
        ),
        HairStyle::Coil => (
            params.coil_start.clamp(0.0, 0.95),
            params.tip_density.max(1.0),
        ),
    };
    let ts = adaptive_ts(segs, start, tip_dens);
    let mut pts = Vec::with_capacity(ts.len());
    let mut nrms = Vec::with_capacity(ts.len());
    let mut widths = Vec::with_capacity(ts.len());
    if g.points.is_empty() {
        return Strand {
            pts,
            nrms,
            widths,
            shade: 1.0,
            alpha: 1.0,
            binds: Default::default(),
        };
    }
    let smooth = params.smooth.clamp(0.0, 1.0);
    for &t in &ts {
        let (p, n) = sample_path(g, t, smooth);
        let lift =
            g.lift * sample_curve(&params.lift_curve, t).clamp(0.0, 2.0) * params.lift_mult;
        let width =
            g.width * sample_curve(&params.width_curve, t).clamp(0.0, 2.0) * params.width_mult;
        pts.push(p + n * lift);
        nrms.push(n);
        widths.push(width.max(0.0005));
    }
    Strand {
        pts,
        nrms,
        widths,
        shade: 1.0,
        alpha: 1.0,
        binds: Default::default(),
    }
}

/// Sample `t` along the strand. Root `[0, start]` keeps the normal
/// `segments` spacing; tip `[start, 1]` is `tip_density`× denser (extra
/// verts added, root is not robbed).
fn adaptive_ts(segs: usize, start: f32, tip_density: f32) -> Vec<f32> {
    let n = segs.max(2);
    let start = start.clamp(0.0, 0.99);
    let dens = tip_density.max(1.0);
    if dens <= 1.01 || start >= 0.99 {
        return (0..=n).map(|i| i as f32 / n as f32).collect();
    }
    // Same step size as a uniform `n`-segment strand over the full [0,1].
    let root_segs = ((start * n as f32).round() as usize).max(1);
    let tip_segs = (((1.0 - start) * n as f32 * dens).round() as usize).max(1);
    let mut out = Vec::with_capacity(root_segs + tip_segs + 1);
    for i in 0..=root_segs {
        out.push(start * (i as f32 / root_segs as f32));
    }
    // Tip starts after `start` (already the last root sample).
    for i in 1..=tip_segs {
        out.push(start + (1.0 - start) * (i as f32 / tip_segs as f32));
    }
    if let Some(last) = out.last_mut() {
        *last = 1.0;
    }
    out
}

fn sample_path(g: &HairGuide, t: f32, smooth: f32) -> (Vec3, Vec3) {
    let lin = sample_polyline(g, t);
    if smooth <= 1e-4 {
        return lin;
    }
    let cr = sample_catmull(g, t);
    (
        lin.0.lerp(cr.0, smooth),
        lin.1.lerp(cr.1, smooth).normalize_or_zero(),
    )
}

fn sample_polyline(g: &HairGuide, t: f32) -> (Vec3, Vec3) {
    let n = g.points.len();
    if n == 0 {
        return (Vec3::ZERO, Vec3::Y);
    }
    if n == 1 {
        return (g.points[0].pos, g.points[0].normal);
    }
    let f = t.clamp(0.0, 1.0) * (n - 1) as f32;
    let i = (f.floor() as usize).min(n - 2);
    let u = f - i as f32;
    let a = &g.points[i];
    let b = &g.points[i + 1];
    (
        a.pos.lerp(b.pos, u),
        a.normal.lerp(b.normal, u).normalize_or_zero(),
    )
}

fn sample_catmull(g: &HairGuide, t: f32) -> (Vec3, Vec3) {
    let n = g.points.len();
    if n < 2 {
        return sample_polyline(g, t);
    }
    let f = t.clamp(0.0, 1.0) * (n - 1) as f32;
    let i = (f.floor() as usize).min(n - 2);
    let u = f - i as f32;
    let p0 = g.points[i.saturating_sub(1)];
    let p1 = g.points[i];
    let p2 = g.points[i + 1];
    let p3 = g.points[(i + 2).min(n - 1)];
    (
        catmull(p0.pos, p1.pos, p2.pos, p3.pos, u),
        catmull(p0.normal, p1.normal, p2.normal, p3.normal, u).normalize_or_zero(),
    )
}

fn catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn lerp_strand(a: &Strand, b: &Strand, t: f32, bulge: f32) -> Strand {
    let n = a.pts.len().min(b.pts.len());
    let mut pts = Vec::with_capacity(n);
    let mut nrms = Vec::with_capacity(n);
    let mut widths = Vec::with_capacity(n);
    let bulge = bulge.max(0.0);
    for i in 0..n {
        let pa = a.pts[i];
        let pb = b.pts[i];
        let na = a.nrms[i];
        let nb = b.nrms[i];
        let nrm = na.lerp(nb, t).normalize_or_zero();
        let p = if bulge <= 1e-5 {
            pa.lerp(pb, t)
        } else {
            let chord = pb - pa;
            let clen = chord.length();
            let cdir = chord.normalize_or_zero();
            let mut up = (na + nb).normalize_or_zero();
            if up.length_squared() < 1e-8 {
                up = nrm;
            }
            up = (up - cdir * up.dot(cdir)).normalize_or_zero();
            if up.length_squared() < 1e-8 {
                up = nrm;
            }
            let ctrl = pa.lerp(pb, 0.5) + up * clen * bulge;
            let omt = 1.0 - t;
            pa * (omt * omt) + ctrl * (2.0 * omt * t) + pb * (t * t)
        };
        pts.push(p);
        nrms.push(nrm);
        let wa = a.widths.get(i).copied().unwrap_or(0.01);
        let wb = b.widths.get(i).copied().unwrap_or(0.01);
        widths.push(wa + (wb - wa) * t);
    }
    Strand {
        pts,
        nrms,
        widths,
        shade: 1.0,
        alpha: (a.alpha + b.alpha) * 0.5,
        binds: Default::default(),
    }
}

/// Planar S-wave in the tangent–normal plane from `start` to the tip.
fn apply_wave(s: &mut Strand, amp: f32, freq: f32, start: f32) {
    if amp.abs() <= 1e-6 || freq.abs() <= 1e-4 || s.pts.len() < 3 {
        return;
    }
    displace_along_strand(s, start, |u, nor, bin| {
        let fade = smoothstep01(0.0, 0.1, u);
        let ang = u * freq * std::f32::consts::TAU;
        nor * (amp * ang.sin() * fade) + bin * (amp * 0.12 * (ang * 0.5).sin() * fade)
    });
    refit_frames(s);
}

/// Tight two-axis zigzag. Higher frequency than wave; binormal harmonic
/// keeps it from reading as a flat S-wave.
fn apply_crimp(s: &mut Strand, amp: f32, freq: f32, start: f32) {
    if amp.abs() <= 1e-6 || freq.abs() <= 1e-4 || s.pts.len() < 3 {
        return;
    }
    displace_along_strand(s, start, |u, nor, bin| {
        let fade = smoothstep01(0.0, 0.06, u);
        let ang = u * freq * std::f32::consts::TAU;
        // Slightly sharpened sine so the kinks read at card resolution.
        let zig = ang.sin();
        let zig = zig.signum() * zig.abs().powf(0.65);
        nor * (amp * zig * fade) + bin * (amp * 0.7 * (ang * 2.15 + 0.8).sin() * fade)
    });
    refit_frames(s);
}

/// Tight spring: same helix as curl, but a shorter ease-in and a radius
/// that tapers toward the tip (`taper` 0 = constant tube).
fn apply_coil(s: &mut Strand, turns: f32, start: f32, radius: f32, taper: f32) {
    if turns.abs() <= 1e-4 || radius.abs() <= 1e-6 || s.pts.len() < 3 {
        return;
    }
    let taper = taper.clamp(0.0, 0.95);
    let sign = turns.signum();
    let turns = turns.abs();
    displace_along_strand(s, start, |u, nor, bin| {
        let fade = smoothstep01(0.0, 0.06, u);
        let ang = sign * u * turns * std::f32::consts::TAU;
        let r = radius * fade * (1.0 - taper * u);
        (nor * ang.cos() + bin * ang.sin()) * r
    });
    refit_frames(s);
}

/// Offset each point after `start` in the transported (nor, bin) frame.
fn displace_along_strand(s: &mut Strand, start: f32, mut offset: impl FnMut(f32, Vec3, Vec3) -> Vec3) {
    let n = s.pts.len();
    let start = start.clamp(0.0, 0.98);
    let i0 = ((start * (n - 1) as f32).ceil() as usize).min(n - 2);
    let mut dist = vec![0.0f32; n];
    for i in i0 + 1..n {
        dist[i] = dist[i - 1] + s.pts[i].distance(s.pts[i - 1]);
    }
    let tip_len = dist[n - 1].max(1e-4);
    let tans = strand_tangents(&s.pts);
    let mut nor = s.nrms.get(i0).copied().unwrap_or(Vec3::Y);
    let tan0 = tans[i0];
    nor = (nor - tan0 * nor.dot(tan0)).normalize_or_zero();
    if nor.length_squared() < 1e-8 {
        nor = tan0.cross(Vec3::Y).normalize_or_zero();
    }
    if nor.length_squared() < 1e-8 {
        nor = tan0.cross(Vec3::X).normalize_or_zero();
    }
    for i in i0 + 1..n {
        let tan = tans[i];
        nor = parallel_transport(nor, tans[i - 1], tan);
        if nor.length_squared() < 1e-8 {
            nor = (s.nrms[i] - tan * s.nrms[i].dot(tan)).normalize_or_zero();
        }
        let bin = tan.cross(nor).normalize_or_zero();
        if bin.length_squared() < 1e-8 {
            continue;
        }
        let u = (dist[i] / tip_len).clamp(0.0, 1.0);
        s.pts[i] += offset(u, nor, bin);
    }
}

/// Helical curl around the strand centerline from `start` to the tip.
/// `turns` = full revolutions, `radius` = tube radius in world units.
fn apply_curl(s: &mut Strand, turns: f32, start: f32, radius: f32) {
    if turns.abs() <= 1e-4 || radius.abs() <= 1e-6 || s.pts.len() < 3 {
        return;
    }
    let n = s.pts.len();
    let start = start.clamp(0.0, 0.98);
    let i0 = ((start * (n - 1) as f32).ceil() as usize).min(n - 2);
    let mut dist = vec![0.0f32; n];
    for i in i0 + 1..n {
        dist[i] = dist[i - 1] + s.pts[i].distance(s.pts[i - 1]);
    }
    let tip_len = dist[n - 1].max(1e-4);
    let tans = strand_tangents(&s.pts);
    let mut nor = s.nrms.get(i0).copied().unwrap_or(Vec3::Y);
    let tan0 = tans[i0];
    nor = (nor - tan0 * nor.dot(tan0)).normalize_or_zero();
    if nor.length_squared() < 1e-8 {
        nor = tan0.cross(Vec3::Y).normalize_or_zero();
    }
    if nor.length_squared() < 1e-8 {
        nor = tan0.cross(Vec3::X).normalize_or_zero();
    }
    let sign = turns.signum();
    let turns = turns.abs();
    for i in i0 + 1..n {
        let tan = tans[i];
        nor = parallel_transport(nor, tans[i - 1], tan);
        if nor.length_squared() < 1e-8 {
            nor = (s.nrms[i] - tan * s.nrms[i].dot(tan)).normalize_or_zero();
        }
        let bin = tan.cross(nor).normalize_or_zero();
        if bin.length_squared() < 1e-8 {
            continue;
        }
        let u = (dist[i] / tip_len).clamp(0.0, 1.0);
        // Soft ease-in so the curl doesn't kink hard at the start ring.
        let fade = smoothstep01(0.0, 0.12, u);
        let ang = sign * u * turns * std::f32::consts::TAU;
        let r = radius * fade;
        s.pts[i] += (nor * ang.cos() + bin * ang.sin()) * r;
        s.nrms[i] = nor;
    }
}

fn smoothstep01(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Planar tip curl in the tangent–normal plane.
/// Low |amount| = slight hook; only high |amount| winds into a tight coil.
/// Negative `amount` winds the opposite way around the same axis.
fn apply_roll(s: &mut Strand, amount: f32, start: f32) {
    if amount.abs() <= 1e-4 || s.pts.len() < 3 {
        return;
    }
    let n = s.pts.len();
    let start = start.clamp(0.0, 0.98);
    let i0 = ((start * (n - 1) as f32).ceil() as usize).min(n - 2);
    let origin = s.pts[i0];
    let tan = (s.pts[i0 + 1] - origin).normalize_or_zero();
    let mut nor = s.nrms.get(i0).copied().unwrap_or(Vec3::Y);
    nor = (nor - tan * nor.dot(tan)).normalize_or_zero();
    if nor.length_squared() < 1e-8 {
        nor = tan.cross(Vec3::Y).normalize_or_zero();
    }
    if nor.length_squared() < 1e-8 {
        nor = tan.cross(Vec3::X).normalize_or_zero();
    }
    let bin = tan.cross(nor).normalize_or_zero();
    if bin.length_squared() < 1e-8 {
        return;
    }

    let mut dist = vec![0.0f32; n];
    for i in i0 + 1..n {
        dist[i] = dist[i - 1] + s.pts[i].distance(s.pts[i - 1]);
    }
    let len = dist[n - 1].max(1e-4);
    // Squared so 0.3–0.6 stays a soft hook; coil only near the top of the slider.
    // Sign carried through separately so negative `amount` reverses winding direction.
    let sign = amount.signum();
    let a = sign * (amount.abs() * 0.5).powf(1.8);
    let mut pos = origin;
    let mut dir = tan;
    let mut nrm = nor;
    let mut prev = origin;
    for i in i0 + 1..n {
        let orig = s.pts[i];
        let ds = orig.distance(prev);
        prev = orig;
        let u = (dist[i - 1] / len).clamp(0.0, 1.0);
        let kappa = a * (2.0 + 18.0 * u.powf(1.5)) / len;
        let dtheta = kappa * ds;
        dir = rotate_around(dir, bin, dtheta);
        nrm = rotate_around(nrm, bin, dtheta);
        pos += dir * ds;
        s.pts[i] = pos;
        s.nrms[i] = nrm;
    }
}

fn rotate_around(v: Vec3, axis: Vec3, ang: f32) -> Vec3 {
    let axis = axis.normalize_or_zero();
    let c = ang.cos();
    let s = ang.sin();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

fn strand_tangents(pts: &[Vec3]) -> Vec<Vec3> {
    let n = pts.len();
    (0..n)
        .map(|i| {
            if n < 2 {
                Vec3::Y
            } else if i == 0 {
                (pts[1] - pts[0]).normalize_or_zero()
            } else if i + 1 == n {
                (pts[n - 1] - pts[n - 2]).normalize_or_zero()
            } else {
                let a = (pts[i] - pts[i - 1]).normalize_or_zero();
                let b = (pts[i + 1] - pts[i]).normalize_or_zero();
                let t = a + b;
                if t.length_squared() > 1e-10 {
                    t.normalize()
                } else {
                    a
                }
            }
        })
        .collect()
}

fn parallel_transport(v: Vec3, t0: Vec3, t1: Vec3) -> Vec3 {
    let t0 = t0.normalize_or_zero();
    let t1 = t1.normalize_or_zero();
    if t0.length_squared() < 1e-10 || t1.length_squared() < 1e-10 {
        return v;
    }
    let axis = t0.cross(t1);
    let s = axis.length();
    let c = t0.dot(t1).clamp(-1.0, 1.0);
    let mut out = if s < 1e-6 {
        if c < 0.0 {
            -v
        } else {
            v
        }
    } else {
        rotate_around(v, axis / s, s.atan2(c))
    };
    out = (out - t1 * out.dot(t1)).normalize_or_zero();
    if out.length_squared() < 1e-10 {
        out = t1.cross(Vec3::Y).normalize_or_zero();
    }
    out
}

fn refit_frames(s: &mut Strand) {
    let n = s.pts.len();
    if n < 2 || s.nrms.len() < n {
        return;
    }
    let tans = strand_tangents(&s.pts);
    let mut nrm = (s.nrms[0] - tans[0] * s.nrms[0].dot(tans[0])).normalize_or_zero();
    if nrm.length_squared() < 1e-8 {
        nrm = tans[0].cross(Vec3::Y).normalize_or_zero();
    }
    if nrm.length_squared() < 1e-8 {
        nrm = tans[0].cross(Vec3::X).normalize_or_zero();
    }
    s.nrms[0] = nrm;
    for i in 1..n {
        nrm = parallel_transport(nrm, tans[i - 1], tans[i]);
        s.nrms[i] = nrm;
    }
}

fn stable_side(tan: Vec3, n: Vec3, prev: Option<Vec3>) -> Vec3 {
    let n = (n - tan * n.dot(tan)).normalize_or_zero();
    let mut side = tan.cross(n).normalize_or_zero();
    if side.length_squared() < 1e-8 {
        side = prev
            .map(|p| (p - tan * p.dot(tan)).normalize_or_zero())
            .filter(|p| p.length_squared() > 1e-8)
            .unwrap_or_else(|| {
                let mut s = tan.cross(Vec3::Y).normalize_or_zero();
                if s.length_squared() < 1e-8 {
                    s = tan.cross(Vec3::X).normalize_or_zero();
                }
                s
            });
    }
    if let Some(p) = prev {
        if side.dot(p) < 0.0 {
            side = -side;
        }
    }
    side
}

/// Vertex color for the hair shader (`colors[0]`):
/// - `x` = projective UV scale `q` (local strand width)
/// - `y` = per-strand shade multiplier
/// - `z` = per-layer opacity
fn strand_weight(width: f32, shade: f32, alpha: f32) -> [f32; 4] {
    [width.max(1e-4), shade, alpha.clamp(0.0, 1.0), 0.0]
}

fn dummy_hair_mesh() -> HairMeshBuffers {
    (
        vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
        vec![[0.0, 1.0, 0.0]; 3],
        vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        vec![[1.0, 1.0, 1.0, 0.0]; 3],
        vec![[0; 4]; 3],
        vec![[1.0, 0.0, 0.0, 0.0]; 3],
        vec![0, 1, 2],
    )
}

pub type HairMeshBuffers = (
    Vec<[f32; 3]>,
    Vec<[f32; 3]>,
    Vec<[f32; 2]>,
    Vec<[f32; 4]>,
    Vec<[u16; 4]>,
    Vec<[f32; 4]>,
    Vec<u32>,
);

/// Front mesh always; ribbons also return a separate back-side mesh so the
/// renderer can draw back → front as two passes (soft-blend order).
pub fn mesh_from_strands(
    strands: &[Strand],
    params: &HairParams,
) -> (HairMeshBuffers, Option<HairMeshBuffers>) {
    match params.shape {
        HairShape::Ribbon => {
            let front = ribbons_from_strands(strands, false);
            let back = ribbons_from_strands(strands, true);
            (front, Some(back))
        }
        HairShape::Tube => (tubes_from_strands(strands, params), None),
    }
}

/// `back_side`: flipped normals + reversed winding (inward face of the card).
pub fn ribbons_from_strands(strands: &[Strand], back_side: bool) -> HairMeshBuffers {
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uv = Vec::new();
    let mut wts = Vec::new();
    let mut joints = Vec::new();
    let mut skin_w = Vec::new();
    let mut idx = Vec::new();
    for s in strands {
        if s.pts.len() < 2 {
            continue;
        }
        let base = pos.len() as u32;
        let nseg = s.pts.len() - 1;
        let tans = strand_tangents(&s.pts);
        let mut prev_side = None;
        for i in 0..s.pts.len() {
            let p = s.pts[i];
            let tan = tans[i];
            let n = s.nrms[i];
            let side = stable_side(tan, n, prev_side);
            prev_side = Some(side);
            // side = tan × scalp_n, so scalp_n = side × tan (outward).
            // tan × side points into the head and makes the outer card unlit.
            let mut n = side.cross(tan).normalize_or_zero();
            if back_side {
                n = -n;
            }
            let w = s.widths.get(i).copied().unwrap_or(0.01);
            let a = p - side * w;
            let b = p + side * w;
            pos.push(a.to_array());
            pos.push(b.to_array());
            nrm.push(n.to_array());
            nrm.push(n.to_array());
            let v = i as f32 / nseg.max(1) as f32;
            uv.push([0.0, v]);
            uv.push([1.0, v]);
            let wt = strand_weight(w, s.shade, s.alpha);
            wts.push(wt);
            wts.push(wt);
            let (j, sw) = skin_vertex(v, &s.binds);
            joints.push(j);
            joints.push(j);
            skin_w.push(sw);
            skin_w.push(sw);
        }
        for i in 0..nseg as u32 {
            let i0 = base + i * 2;
            if back_side {
                idx.extend_from_slice(&[i0, i0 + 3, i0 + 1, i0, i0 + 2, i0 + 3]);
            } else {
                idx.extend_from_slice(&[i0, i0 + 1, i0 + 3, i0, i0 + 3, i0 + 2]);
            }
        }
    }
    if pos.len() < 3 {
        return dummy_hair_mesh();
    }
    (pos, nrm, uv, wts, joints, skin_w, idx)
}

fn tubes_from_strands(strands: &[Strand], params: &HairParams) -> HairMeshBuffers {
    let poly = section_polygon(&params.section_curve);
    let sides = poly.len();
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uv = Vec::new();
    let mut wts = Vec::new();
    let mut joints = Vec::new();
    let mut skin_w = Vec::new();
    let mut idx = Vec::new();
    for s in strands {
        if s.pts.len() < 2 || sides < 3 {
            continue;
        }
        let rings = s.pts.len();
        let nseg = rings - 1;
        let tans = strand_tangents(&s.pts);
        let mut prev_n = None;
        let mut ring_pts = Vec::with_capacity(rings * sides);
        for i in 0..rings {
            let p = s.pts[i];
            let tan = tans[i];
            let n = (s.nrms[i] - tan * s.nrms[i].dot(tan)).normalize_or_zero();
            let mut n = if n.length_squared() > 1e-8 {
                n
            } else {
                parallel_transport(
                    prev_n.unwrap_or(s.nrms[i]),
                    if i == 0 { tan } else { tans[i - 1] },
                    tan,
                )
            };
            if n.length_squared() < 1e-8 {
                n = tan.cross(Vec3::Y).normalize_or_zero();
            }
            if n.length_squared() < 1e-8 {
                n = tan.cross(Vec3::X).normalize_or_zero();
            }
            if let Some(pn) = prev_n {
                if n.dot(pn) < 0.0 {
                    n = -n;
                }
            }
            prev_n = Some(n);
            let b = tan.cross(n).normalize_or_zero();
            let w = s.widths.get(i).copied().unwrap_or(0.01);
            for k in 0..sides {
                let q = poly[k];
                ring_pts.push(p + n * (q.y * w) + b * (q.x * w));
            }
        }
        for i in 0..nseg {
            let v0 = i as f32 / nseg.max(1) as f32;
            let v1 = (i + 1) as f32 / nseg.max(1) as f32;
            // Strand point sits near the scalp side of the profile; use the ring
            // centroid so "outward" is out of the volume, not toward the inner wall.
            let mut centroid = Vec3::ZERO;
            for k in 0..sides {
                centroid += ring_pts[i * sides + k] + ring_pts[(i + 1) * sides + k];
            }
            centroid /= (sides * 2) as f32;
            for k in 0..sides {
                let k1 = (k + 1) % sides;
                let a0 = ring_pts[i * sides + k];
                let a1 = ring_pts[i * sides + k1];
                let b0 = ring_pts[(i + 1) * sides + k];
                let b1 = ring_pts[(i + 1) * sides + k1];
                let mut fnrm = (a1 - a0).cross(b0 - a0).normalize_or_zero();
                if fnrm.length_squared() < 1e-10 {
                    fnrm = (b1 - a0).cross(b0 - a1).normalize_or_zero();
                }
                let mid = (a0 + a1 + b0 + b1) * 0.25;
                let flip = fnrm.dot(mid - centroid) < 0.0;
                if flip {
                    fnrm = -fnrm;
                }
                let base = pos.len() as u32;
                let na = fnrm.to_array();
                let u0 = k as f32 / sides as f32;
                let u1 = (k + 1) as f32 / sides as f32;
                pos.push(a0.to_array());
                pos.push(a1.to_array());
                pos.push(b0.to_array());
                pos.push(b1.to_array());
                nrm.extend_from_slice(&[na, na, na, na]);
                uv.push([u0, v0]);
                uv.push([u1, v0]);
                uv.push([u0, v1]);
                uv.push([u1, v1]);
                let w0 = s.widths.get(i).copied().unwrap_or(0.01);
                let w1 = s.widths.get(i + 1).copied().unwrap_or(0.01);
                let wt0 = strand_weight(w0, s.shade, s.alpha);
                let wt1 = strand_weight(w1, s.shade, s.alpha);
                wts.push(wt0);
                wts.push(wt0);
                wts.push(wt1);
                wts.push(wt1);
                let (j0, sw0) = skin_vertex(v0, &s.binds);
                let (j1, sw1) = skin_vertex(v1, &s.binds);
                joints.push(j0);
                joints.push(j0);
                joints.push(j1);
                joints.push(j1);
                skin_w.push(sw0);
                skin_w.push(sw0);
                skin_w.push(sw1);
                skin_w.push(sw1);
                if flip {
                    idx.extend_from_slice(&[base, base + 2, base + 3, base, base + 3, base + 1]);
                } else {
                    idx.extend_from_slice(&[base, base + 3, base + 2, base, base + 1, base + 3]);
                }
            }
        }
    }
    if pos.len() < 3 {
        return dummy_hair_mesh();
    }
    (pos, nrm, uv, wts, joints, skin_w, idx)
}

/// Closed prism from the section curve: X is always full lock width,
/// Y is height at each key. Keys become hard corners (no polar radius).
fn section_polygon(curve: &HairCurve) -> Vec<Vec2> {
    let mut top: Vec<Vec2> = curve
        .points
        .iter()
        .map(|p| Vec2::new(p.t.clamp(0.0, 1.0) * 2.0 - 1.0, p.v.max(0.0)))
        .collect();
    if top.len() < 2 {
        top = vec![Vec2::new(-1.0, 1.0), Vec2::new(1.0, 1.0)];
    }
    top.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    top[0].x = -1.0;
    if let Some(last) = top.last_mut() {
        last.x = 1.0;
    }
    let mut dedup: Vec<Vec2> = Vec::with_capacity(top.len());
    for p in top {
        if let Some(prev) = dedup.last() {
            if prev.distance_squared(p) < 1e-10 {
                continue;
            }
        }
        dedup.push(p);
    }
    let max_y = dedup
        .iter()
        .map(|p| p.y)
        .fold(0.0_f32, f32::max)
        .max(0.05);
    let back_y = -0.18 * max_y;
    let mut poly = dedup;
    poly.push(Vec2::new(1.0, back_y));
    poly.push(Vec2::new(-1.0, back_y));
    poly
}

pub fn apply_hair_mesh(mesh: &mut Mesh, (pos, nrm, uv, colors, joints, weights, idx): HairMeshBuffers) {
    mesh.positions = pos;
    mesh.normals = nrm;
    mesh.uvs = vec![uv];
    mesh.indices = idx;
    mesh.joints = vec![joints];
    mesh.weights = vec![weights];
    mesh.colors = vec![colors];
    mesh.mark_changed();
}

