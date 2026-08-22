//! Hermite animation curve (same math as mega-ui, no UI dependency).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HairCurvePreset {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Custom,
}

#[derive(Clone, Copy, Debug)]
pub struct HairCurvePoint {
    pub t: f32,
    pub v: f32,
    pub tangent_out: f32,
}

#[derive(Clone, Debug)]
pub struct HairCurve {
    pub points: Vec<HairCurvePoint>,
    pub preset: HairCurvePreset,
}

impl Default for HairCurve {
    fn default() -> Self {
        ease_in_out()
    }
}

pub fn ease_in_out() -> HairCurve {
    let mut c = HairCurve {
        points: vec![
            HairCurvePoint {
                t: 0.0,
                v: 0.0,
                tangent_out: 0.0,
            },
            HairCurvePoint {
                t: 1.0,
                v: 1.0,
                tangent_out: 0.0,
            },
        ],
        preset: HairCurvePreset::EaseInOut,
    };
    let t = auto_smooth_tangents(&c);
    for (i, p) in c.points.iter_mut().enumerate() {
        p.tangent_out = t[i];
    }
    c
}

pub fn sample_curve(curve: &HairCurve, t: f32) -> f32 {
    let pts = &curve.points;
    if pts.is_empty() {
        return 0.0;
    }
    if pts.len() == 1 {
        return pts[0].v;
    }
    if t <= pts[0].t {
        return pts[0].v;
    }
    if t >= pts[pts.len() - 1].t {
        return pts[pts.len() - 1].v;
    }
    let smooth = auto_smooth_tangents(curve);
    for i in 0..pts.len() - 1 {
        let p0 = &pts[i];
        let p1 = &pts[i + 1];
        if t >= p0.t && t <= p1.t {
            let dt = (p1.t - p0.t).max(1e-5);
            let u = (t - p0.t) / dt;
            let m0 = sample_tangent(curve, &smooth, i) * dt;
            let m1 = sample_tangent(curve, &smooth, i + 1) * dt;
            return hermite(p0.v, p1.v, m0, m1, u);
        }
    }
    pts[pts.len() - 1].v
}

fn auto_smooth_tangents(curve: &HairCurve) -> Vec<f32> {
    let n = curve.points.len();
    let mut tangents = vec![0.0; n];
    if n < 2 {
        return tangents;
    }
    for i in 0..n {
        if i == 0 {
            let dt = curve.points[1].t - curve.points[0].t;
            tangents[i] = if dt > 1e-5 {
                (curve.points[1].v - curve.points[0].v) / dt
            } else {
                0.0
            };
        } else if i == n - 1 {
            let dt = curve.points[i].t - curve.points[i - 1].t;
            tangents[i] = if dt > 1e-5 {
                (curve.points[i].v - curve.points[i - 1].v) / dt
            } else {
                0.0
            };
        } else {
            let dt = curve.points[i + 1].t - curve.points[i - 1].t;
            tangents[i] = if dt > 1e-5 {
                (curve.points[i + 1].v - curve.points[i - 1].v) / dt
            } else {
                0.0
            };
        }
    }
    tangents
}

fn sample_tangent(curve: &HairCurve, smooth: &[f32], i: usize) -> f32 {
    if curve.preset == HairCurvePreset::Custom {
        smooth.get(i).copied().unwrap_or(0.0)
    } else {
        curve.points.get(i).map(|p| p.tangent_out).unwrap_or(0.0)
    }
}

fn hermite(p0: f32, p1: f32, m0: f32, m1: f32, u: f32) -> f32 {
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

pub fn sample_gradient(stops: &[HairColorStop], t: f32) -> [f32; 4] {
    if stops.is_empty() {
        return [1.0, 1.0, 1.0, 1.0];
    }
    if stops.len() == 1 {
        return stops[0].color;
    }
    let t = t.clamp(0.0, 1.0);
    let mut ordered: Vec<&HairColorStop> = stops.iter().collect();
    ordered.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    if t <= ordered[0].t {
        return ordered[0].color;
    }
    let last = *ordered.last().unwrap();
    if t >= last.t {
        return last.color;
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
                a.color[3] + (b.color[3] - a.color[3]) * u,
            ];
        }
    }
    last.color
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairColorStop {
    pub t: f32,
    pub color: [f32; 4],
}
