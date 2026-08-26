use glam::{Mat4, Quat, Vec3};
use std::f32::consts::TAU;

/// Stick skeleton line / joint sizes (screen-space px).
pub const SKELETON_LINE_W: f32 = 3.5;
pub const SKELETON_OUTLINE_W: f32 = 7.5;
pub const SKELETON_JOINT: f32 = 8.0;
pub const SKELETON_JOINT_OUTLINE: f32 = 12.0;

pub const SKELETON_FILL: [f32; 4] = [0.86, 0.86, 0.90, 1.0];
pub const SKELETON_OUTLINE: [f32; 4] = [0.02, 0.02, 0.04, 1.0];
pub const SKELETON_SEL_FILL: [f32; 4] = [0.35, 0.72, 1.0, 1.0];
pub const SKELETON_SEL_OUTLINE: [f32; 4] = [0.06, 0.22, 0.55, 1.0];
pub const SKELETON_IK_TARGET_FILL: [f32; 4] = [1.0, 0.55, 0.12, 1.0];
pub const SKELETON_IK_TARGET_OUTLINE: [f32; 4] = [0.45, 0.18, 0.02, 1.0];
pub const SKELETON_IK_POLE_FILL: [f32; 4] = [0.85, 0.35, 0.95, 1.0];
pub const SKELETON_IK_POLE_OUTLINE: [f32; 4] = [0.35, 0.08, 0.45, 1.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    /// Translate plane handle (XY).
    Xy,
    /// Translate plane handle (YZ).
    Yz,
    /// Translate plane handle (ZX).
    Zx,
    /// Uniform scale (center cuboid).
    Uniform,
}

#[derive(Clone, Copy)]
pub struct GizmoOpts {
    pub mode: GizmoMode,
    /// World-space axis length / ring radius.
    pub size: f32,
    pub highlight: Option<GizmoAxis>,
    /// Camera eye — fades the back half of rotation rings when set.
    pub eye: Option<Vec3>,
    /// Active rotation feedback (start → current angle on the ring).
    pub rotate_arc: Option<GizmoRotateArc>,
    /// Gizmos default to overlay (no depth test).
    pub depth_test: bool,
}

/// Rotation drag visualization on a ring.
///
/// `u`/`v` must stay frozen for the drag (world basis at grab time) so the
/// filled sector stays aligned while the object rotates.
#[derive(Clone, Copy)]
pub struct GizmoRotateArc {
    pub axis: GizmoAxis,
    pub u: Vec3,
    pub v: Vec3,
    /// Radians: 0 along `u`, +π/2 toward `v`.
    pub start: f32,
    pub current: f32,
}

impl Default for GizmoOpts {
    fn default() -> Self {
        Self {
            mode: GizmoMode::Translate,
            size: 1.0,
            highlight: None,
            eye: None,
            rotate_arc: None,
            depth_test: false,
        }
    }
}

/// Orthonormal ring basis for a rotation axis (angle 0 along `u`, +90° toward `v`).
/// Matches right-handed `Quat::from_axis_angle` around the axis.
pub fn gizmo_ring_basis(axis: GizmoAxis, x: Vec3, y: Vec3, z: Vec3) -> (Vec3, Vec3) {
    match axis {
        GizmoAxis::X => (y, z),
        // +Y RH: +X rotates toward -Z
        GizmoAxis::Y => (x, -z),
        GizmoAxis::Z => (x, y),
        _ => (x, y),
    }
}

/// World size that covers roughly `pixels` of screen height at `distance`.
pub fn gizmo_screen_size(distance: f32, fov_y: f32, viewport_h: f32, pixels: f32) -> f32 {
    let h = viewport_h.max(1.0);
    (fov_y * 0.5).tan() * distance.max(1e-3) * 2.0 * pixels / h
}

#[derive(Clone, Copy)]
pub struct LineOpts {
    pub color_from: [f32; 4],
    pub color_to: [f32; 4],
    /// Screen-space width in pixels at `start`.
    pub width_from: f32,
    /// Screen-space width in pixels at `end`.
    pub width_to: f32,
    pub depth_test: bool,
}

impl Default for LineOpts {
    fn default() -> Self {
        Self {
            color_from: [1.0, 1.0, 1.0, 1.0],
            color_to: [1.0, 1.0, 1.0, 1.0],
            width_from: 1.0,
            width_to: 1.0,
            depth_test: true,
        }
    }
}

impl LineOpts {
    /// Solid color, default width and depth test.
    pub fn color(color: [f32; 4]) -> Self {
        Self {
            color_from: color,
            color_to: color,
            ..Self::default()
        }
    }

    pub fn width(mut self, width: f32) -> Self {
        self.width_from = width;
        self.width_to = width;
        self
    }

    pub fn overlay(mut self) -> Self {
        self.depth_test = false;
        self
    }
}

#[derive(Clone, Copy)]
pub struct DebugLine {
    pub start: Vec3,
    pub end: Vec3,
    pub opts: LineOpts,
}

#[derive(Clone, Copy)]
pub struct DebugPoint {
    pub position: Vec3,
    pub color: [f32; 4],
    /// Size in pixels (screen-space square).
    pub size: f32,
    pub depth_test: bool,
}

#[derive(Clone, Copy)]
pub struct PolyOpts {
    pub color: [f32; 4],
    pub depth_test: bool,
}

impl Default for PolyOpts {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 0.45],
            depth_test: false,
        }
    }
}

impl PolyOpts {
    pub fn color(color: [f32; 4]) -> Self {
        Self {
            color,
            ..Self::default()
        }
    }

    pub fn overlay(mut self) -> Self {
        self.depth_test = false;
        self
    }
}

#[derive(Clone, Copy)]
pub struct DebugTri {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub color: [f32; 4],
    pub depth_test: bool,
}

#[derive(Default)]
pub struct DebugDraw {
    pub lines: Vec<DebugLine>,
    pub points: Vec<DebugPoint>,
    pub tris: Vec<DebugTri>,
}

impl DebugDraw {
    pub fn clear(&mut self) {
        self.lines.clear();
        self.points.clear();
        self.tris.clear();
    }

    pub fn line(&mut self, start: Vec3, end: Vec3, opts: LineOpts) {
        self.lines.push(DebugLine { start, end, opts });
    }

    pub fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, opts: PolyOpts) {
        self.tris.push(DebugTri {
            a,
            b,
            c,
            color: opts.color,
            depth_test: opts.depth_test,
        });
    }

    /// Convex n-gon as a triangle fan from `pts[0]`.
    pub fn polygon(&mut self, pts: &[Vec3], opts: PolyOpts) {
        if pts.len() < 3 {
            return;
        }
        for i in 1..pts.len() - 1 {
            self.tri(pts[0], pts[i], pts[i + 1], opts);
        }
    }

    /// Filled circular sector in plane spanned by `u`/`v` (unit preferred).
    pub fn sector(
        &mut self,
        origin: Vec3,
        u: Vec3,
        v: Vec3,
        radius: f32,
        angle0: f32,
        angle1: f32,
        segments: u32,
        opts: PolyOpts,
    ) {
        let mut delta = angle1 - angle0;
        while delta > std::f32::consts::PI {
            delta -= TAU;
        }
        while delta < -std::f32::consts::PI {
            delta += TAU;
        }
        if delta.abs() < 1e-5 || radius <= 1e-6 {
            return;
        }
        let segs = segments.max(2);
        let mut rim = Vec::with_capacity(segs as usize + 1);
        for i in 0..=segs {
            let t = i as f32 / segs as f32;
            let a = angle0 + delta * t;
            rim.push(origin + (u * a.cos() + v * a.sin()) * radius);
        }
        for w in rim.windows(2) {
            self.tri(origin, w[0], w[1], opts);
        }
    }

    pub fn point(&mut self, position: Vec3, color: [f32; 4]) {
        self.point_sized(position, color, 8.0);
    }

    pub fn point_overlay(&mut self, position: Vec3, color: [f32; 4]) {
        self.point_ex(position, color, 8.0, false);
    }

    pub fn point_sized(&mut self, position: Vec3, color: [f32; 4], size: f32) {
        self.point_ex(position, color, size, true);
    }

    pub fn point_ex(&mut self, position: Vec3, color: [f32; 4], size: f32, depth_test: bool) {
        self.points.push(DebugPoint {
            position,
            color,
            size,
            depth_test,
        });
    }

    /// XYZ axes: X red, Y green, Z blue.
    pub fn axes(&mut self, origin: Vec3, len: f32) {
        self.line(origin, origin + Vec3::X * len, LineOpts::color([1.0, 0.2, 0.2, 1.0]));
        self.line(origin, origin + Vec3::Y * len, LineOpts::color([0.2, 1.0, 0.2, 1.0]));
        self.line(origin, origin + Vec3::Z * len, LineOpts::color([0.3, 0.5, 1.0, 1.0]));
    }

    /// XZ ground grid centered on `origin` (typically [`Vec3::ZERO`]).
    ///
    /// `half_extent` is half the grid width/depth; `step` is cell size.
    /// Every `major_every` lines (including the center) use `major_color`.
    pub fn grid(
        &mut self,
        origin: Vec3,
        half_extent: f32,
        step: f32,
        color: [f32; 4],
        major_every: u32,
        major_color: [f32; 4],
    ) {
        let step = step.max(1e-4);
        let half = half_extent.max(step);
        let n = (half / step).round() as i32;
        let major_every = major_every.max(1) as i32;
        // Ground is XZ (Y up): highlight world X (red) and Z (green) through origin.
        let axis_x = [0.55, 0.18, 0.18, (major_color[3] * 0.85).clamp(0.2, 0.45)];
        let axis_z = [0.18, 0.50, 0.22, (major_color[3] * 0.85).clamp(0.2, 0.45)];
        for i in -n..=n {
            let t = i as f32 * step;
            let is_major = i % major_every == 0;
            let col = if is_major { major_color } else { color };
            // Lines parallel to X (vary Z). z=0 → world X axis.
            let col_x = if i == 0 { axis_x } else { col };
            self.line(
                origin + Vec3::new(-half, 0.0, t),
                origin + Vec3::new(half, 0.0, t),
                LineOpts::color(col_x),
            );
            // Lines parallel to Z (vary X). x=0 → world Z axis.
            let col_z = if i == 0 { axis_z } else { col };
            self.line(
                origin + Vec3::new(t, 0.0, -half),
                origin + Vec3::new(t, 0.0, half),
                LineOpts::color(col_z),
            );
        }
    }

    pub fn axes_overlay(&mut self, origin: Vec3, len: f32) {
        self.line(
            origin,
            origin + Vec3::X * len,
            LineOpts::color([1.0, 0.2, 0.2, 1.0]).overlay(),
        );
        self.line(
            origin,
            origin + Vec3::Y * len,
            LineOpts::color([0.2, 1.0, 0.2, 1.0]).overlay(),
        );
        self.line(
            origin,
            origin + Vec3::Z * len,
            LineOpts::color([0.3, 0.5, 1.0, 1.0]).overlay(),
        );
    }

    /// Unit box `[-0.5, 0.5]³` transformed by `m`.
    pub fn box_transform(&mut self, m: Mat4, color: [f32; 4], depth_test: bool) {
        let c = |x, y, z| m.transform_point3(Vec3::new(x, y, z));
        let p = [
            c(-0.5, -0.5, -0.5),
            c(0.5, -0.5, -0.5),
            c(0.5, 0.5, -0.5),
            c(-0.5, 0.5, -0.5),
            c(-0.5, -0.5, 0.5),
            c(0.5, -0.5, 0.5),
            c(0.5, 0.5, 0.5),
            c(-0.5, 0.5, 0.5),
        ];
        let opts = LineOpts {
            depth_test,
            ..LineOpts::color(color)
        };
        for (a, b) in [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ] {
            self.line(p[a], p[b], opts);
        }
    }

    pub fn box_aabb(&mut self, min: Vec3, max: Vec3, color: [f32; 4], depth_test: bool) {
        let center = (min + max) * 0.5;
        let size = max - min;
        self.box_transform(
            Mat4::from_translation(center) * Mat4::from_scale(size),
            color,
            depth_test,
        );
    }

    /// Unit sphere transformed by `m` (scale → ellipsoid). Three great circles.
    pub fn sphere_transform(&mut self, m: Mat4, color: [f32; 4], segments: u32, depth_test: bool) {
        let segs = segments.max(3);
        let opts = LineOpts {
            depth_test,
            ..LineOpts::color(color)
        };
        for axis in 0..3 {
            for i in 0..segs {
                let a0 = TAU * i as f32 / segs as f32;
                let a1 = TAU * (i + 1) as f32 / segs as f32;
                let (c0, s0) = (a0.cos(), a0.sin());
                let (c1, s1) = (a1.cos(), a1.sin());
                let (p0, p1) = match axis {
                    0 => (Vec3::new(c0, s0, 0.0), Vec3::new(c1, s1, 0.0)), // XY
                    1 => (Vec3::new(c0, 0.0, s0), Vec3::new(c1, 0.0, s1)), // XZ
                    _ => (Vec3::new(0.0, c0, s0), Vec3::new(0.0, c1, s1)), // YZ
                };
                self.line(m.transform_point3(p0), m.transform_point3(p1), opts);
            }
        }
    }

    pub fn sphere(&mut self, center: Vec3, radius: f32, color: [f32; 4], depth_test: bool) {
        self.sphere_transform(
            Mat4::from_translation(center) * Mat4::from_scale(Vec3::splat(radius)),
            color,
            24,
            depth_test,
        );
    }

    /// Wireframe capsule: cylinder between `a`/`b` plus hemispherical caps.
    pub fn capsule(&mut self, a: Vec3, b: Vec3, radius: f32, color: [f32; 4], depth_test: bool) {
        let r = radius.max(1e-5);
        let ab = b - a;
        let len = ab.length();
        if len < 1e-5 {
            self.sphere(a, r, color, depth_test);
            return;
        }
        let axis = ab / len;
        let u = axis.any_orthonormal_vector();
        let v = axis.cross(u);
        let lon = 16u32;
        let hemi = 6u32;
        let mut opts = LineOpts::color(color);
        opts.depth_test = depth_test;
        if !depth_test {
            opts = opts.overlay();
        }

        let ring = |center: Vec3, ru: f32| {
            (0..lon).map(move |i| {
                let ang = TAU * i as f32 / lon as f32;
                center + (u * ang.cos() + v * ang.sin()) * ru
            })
        };

        let draw_ring = |s: &mut Self, center: Vec3, ru: f32| {
            let mut prev = center + u * ru;
            for i in 1..=lon {
                let ang = TAU * i as f32 / lon as f32;
                let p = center + (u * ang.cos() + v * ang.sin()) * ru;
                s.line(prev, p, opts);
                prev = p;
            }
        };

        // Cylinder wall: equator rings at both ends + longitudes.
        draw_ring(self, a, r);
        draw_ring(self, b, r);
        for i in 0..lon {
            let ang = TAU * i as f32 / lon as f32;
            let off = (u * ang.cos() + v * ang.sin()) * r;
            self.line(a + off, b + off, opts);
        }

        // Hemispheres: a points opposite `axis`, b along `axis`.
        for cap_sign in [-1.0f32, 1.0] {
            let origin = if cap_sign < 0.0 { a } else { b };
            let mut prev_pts: Vec<Vec3> = ring(origin, r).collect();
            for j in 1..=hemi {
                let t = j as f32 / hemi as f32;
                let lat = t * std::f32::consts::FRAC_PI_2;
                let ring_r = r * lat.cos();
                let center = origin + axis * cap_sign * (r * lat.sin());
                let mut pts = Vec::with_capacity(lon as usize);
                for i in 0..lon {
                    let ang = TAU * i as f32 / lon as f32;
                    pts.push(center + (u * ang.cos() + v * ang.sin()) * ring_r);
                }
                if j < hemi {
                    draw_ring(self, center, ring_r);
                } else {
                    let pole = origin + axis * cap_sign * r;
                    for &p in &prev_pts {
                        self.line(p, pole, opts);
                    }
                    break;
                }
                for i in 0..lon as usize {
                    self.line(prev_pts[i], pts[i], opts);
                }
                prev_pts = pts;
            }
        }
    }

    /// Stick bone: thick outline under thinner fill (screen-space px widths).
    pub fn bone(
        &mut self,
        from: Vec3,
        to: Vec3,
        fill: [f32; 4],
        outline: [f32; 4],
        overlay: bool,
    ) {
        let mut outline_opts = LineOpts::color(outline).width(SKELETON_OUTLINE_W);
        let mut fill_opts = LineOpts::color(fill).width(SKELETON_LINE_W);
        if overlay {
            outline_opts = outline_opts.overlay();
            fill_opts = fill_opts.overlay();
        }
        self.line(from, to, outline_opts);
        self.line(from, to, fill_opts);
    }

    /// Joint dot: outline then fill (points draw after lines → sit on top).
    pub fn bone_joint(
        &mut self,
        pos: Vec3,
        fill: [f32; 4],
        outline: [f32; 4],
        fill_px: f32,
        outline_px: f32,
        overlay: bool,
    ) {
        let depth_test = !overlay;
        self.point_ex(pos, outline, outline_px, depth_test);
        self.point_ex(pos, fill, fill_px, depth_test);
    }

    /// Transform gizmo at `pos` with local `rotation` (object scale ignored).
    pub fn gizmo(&mut self, pos: Vec3, rotation: Quat, opts: GizmoOpts) {
        let size = opts.size.max(1e-4);
        let x = rotation * Vec3::X;
        let y = rotation * Vec3::Y;
        let z = rotation * Vec3::Z;
        match opts.mode {
            GizmoMode::Translate => self.gizmo_translate(pos, x, y, z, size, &opts),
            GizmoMode::Rotate => self.gizmo_rotate(pos, x, y, z, size, &opts),
            GizmoMode::Scale => self.gizmo_scale(pos, x, y, z, size, &opts),
        }
    }

    fn gizmo_translate(
        &mut self,
        pos: Vec3,
        x: Vec3,
        y: Vec3,
        z: Vec3,
        size: f32,
        opts: &GizmoOpts,
    ) {
        self.gizmo_arrow(pos, x, size, GizmoAxis::X, opts);
        self.gizmo_arrow(pos, y, size, GizmoAxis::Y, opts);
        self.gizmo_arrow(pos, z, size, GizmoAxis::Z, opts);
        self.gizmo_plane(pos, x, y, size, GizmoAxis::Xy, opts);
        self.gizmo_plane(pos, y, z, size, GizmoAxis::Yz, opts);
        self.gizmo_plane(pos, z, x, size, GizmoAxis::Zx, opts);
        // Pivot
        let (c, w) = gizmo_neutral_style(opts.highlight.is_some());
        self.gizmo_line(pos - x * size * 0.04, pos + x * size * 0.04, c, w, opts);
        self.gizmo_line(pos - y * size * 0.04, pos + y * size * 0.04, c, w, opts);
        self.gizmo_line(pos - z * size * 0.04, pos + z * size * 0.04, c, w, opts);
    }

    fn gizmo_rotate(
        &mut self,
        pos: Vec3,
        x: Vec3,
        y: Vec3,
        z: Vec3,
        size: f32,
        opts: &GizmoOpts,
    ) {
        for axis in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
            let (u, v) = gizmo_ring_basis(axis, x, y, z);
            self.gizmo_ring(pos, u, v, size, axis, opts);
        }
        // View-aligned helper ring (screen orbit cue)
        if let Some(eye) = opts.eye {
            let view = (eye - pos).normalize_or_zero();
            if view.length_squared() > 1e-6 {
                let a = view.any_orthonormal_vector();
                let b = view.cross(a).normalize_or_zero();
                let (c, w) = gizmo_neutral_style(opts.highlight.is_some());
                self.gizmo_ring_raw(pos, a, b, size * 1.12, c, w * 0.85, opts, None);
            }
        }
        if let Some(arc) = opts.rotate_arc {
            self.gizmo_rotate_arc(pos, size, arc, opts);
        }
    }

    fn gizmo_rotate_arc(&mut self, pos: Vec3, size: f32, arc: GizmoRotateArc, opts: &GizmoOpts) {
        let u = arc.u.normalize_or_zero();
        let v = arc.v.normalize_or_zero();
        let (axis_color, _) = gizmo_axis_style(arc.axis, Some(arc.axis));
        let fill = [axis_color[0], axis_color[1], axis_color[2], 0.35];
        let start_c = [0.95, 0.95, 0.98, 1.0];
        let cur_c = axis_color;

        let rim = |a: f32| pos + (u * a.cos() + v * a.sin()) * size;
        let p0 = rim(arc.start);
        let p1 = rim(arc.current);

        // Filled sector (frozen u/v from drag start)
        self.sector(
            pos,
            u,
            v,
            size,
            arc.start,
            arc.current,
            48,
            PolyOpts {
                color: fill,
                depth_test: opts.depth_test,
            },
        );

        // Start / current radials + outer stroke
        self.gizmo_line(pos, p0, start_c, 2.4, opts);
        self.gizmo_line(pos, p1, cur_c, 3.2, opts);

        let mut delta = arc.current - arc.start;
        while delta > std::f32::consts::PI {
            delta -= TAU;
        }
        while delta < -std::f32::consts::PI {
            delta += TAU;
        }
        if delta.abs() < 1e-4 {
            return;
        }
        let steps = ((delta.abs() / TAU) * 48.0).ceil().max(2.0) as i32;
        for i in 0..steps {
            let t0 = i as f32 / steps as f32;
            let t1 = (i + 1) as f32 / steps as f32;
            self.gizmo_line(
                rim(arc.start + delta * t0),
                rim(arc.start + delta * t1),
                cur_c,
                3.4,
                opts,
            );
        }
    }

    fn gizmo_scale(
        &mut self,
        pos: Vec3,
        x: Vec3,
        y: Vec3,
        z: Vec3,
        size: f32,
        opts: &GizmoOpts,
    ) {
        self.gizmo_scale_arm(pos, x, size, GizmoAxis::X, opts);
        self.gizmo_scale_arm(pos, y, size, GizmoAxis::Y, opts);
        self.gizmo_scale_arm(pos, z, size, GizmoAxis::Z, opts);
        // Uniform handle — larger center cube
        let (c, w) = gizmo_axis_style(GizmoAxis::Uniform, opts.highlight);
        let hs = size * 0.16;
        let m = Mat4::from_translation(pos) * Mat4::from_scale(Vec3::splat(hs * 2.0));
        self.box_transform(m, c, opts.depth_test);
        // Cross accent so the center reads clearly
        self.gizmo_line(pos - x * hs, pos + x * hs, c, w, opts);
        self.gizmo_line(pos - y * hs, pos + y * hs, c, w, opts);
        self.gizmo_line(pos - z * hs, pos + z * hs, c, w, opts);
    }

    fn gizmo_arrow(&mut self, pos: Vec3, dir: Vec3, size: f32, axis: GizmoAxis, opts: &GizmoOpts) {
        let (color, width) = gizmo_axis_style(axis, opts.highlight);
        let tip = pos + dir * size;
        let neck = pos + dir * (size * 0.78);
        self.gizmo_line(pos + dir * (size * 0.08), neck, color, width, opts);

        // Cone head
        let radius = size * 0.075;
        let ortho = dir.any_orthonormal_vector() * radius;
        let ortho2 = dir.cross(ortho).normalize_or_zero() * radius;
        let segs = 10;
        for i in 0..segs {
            let a0 = TAU * i as f32 / segs as f32;
            let a1 = TAU * (i + 1) as f32 / segs as f32;
            let p0 = neck + ortho * a0.cos() + ortho2 * a0.sin();
            let p1 = neck + ortho * a1.cos() + ortho2 * a1.sin();
            self.gizmo_line(tip, p0, color, width, opts);
            self.gizmo_line(p0, p1, color, width * 0.85, opts);
        }
    }

    fn gizmo_plane(
        &mut self,
        pos: Vec3,
        a: Vec3,
        b: Vec3,
        size: f32,
        axis: GizmoAxis,
        opts: &GizmoOpts,
    ) {
        let (color, width) = gizmo_axis_style(axis, opts.highlight);
        let u0 = size * 0.22;
        let u1 = size * 0.42;
        let p00 = pos + a * u0 + b * u0;
        let p10 = pos + a * u1 + b * u0;
        let p11 = pos + a * u1 + b * u1;
        let p01 = pos + a * u0 + b * u1;
        // Square + inner cross for readability
        self.gizmo_line(p00, p10, color, width, opts);
        self.gizmo_line(p10, p11, color, width, opts);
        self.gizmo_line(p11, p01, color, width, opts);
        self.gizmo_line(p01, p00, color, width, opts);
        self.gizmo_line(p00, p11, color, width * 0.65, opts);
    }

    fn gizmo_ring(
        &mut self,
        pos: Vec3,
        u: Vec3,
        v: Vec3,
        radius: f32,
        axis: GizmoAxis,
        opts: &GizmoOpts,
    ) {
        let (color, width) = gizmo_axis_style(axis, opts.highlight);
        self.gizmo_ring_raw(pos, u, v, radius, color, width, opts, opts.eye);
    }

    fn gizmo_ring_raw(
        &mut self,
        pos: Vec3,
        u: Vec3,
        v: Vec3,
        radius: f32,
        color: [f32; 4],
        width: f32,
        opts: &GizmoOpts,
        eye: Option<Vec3>,
    ) {
        let segs = 64;
        let to_cam = eye.map(|e| (e - pos).normalize_or_zero());
        for i in 0..segs {
            let a0 = TAU * i as f32 / segs as f32;
            let a1 = TAU * (i + 1) as f32 / segs as f32;
            let p0 = pos + (u * a0.cos() + v * a0.sin()) * radius;
            let p1 = pos + (u * a1.cos() + v * a1.sin()) * radius;
            let mid = (p0 + p1) * 0.5;
            let mut c = color;
            let mut w = width;
            if let Some(tc) = to_cam {
                // Keep the camera-facing half crisp; fade the back.
                let facing = (mid - pos).normalize_or_zero().dot(tc);
                if facing < -0.05 {
                    c[3] *= 0.18;
                    w *= 0.7;
                } else if facing < 0.25 {
                    let t = (facing + 0.05) / 0.3;
                    c[3] *= 0.18 + 0.82 * t;
                }
            }
            self.gizmo_line(p0, p1, c, w, opts);
        }
    }

    fn gizmo_scale_arm(
        &mut self,
        pos: Vec3,
        dir: Vec3,
        size: f32,
        axis: GizmoAxis,
        opts: &GizmoOpts,
    ) {
        let (color, width) = gizmo_axis_style(axis, opts.highlight);
        let tip = pos + dir * size;
        self.gizmo_line(pos + dir * (size * 0.1), tip - dir * (size * 0.08), color, width, opts);
        let hs = size * 0.09;
        // Axis-aligned handle cube in local frame
        let right = dir.any_orthonormal_vector();
        let up = dir.cross(right).normalize_or_zero();
        let m = Mat4::from_cols(
            (right * hs * 2.0).extend(0.0),
            (up * hs * 2.0).extend(0.0),
            (dir * hs * 2.0).extend(0.0),
            tip.extend(1.0),
        );
        self.box_transform(m, color, opts.depth_test);
    }

    fn gizmo_line(
        &mut self,
        a: Vec3,
        b: Vec3,
        color: [f32; 4],
        width: f32,
        opts: &GizmoOpts,
    ) {
        self.line(
            a,
            b,
            LineOpts {
                color_from: color,
                color_to: color,
                width_from: width,
                width_to: width,
                depth_test: opts.depth_test,
            },
        );
    }
}

fn gizmo_axis_style(axis: GizmoAxis, highlight: Option<GizmoAxis>) -> ([f32; 4], f32) {
    let base = match axis {
        GizmoAxis::X => [0.92, 0.24, 0.22, 1.0],
        GizmoAxis::Y => [0.28, 0.82, 0.32, 1.0],
        GizmoAxis::Z => [0.28, 0.48, 0.98, 1.0],
        GizmoAxis::Xy => [0.95, 0.78, 0.22, 1.0],
        GizmoAxis::Yz => [0.35, 0.88, 0.92, 1.0],
        GizmoAxis::Zx => [0.92, 0.42, 0.95, 1.0],
        GizmoAxis::Uniform => [0.92, 0.92, 0.95, 1.0],
    };
    match highlight {
        Some(h) if h == axis => (
            [
                f32::min(base[0] * 0.35 + 0.65, 1.0),
                f32::min(base[1] * 0.35 + 0.65, 1.0),
                f32::min(base[2] * 0.35 + 0.65, 1.0),
                1.0,
            ],
            3.6,
        ),
        Some(_) => (
            [base[0] * 0.45, base[1] * 0.45, base[2] * 0.45, 0.55],
            1.4,
        ),
        None => (base, 2.15),
    }
}

fn gizmo_neutral_style(dimmed: bool) -> ([f32; 4], f32) {
    if dimmed {
        ([0.55, 0.58, 0.62, 0.45], 1.2)
    } else {
        ([0.75, 0.78, 0.82, 0.85], 1.6)
    }
}
