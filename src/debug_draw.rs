use glam::{Mat4, Vec3};
use std::f32::consts::TAU;

#[derive(Clone, Copy)]
pub struct DebugLine {
    pub start: Vec3,
    pub end: Vec3,
    pub color: [f32; 4],
    pub depth_test: bool,
}

#[derive(Clone, Copy)]
pub struct DebugPoint {
    pub position: Vec3,
    pub color: [f32; 4],
    /// Size in pixels (screen-space square).
    pub size: f32,
    pub depth_test: bool,
}

#[derive(Default)]
pub struct DebugDraw {
    pub lines: Vec<DebugLine>,
    pub points: Vec<DebugPoint>,
}

impl DebugDraw {
    pub fn clear(&mut self) {
        self.lines.clear();
        self.points.clear();
    }

    pub fn line(&mut self, start: Vec3, end: Vec3, color: [f32; 4]) {
        self.line_ex(start, end, color, true);
    }

    pub fn line_overlay(&mut self, start: Vec3, end: Vec3, color: [f32; 4]) {
        self.line_ex(start, end, color, false);
    }

    pub fn line_ex(&mut self, start: Vec3, end: Vec3, color: [f32; 4], depth_test: bool) {
        self.lines.push(DebugLine {
            start,
            end,
            color,
            depth_test,
        });
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
        self.line(origin, origin + Vec3::X * len, [1.0, 0.2, 0.2, 1.0]);
        self.line(origin, origin + Vec3::Y * len, [0.2, 1.0, 0.2, 1.0]);
        self.line(origin, origin + Vec3::Z * len, [0.3, 0.5, 1.0, 1.0]);
    }

    pub fn axes_overlay(&mut self, origin: Vec3, len: f32) {
        self.line_overlay(origin, origin + Vec3::X * len, [1.0, 0.2, 0.2, 1.0]);
        self.line_overlay(origin, origin + Vec3::Y * len, [0.2, 1.0, 0.2, 1.0]);
        self.line_overlay(origin, origin + Vec3::Z * len, [0.3, 0.5, 1.0, 1.0]);
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
            self.line_ex(p[a], p[b], color, depth_test);
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
                self.line_ex(
                    m.transform_point3(p0),
                    m.transform_point3(p1),
                    color,
                    depth_test,
                );
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

    /// Classic diamond bone: parent tip → mid cross → child tip.
    pub fn bone(&mut self, from: Vec3, to: Vec3, color: [f32; 4], depth_test: bool) {
        let dir = to - from;
        let len = dir.length();
        if len < 1e-6 {
            return;
        }
        let axis = dir / len;
        let up = if axis.cross(Vec3::Y).length_squared() < 1e-4 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let side = axis.cross(up).normalize();
        let up = side.cross(axis).normalize();
        let mid = from + axis * (len * 0.2);
        let r = len * 0.1;
        let c = [
            mid + side * r,
            mid + up * r,
            mid - side * r,
            mid - up * r,
        ];
        for i in 0..4 {
            self.line_ex(from, c[i], color, depth_test);
            self.line_ex(c[i], to, color, depth_test);
            self.line_ex(c[i], c[(i + 1) % 4], color, depth_test);
        }
    }
}
