//! Viewport corner orientation gizmo (Unity-style axis triad).
//!
//! Draw + hit-test live here; the host applies snaps to its own orbit/camera controller.

use glam::{Vec2, Vec3, Vec4};

use crate::camera::Camera;
use crate::hud::{Hud, Rect as HudRect};

/// Which view to snap to (camera sits on this world axis, looking at target).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewAxis {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl ViewAxis {
    pub fn dir(self) -> Vec3 {
        match self {
            Self::PosX => Vec3::X,
            Self::NegX => -Vec3::X,
            Self::PosY => Vec3::Y,
            Self::NegY => -Vec3::Y,
            Self::PosZ => Vec3::Z,
            Self::NegZ => -Vec3::Z,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PosX => "+X",
            Self::NegX => "-X",
            Self::PosY => "+Y",
            Self::NegY => "-Y",
            Self::PosZ => "+Z",
            Self::NegZ => "-Z",
        }
    }

    fn color(self) -> [f32; 4] {
        match self {
            Self::PosX | Self::NegX => [0.90, 0.28, 0.28, 1.0],
            Self::PosY | Self::NegY => [0.35, 0.82, 0.38, 1.0],
            Self::PosZ | Self::NegZ => [0.32, 0.55, 0.95, 1.0],
        }
    }

    fn letter(self) -> Option<&'static str> {
        match self {
            Self::PosX => Some("X"),
            Self::PosY => Some("Y"),
            Self::PosZ => Some("Z"),
            _ => None,
        }
    }

    fn is_positive(self) -> bool {
        matches!(self, Self::PosX | Self::PosY | Self::PosZ)
    }

    fn all() -> [ViewAxis; 6] {
        [
            Self::PosX,
            Self::NegX,
            Self::PosY,
            Self::NegY,
            Self::PosZ,
            Self::NegZ,
        ]
    }
}

#[derive(Clone, Copy)]
struct Tip {
    axis: ViewAxis,
    /// Screen position relative to widget center.
    offset: Vec2,
    /// View-space depth (larger = farther; draw first).
    depth: f32,
}

const WIDGET_SIZE: f32 = 120.0;
const MARGIN: f32 = 12.0;
const AXIS_LEN: f32 = 40.0;
const TIP_R: f32 = 12.0;
const TIP_R_NEG: f32 = 8.0;

/// Top-right corner rect of the view gizmo in viewport-local pixels.
pub fn widget_rect(viewport_size: Vec2) -> HudRect {
    let s = WIDGET_SIZE;
    HudRect::from_min_size(
        Vec2::new(viewport_size.x - MARGIN - s, MARGIN),
        Vec2::splat(s),
    )
}

fn tips(camera: &Camera) -> [Tip; 6] {
    // Same LH view as the renderer — axes match what you see in the viewport.
    let view = glam::camera::lh::view::look_at_mat4(camera.eye, camera.target, camera.up);

    ViewAxis::all().map(|axis| {
        let d = axis.dir();
        let local = (view * Vec4::new(d.x, d.y, d.z, 0.0)).truncate();
        Tip {
            axis,
            // HUD Y grows downward; view Y is up.
            offset: Vec2::new(local.x, -local.y) * AXIS_LEN,
            depth: local.z,
        }
    })
}

/// Draw into `hud` (viewport pixel space). Call after [`Hud::begin`].
///
/// Stems are stamped as quads (not [`Hud::line`]): the HUD pass draws all lines
/// *after* all quads, so lines would always sit on top of tips.
pub fn draw(hud: &mut Hud, camera: &Camera, viewport_size: Vec2, cursor_local: Vec2) {
    let rect = widget_rect(viewport_size);
    let center = Vec2::new(
        (rect.min.x + rect.max.x) * 0.5,
        (rect.min.y + rect.max.y) * 0.5,
    );

    let mut tips = tips(camera);
    // Far → near so front tips cover stems behind them.
    tips.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));

    let hover = hit_test(camera, viewport_size, cursor_local);

    for tip in &tips {
        let pos = center + tip.offset;
        let positive = tip.axis.is_positive();
        let r = if positive { TIP_R } else { TIP_R_NEG };
        let col = tip.axis.color();
        let stem_col = [col[0], col[1], col[2], 0.85];

        // Stop stem before the tip so it doesn't poke through the handle.
        let delta = tip.offset;
        let len = delta.length();
        if len > r + 1.0 {
            let end = center + delta * ((len - r) / len);
            draw_stem(hud, center, end, stem_col, 2.0);
        }

        draw_tip(
            hud,
            pos,
            r,
            col,
            tip.axis.letter(),
            positive,
            hover == Some(tip.axis),
        );
    }
}

/// Axis-aligned stamp chain approximating a thick line (HUD has no rotated quads).
fn draw_stem(hud: &mut Hud, from: Vec2, to: Vec2, color: [f32; 4], thickness: f32) {
    let delta = to - from;
    let len = delta.length();
    if len < 0.5 {
        return;
    }
    let dir = delta / len;
    let half = thickness * 0.5;
    let steps = ((len / 1.5).ceil() as i32).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = from + dir * (len * t);
        hud.fill(
            HudRect {
                min: p - Vec2::splat(half),
                max: p + Vec2::splat(half),
            },
            color,
        );
    }
}

fn draw_tip(
    hud: &mut Hud,
    center: Vec2,
    r: f32,
    color: [f32; 4],
    letter: Option<&str>,
    filled: bool,
    hovered: bool,
) {
    let rect = HudRect {
        min: center - Vec2::splat(r),
        max: center + Vec2::splat(r),
    };
    if filled {
        let bg = if hovered {
            [
                (color[0] + 0.25).min(1.0),
                (color[1] + 0.25).min(1.0),
                (color[2] + 0.25).min(1.0),
                1.0,
            ]
        } else {
            color
        };
        hud.fill(rect, bg);
        if let Some(letter) = letter {
            let scale = 2.0;
            let tw = hud.text_width(letter, scale);
            let th = 8.0 * scale;
            hud.text(
                Vec2::new(center.x - tw * 0.5, center.y - th * 0.5),
                letter,
                [1.0, 1.0, 1.0, 1.0],
                scale,
            );
        }
    } else {
        let alpha = if hovered { 0.95 } else { 0.55 };
        let ring = [color[0], color[1], color[2], alpha];
        let outer = HudRect {
            min: center - Vec2::splat(r),
            max: center + Vec2::splat(r),
        };
        let inner = outer.inset(2.0);
        hud.fill(outer, ring);
        // Punch hole (matches typical black viewport clear).
        hud.fill(inner, [0.0, 0.0, 0.0, 1.0]);
    }
}

/// Pick axis under cursor (viewport-local pixels). Prefers front-most tip.
pub fn hit_test(camera: &Camera, viewport_size: Vec2, cursor_local: Vec2) -> Option<ViewAxis> {
    let rect = widget_rect(viewport_size);
    if !rect.contains(cursor_local) {
        return None;
    }
    let center = Vec2::new(
        (rect.min.x + rect.max.x) * 0.5,
        (rect.min.y + rect.max.y) * 0.5,
    );
    let mut tips = tips(camera);
    tips.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));

    let mut best: Option<(f32, ViewAxis)> = None;
    for tip in tips {
        let pos = center + tip.offset;
        let r = if tip.axis.is_positive() {
            TIP_R + 3.0
        } else {
            TIP_R_NEG + 3.0
        };
        let d = (cursor_local - pos).length();
        if d <= r {
            let score = d - tip.depth * 0.01;
            if best.is_none_or(|(bd, _)| score < bd) {
                best = Some((score, tip.axis));
            }
        }
    }
    best.map(|(_, a)| a)
}

/// True if cursor is over the widget plate (block orbit / scene tools).
pub fn contains_cursor(viewport_size: Vec2, cursor_local: Vec2) -> bool {
    widget_rect(viewport_size).contains(cursor_local)
}
