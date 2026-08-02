use super::store::Handle;
use glam::{Quat, Vec3};

pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
}

pub struct AnimChannel {
    pub target: Handle<super::node::Node>,
    pub path: AnimPath,
    pub times: Vec<f32>,
    pub values: AnimValues,
    pub step: bool,
}

pub enum AnimPath {
    Translation,
    Rotation,
    Scale,
}

pub enum AnimValues {
    Vec3(Vec<Vec3>),
    Quat(Vec<Quat>),
}

pub struct Animator {
    pub clip: Handle<AnimationClip>,
    pub time: f32,
    pub speed: f32,
    pub playing: bool,
    pub looping: bool,
}

impl Animator {
    pub fn new(clip: Handle<AnimationClip>) -> Self {
        Self {
            clip,
            time: 0.0,
            speed: 1.0,
            playing: true,
            looping: true,
        }
    }
}

pub fn sample_vec3(times: &[f32], values: &[Vec3], t: f32, step: bool) -> Vec3 {
    if values.is_empty() {
        return Vec3::ZERO;
    }
    if times.len() < 2 || t <= times[0] {
        return values[0];
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return values[last];
    }
    let i = times.partition_point(|&x| x <= t).saturating_sub(1);
    if step || i + 1 >= values.len() {
        return values[i];
    }
    let a = (t - times[i]) / (times[i + 1] - times[i]);
    values[i].lerp(values[i + 1], a)
}

pub fn sample_quat(times: &[f32], values: &[Quat], t: f32, step: bool) -> Quat {
    if values.is_empty() {
        return Quat::IDENTITY;
    }
    if times.len() < 2 || t <= times[0] {
        return values[0];
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return values[last];
    }
    let i = times.partition_point(|&x| x <= t).saturating_sub(1);
    if step || i + 1 >= values.len() {
        return values[i];
    }
    let a = (t - times[i]) / (times[i + 1] - times[i]);
    values[i].slerp(values[i + 1], a)
}
