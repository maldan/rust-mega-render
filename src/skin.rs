use super::node::Node;
use super::store::Handle;
use glam::Mat4;

pub struct Skin {
    pub joints: Vec<Handle<Node>>,
    pub inverse_bind: Vec<Mat4>,
}
