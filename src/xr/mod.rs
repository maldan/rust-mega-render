//! Optional OpenXR/VR support (feature = "xr").
//!
//! Stage 1: a minimal, standalone OpenXR + Vulkan session bootstrap. No wgpu
//! interop yet, no scene rendering — just enough to open a session, acquire
//! per-eye swapchain images, and prove the OpenXR runtime round-trip works.
//! See `examples/xr_scene.rs` for the frame loop.
//!
//! The XR session always uses its own dedicated Vulkan instance/device (a
//! requirement of the OpenXR<->graphics-API interop, see `XR_KHR_vulkan_enable2`).
//! This is completely independent from the desktop `WgpuVisualizer` path, which
//! keeps using whatever wgpu backend it already uses.

mod actions;
mod loader;
mod session;
mod wgpu_interop;

pub use actions::{Hand, HandPose, XrActions};
pub use session::{XrContext, XrError, XrEyeViews, XrFrame, XrPollResult, XrSwapchain, VIEW_COUNT};
pub use wgpu_interop::{
    wrap_swapchain_images, XrSwapchainImage, XrWgpu, XR_COLOR_FORMAT_VK, XR_COLOR_FORMAT_WGPU,
};
