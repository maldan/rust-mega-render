//! Bridges the OpenXR-owned Vulkan instance/device (see [`super::session::XrContext`])
//! into a `wgpu::Device`/`Queue` via `wgpu-hal`'s raw Vulkan interop, and wraps each
//! OpenXR swapchain image as a `wgpu::Texture` so [`crate::WgpuVisualizer`] can render
//! into it directly — no extra copy.
//!
//! This is the only place in the crate that reaches into `wgpu_hal`; everything above
//! (scene, visualizer, host loop) sees plain `wgpu` types.

use super::session::{XrContext, XrError, VIEW_COUNT};
use ash::vk;
use wgpu::hal;

/// Color format used for the XR swapchain — must stay in sync with the format
/// passed to [`XrContext::create_swapchain`] by the host loop.
pub const XR_COLOR_FORMAT_WGPU: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const XR_COLOR_FORMAT_VK: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// `wgpu` handles bridged from [`XrContext`]'s Vulkan instance/device.
///
/// Must outlive every [`XrSwapchainTextures`] built from it.
pub struct XrWgpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl XrWgpu {
    /// # Safety
    /// `xr` must outlive the returned [`XrWgpu`] — we wrap its Vulkan instance/device
    /// by raw handle without taking a lifetime-checked reference. wgpu takes ownership
    /// of destroying the wrapped instance/device on drop (see `hal::vulkan::Instance::from_raw`
    /// docs), so `xr`'s own Vulkan objects must not be destroyed separately.
    pub unsafe fn from_xr(xr: &XrContext) -> Result<Self, XrError> {
        let hal_instance = unsafe {
            hal::vulkan::Instance::from_raw(
                xr.vk_entry.clone(),
                xr.vk_instance.clone(),
                xr.vk_api_version,
                0,
                None,
                Vec::new(),
                wgpu::wgt::InstanceFlags::empty(),
                wgpu::wgt::MemoryBudgetThresholds::default(),
                false,
                None,
            )
            .map_err(|e| XrError(format!("wgpu-hal: Instance::from_raw: {e}")))?
        };

        let exposed = hal_instance
            .expose_adapter(xr.vk_physical_device)
            .ok_or_else(|| XrError("wgpu-hal: expose_adapter failed".into()))?;

        let required_extensions = exposed
            .adapter
            .required_device_extensions(wgpu::wgt::Features::empty());
        // `wgt::Limits::default()` is the *minimum* WebGPU-spec limits, not what the
        // hardware actually supports — using it here (rather than the adapter's real
        // capabilities) made the desktop mesh pipeline's 5-target G-buffer exceed the
        // reported per-sample color attachment budget. Use the adapter's real limits.
        let mut limits = exposed.capabilities.limits.clone();
        // Some Vulkan drivers report `maxBufferSize`/`maxStorageBufferRange` (via
        // VkPhysicalDeviceMaintenance4/3Properties) as `u64::MAX` or another value that
        // doesn't fit `u32` — legal per the Vulkan spec, but `wgpu-core`'s indirect-draw
        // validation asserts `limits.max_buffer_size <= u32::MAX as u64` unconditionally.
        // Clamp so the device we hand back never trips that assertion.
        limits.max_buffer_size = limits.max_buffer_size.min(u32::MAX as u64);
        let open_device = unsafe {
            exposed
                .adapter
                .device_from_raw(
                    xr.vk_device.clone(),
                    None,
                    &required_extensions,
                    wgpu::wgt::Features::empty(),
                    &limits,
                    &wgpu::wgt::MemoryHints::default(),
                    xr.queue_family_index,
                    0,
                )
                .map_err(|e| XrError(format!("wgpu-hal: device_from_raw: {e}")))?
        };

        let instance = unsafe { wgpu::Instance::from_hal::<hal::api::Vulkan>(hal_instance) };
        let adapter = unsafe { instance.create_adapter_from_hal::<hal::api::Vulkan>(exposed) };
        // `create_device_from_hal` uses this descriptor's `required_limits` as the
        // device's *actual* validation limits (unlike the safe `request_device` path,
        // it does not fall back to the adapter's real capabilities) — reuse the same
        // real hardware limits queried above, or the mesh pipeline's 5-target G-buffer
        // trips the (much lower) `Limits::default()` per-sample color attachment budget.
        let device_desc = wgpu::DeviceDescriptor {
            label: Some("xr_device"),
            required_limits: limits,
            ..Default::default()
        };
        let (device, queue) = unsafe {
            adapter
                .create_device_from_hal::<hal::api::Vulkan>(open_device, &device_desc)
                .map_err(|e| XrError(format!("wgpu-hal: create_device_from_hal: {e}")))?
        };

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

/// A swapchain image wrapped as a `wgpu::Texture` (both eyes, `VIEW_COUNT` array layers)
/// plus one `TextureView` per eye for rendering.
pub struct XrSwapchainImage {
    pub texture: wgpu::Texture,
    /// Per-eye render target views (`eye_views[0]` = left, `[1]` = right).
    pub eye_views: [wgpu::TextureView; VIEW_COUNT as usize],
}

/// Wrap every image of an OpenXR swapchain as a `wgpu::Texture` exactly once — the
/// underlying `vk::Image`s are fixed for the swapchain's lifetime, only their
/// acquire/wait/release cycling changes per frame.
///
/// # Safety
/// `images` must come from the swapchain that owns them, sized `resolution` with
/// `VIEW_COUNT` array layers and [`XR_COLOR_FORMAT_VK`], and must not be manually
/// destroyed — OpenXR owns them for the swapchain's lifetime.
pub unsafe fn wrap_swapchain_images(
    xr_wgpu: &XrWgpu,
    images: &[vk::Image],
    resolution: vk::Extent2D,
) -> Vec<XrSwapchainImage> {
    let hal_device = unsafe { xr_wgpu.device.as_hal::<hal::api::Vulkan>() }
        .expect("wgpu device was not created from the Vulkan backend");

    images
        .iter()
        .map(|&image| {
            let hal_desc = hal::TextureDescriptor {
                label: Some("xr_swapchain_image"),
                size: wgpu::wgt::Extent3d {
                    width: resolution.width,
                    height: resolution.height,
                    depth_or_array_layers: VIEW_COUNT,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::wgt::TextureDimension::D2,
                format: XR_COLOR_FORMAT_WGPU,
                usage: wgpu::wgt::TextureUses::COLOR_TARGET
                    | wgpu::wgt::TextureUses::RESOURCE
                    | wgpu::wgt::TextureUses::COPY_DST,
                memory_flags: hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            let hal_texture = unsafe {
                hal_device.texture_from_raw(image, &hal_desc, None, hal::vulkan::TextureMemory::External)
            };
            let wgpu_desc = wgpu::TextureDescriptor {
                label: Some("xr_swapchain_image"),
                size: wgpu::Extent3d {
                    width: resolution.width,
                    height: resolution.height,
                    depth_or_array_layers: VIEW_COUNT,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: XR_COLOR_FORMAT_WGPU,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            let texture = unsafe {
                xr_wgpu.device.create_texture_from_hal::<hal::api::Vulkan>(
                    hal_texture,
                    &wgpu_desc,
                    wgpu::wgt::TextureUses::UNINITIALIZED,
                )
            };
            let eye_view = |eye: u32| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("xr_eye_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: eye,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            };
            XrSwapchainImage {
                eye_views: [eye_view(0), eye_view(1)],
                texture,
            }
        })
        .collect()
}
