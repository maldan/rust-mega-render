//! OpenXR + Vulkan session bootstrap (stage 1: no wgpu interop yet).
//!
//! Follows the interop pattern required by `XR_KHR_vulkan_enable2`: the XR
//! runtime must create (or approve) the Vulkan instance/device used for the
//! session, so we drive `ash` directly here instead of going through wgpu.
//! Stage 2 will wrap the resulting swapchain images as `wgpu::Texture`s via
//! `wgpu-hal` so the existing [`crate::Visualizer`] can render into them.

use ash::vk::{self, Handle};
use openxr as xr;

/// OpenXR always gives us exactly 2 views for `PRIMARY_STEREO`.
pub const VIEW_COUNT: u32 = 2;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const FORM_FACTOR: xr::FormFactor = xr::FormFactor::HEAD_MOUNTED_DISPLAY;

#[derive(Debug)]
pub struct XrError(pub String);

impl std::fmt::Display for XrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for XrError {}

impl From<xr::sys::Result> for XrError {
    fn from(e: xr::sys::Result) -> Self {
        XrError(format!("OpenXR error: {e}"))
    }
}

impl From<vk::Result> for XrError {
    fn from(e: vk::Result) -> Self {
        XrError(format!("Vulkan error: {e}"))
    }
}

/// Per-eye pose + fov reported by OpenXR for the current frame.
pub type XrEyeViews = Vec<xr::View>;

/// Result of [`XrContext::poll_events`].
pub enum XrPollResult {
    /// Keep running; check [`XrContext::session_running`] before rendering.
    Continue,
    /// Runtime asked us to exit (headset disconnected, instance loss, etc).
    Exit,
}

/// A frame acquired via [`XrContext::wait_frame`]; must be passed to
/// [`XrContext::end_frame`] once rendering is recorded.
pub struct XrFrame {
    pub state: xr::FrameState,
}

/// A swapchain sized for both eyes (texture array, [`VIEW_COUNT`] layers).
pub struct XrSwapchain {
    pub handle: xr::Swapchain<xr::Vulkan>,
    pub images: Vec<vk::Image>,
    pub resolution: vk::Extent2D,
}

/// Owns the OpenXR instance/session plus the dedicated Vulkan instance/device
/// created for XR interop. Raw Vulkan handles are exposed as `pub` fields so
/// callers (and, later, the wgpu-hal interop layer) can record their own
/// command buffers against the same device.
pub struct XrContext {
    pub xr_instance: xr::Instance,
    pub system: xr::SystemId,
    pub vk_entry: ash::Entry,
    pub vk_instance: ash::Instance,
    /// Vulkan API version the instance/device were created with (see
    /// [`crate::xr::wgpu_interop`], which needs it to build the matching
    /// `wgpu-hal` instance wrapper).
    pub vk_api_version: u32,
    pub vk_physical_device: vk::PhysicalDevice,
    pub vk_device: ash::Device,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    session: xr::Session<xr::Vulkan>,
    frame_wait: xr::FrameWaiter,
    frame_stream: xr::FrameStream<xr::Vulkan>,
    stage: xr::Space,
    blend_mode: xr::EnvironmentBlendMode,
    event_storage: xr::EventDataBuffer,
    session_running: bool,
}

impl XrContext {
    pub fn new(app_name: &str) -> Result<Self, XrError> {
        // SAFETY: the probed library (default name, OPENXR_LOADER, or a
        // runtime install such as SteamVR) must be a spec-compliant OpenXR
        // loader. We load it in-place — no copy next to the exe.
        let entry = unsafe { super::loader::load_entry()? };

        let available = entry.enumerate_extensions()?;
        if !available.khr_vulkan_enable2 {
            return Err(XrError(
                "OpenXR runtime does not support XR_KHR_vulkan_enable2".into(),
            ));
        }
        let mut enabled = xr::ExtensionSet::default();
        enabled.khr_vulkan_enable2 = true;

        let xr_instance = entry.create_instance(
            &xr::ApplicationInfo {
                application_name: app_name,
                application_version: 0,
                engine_name: "mega-render",
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            &enabled,
            &[],
        )?;

        let props = xr_instance.properties()?;
        eprintln!(
            "xr: runtime = {} {}",
            props.runtime_name, props.runtime_version
        );

        let system = xr_instance.system(FORM_FACTOR)?;
        let blend_mode = xr_instance
            .enumerate_environment_blend_modes(system, VIEW_TYPE)?
            .into_iter()
            .next()
            .ok_or_else(|| XrError("no environment blend mode available".into()))?;

        let vk_target_version = vk::make_api_version(0, 1, 1, 0);
        let vk_target_version_xr = xr::Version::new(1, 1, 0);
        let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(system)?;
        if vk_target_version_xr < reqs.min_api_version_supported
            || vk_target_version_xr.major() > reqs.max_api_version_supported.major()
        {
            return Err(XrError(format!(
                "OpenXR runtime requires Vulkan > {}, < {}.0.0",
                reqs.min_api_version_supported,
                reqs.max_api_version_supported.major() + 1
            )));
        }

        // SAFETY: mandated interop pattern for XR_KHR_vulkan_enable2 — the XR
        // runtime creates/validates the Vulkan instance and device for us.
        let (vk_entry, vk_instance, vk_physical_device, vk_device, queue_family_index, queue) = unsafe {
            let vk_entry =
                ash::Entry::load().map_err(|e| XrError(format!("failed to load Vulkan: {e}")))?;

            let vk_app_info = vk::ApplicationInfo::default()
                .application_version(0)
                .engine_version(0)
                .api_version(vk_target_version);

            let vk_instance_handle = xr_instance
                .create_vulkan_instance(
                    system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    &vk::InstanceCreateInfo::default().application_info(&vk_app_info) as *const _
                        as *const _,
                )?
                .map_err(vk::Result::from_raw)?;
            let vk_instance = ash::Instance::load(
                vk_entry.static_fn(),
                vk::Instance::from_raw(vk_instance_handle as _),
            );

            let vk_physical_device = vk::PhysicalDevice::from_raw(
                xr_instance.vulkan_graphics_device(system, vk_instance.handle().as_raw() as _)? as _,
            );

            let props = vk_instance.get_physical_device_properties(vk_physical_device);
            if props.api_version < vk_target_version {
                return Err(XrError(
                    "Vulkan physical device does not support version 1.1 (required for multiview)"
                        .into(),
                ));
            }

            let queue_family_index = vk_instance
                .get_physical_device_queue_family_properties(vk_physical_device)
                .into_iter()
                .enumerate()
                .find_map(|(i, info)| {
                    info.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        .then_some(i as u32)
                })
                .ok_or_else(|| XrError("Vulkan device has no graphics queue".into()))?;

            let vk_device_handle = xr_instance
                .create_vulkan_device(
                    system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    vk_physical_device.as_raw() as _,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(queue_family_index)
                            .queue_priorities(&[1.0])])
                        .push_next(&mut vk::PhysicalDeviceMultiviewFeatures {
                            multiview: vk::TRUE,
                            ..Default::default()
                        }) as *const _ as *const _,
                )?
                .map_err(vk::Result::from_raw)?;
            let vk_device = ash::Device::load(
                vk_instance.fp_v1_0(),
                vk::Device::from_raw(vk_device_handle as _),
            );

            let queue = vk_device.get_device_queue(queue_family_index, 0);

            (
                vk_entry,
                vk_instance,
                vk_physical_device,
                vk_device,
                queue_family_index,
                queue,
            )
        };

        // SAFETY: instance/device/queue above were created specifically for
        // (and approved by) this OpenXR system, as required by the session.
        let (session, frame_wait, frame_stream) = unsafe {
            xr_instance.create_session::<xr::Vulkan>(
                system,
                &xr::vulkan::SessionCreateInfo {
                    instance: vk_instance.handle().as_raw() as _,
                    physical_device: vk_physical_device.as_raw() as _,
                    device: vk_device.handle().as_raw() as _,
                    queue_family_index,
                    queue_index: 0,
                },
            )?
        };

        let stage =
            session.create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)?;

        Ok(Self {
            xr_instance,
            system,
            vk_entry,
            vk_instance,
            vk_api_version: vk_target_version,
            vk_physical_device,
            vk_device,
            queue_family_index,
            queue,
            session,
            frame_wait,
            frame_stream,
            stage,
            blend_mode,
            event_storage: xr::EventDataBuffer::new(),
            session_running: false,
        })
    }

    pub fn session_running(&self) -> bool {
        self.session_running
    }

    /// Reference space views/actions are located against (`STAGE`, i.e. room-scale origin).
    pub fn stage(&self) -> &xr::Space {
        &self.stage
    }

    pub fn session(&self) -> &xr::Session<xr::Vulkan> {
        &self.session
    }

    pub fn attach_action_sets(&self, sets: &[&xr::ActionSet]) -> Result<(), XrError> {
        self.session.attach_action_sets(sets)?;
        Ok(())
    }

    pub fn sync_actions(&self, sets: &[xr::ActiveActionSet<'_>]) -> Result<(), XrError> {
        self.session.sync_actions(sets)?;
        Ok(())
    }

    /// Recommended per-eye resolution reported by the runtime (both eyes use
    /// the same resolution — required for the multiview / texture-array
    /// swapchain layout).
    pub fn recommended_resolution(&self) -> Result<vk::Extent2D, XrError> {
        let views = self
            .xr_instance
            .enumerate_view_configuration_views(self.system, VIEW_TYPE)?;
        Ok(vk::Extent2D {
            width: views[0].recommended_image_rect_width,
            height: views[0].recommended_image_rect_height,
        })
    }

    pub fn create_swapchain(
        &self,
        resolution: vk::Extent2D,
        format: vk::Format,
    ) -> Result<XrSwapchain, XrError> {
        let handle = self.session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED
                | xr::SwapchainUsageFlags::TRANSFER_DST,
            format: format.as_raw() as _,
            sample_count: 1,
            width: resolution.width,
            height: resolution.height,
            face_count: 1,
            array_size: VIEW_COUNT,
            mip_count: 1,
        })?;
        let images = handle
            .enumerate_images()?
            .into_iter()
            .map(vk::Image::from_raw)
            .collect();
        Ok(XrSwapchain {
            handle,
            images,
            resolution,
        })
    }

    /// Poll the OpenXR event queue; drives session begin/end transitions.
    /// Call this once per host-loop tick, before [`Self::wait_frame`].
    pub fn poll_events(&mut self) -> XrPollResult {
        loop {
            let event = match self.xr_instance.poll_event(&mut self.event_storage) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("xr: poll_event error: {e}");
                    return XrPollResult::Exit;
                }
            };
            let Some(event) = event else {
                return XrPollResult::Continue;
            };
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => match e.state() {
                    xr::SessionState::READY => {
                        if self.session.begin(VIEW_TYPE).is_err() {
                            return XrPollResult::Exit;
                        }
                        self.session_running = true;
                    }
                    xr::SessionState::STOPPING => {
                        let _ = self.session.end();
                        self.session_running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                        return XrPollResult::Exit;
                    }
                    _ => {}
                },
                InstanceLossPending(_) => return XrPollResult::Exit,
                EventsLost(e) => eprintln!("xr: lost {} events", e.lost_event_count()),
                _ => {}
            }
        }
    }

    /// Ask the runtime to end the session gracefully (e.g. on Ctrl-C). The
    /// runtime will report `EXITING` via [`Self::poll_events`] once it's safe
    /// to tear down.
    pub fn request_exit(&self) {
        match self.session.request_exit() {
            Ok(()) | Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => {}
            Err(e) => eprintln!("xr: request_exit error: {e}"),
        }
    }

    /// Block until the runtime wants the next frame, and begin it. Returns
    /// `None` when the runtime asked us to skip rendering this frame (an
    /// empty frame is submitted internally in that case).
    pub fn wait_frame(&mut self) -> Result<Option<XrFrame>, XrError> {
        let state = self.frame_wait.wait()?;
        self.frame_stream.begin()?;
        if !state.should_render {
            self.frame_stream
                .end(state.predicted_display_time, self.blend_mode, &[])?;
            return Ok(None);
        }
        Ok(Some(XrFrame { state }))
    }

    /// Head + eye poses/fovs predicted for `frame`, in the session's stage space.
    pub fn locate_views(&self, frame: &XrFrame) -> Result<XrEyeViews, XrError> {
        let (_flags, views) =
            self.session
                .locate_views(VIEW_TYPE, frame.state.predicted_display_time, &self.stage)?;
        Ok(views)
    }

    /// Submit the rendered swapchain image(s) for this frame.
    pub fn end_frame(
        &mut self,
        frame: XrFrame,
        views: &[xr::View],
        swapchain: &XrSwapchain,
    ) -> Result<(), XrError> {
        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: swapchain.resolution.width as _,
                height: swapchain.resolution.height as _,
            },
        };
        let proj_views: Vec<_> = (0..VIEW_COUNT as usize)
            .map(|i| {
                xr::CompositionLayerProjectionView::new()
                    .pose(views[i].pose)
                    .fov(views[i].fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&swapchain.handle)
                            .image_array_index(i as u32)
                            .image_rect(rect),
                    )
            })
            .collect();
        let layer = xr::CompositionLayerProjection::new()
            .space(&self.stage)
            .views(&proj_views);
        self.frame_stream
            .end(frame.state.predicted_display_time, self.blend_mode, &[&layer])?;
        Ok(())
    }
}

// NOTE: `ash::Instance`/`ash::Device` do not destroy the underlying Vulkan
// objects on drop (that's the caller's job via `destroy_device`/
// `destroy_instance`). Stage 1 is a short-lived example process, so we
// intentionally skip explicit teardown here — the OS reclaims everything on
// exit. Once `XrContext` is embedded in a long-running host (stage 2+),
// add an explicit `shutdown(self)` that drops the OpenXR session/space
// objects *before* calling `vk_device.destroy_device` / `vk_instance.destroy_instance`,
// per the Vulkan/OpenXR teardown ordering requirement.
