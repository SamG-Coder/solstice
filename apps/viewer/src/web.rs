//! The browser front end: a small API for `web/index.html` to drive.
//!
//! JavaScript owns the page: the animation loop, input and UI. Rust owns the renderer.

use crate::{default_camera, default_environment, CameraInput, FlyCamera};
use glam::Vec3;
use solstice_render::{DebugView, Environment, Gpu, Renderer, RendererSettings};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

fn error_string(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Sets up logging and panic messages. Runs once when the module loads.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

/// A running viewer, bound to one canvas.
#[wasm_bindgen]
pub struct WebViewer {
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    camera: FlyCamera,
    environment: Environment,
    gpu_errors: Arc<AtomicU32>,
}

#[wasm_bindgen]
impl WebViewer {
    /// Creates a viewer from the bytes of a scene file. Rejects if WebGPU is unavailable or
    /// the GPU lacks required capabilities, with a message suitable for showing to the user.
    pub async fn create(
        canvas: HtmlCanvasElement,
        scene_bytes: Vec<u8>,
    ) -> Result<WebViewer, JsValue> {
        let scene = solstice_asset::file::decode(&scene_bytes).map_err(error_string)?;
        drop(scene_bytes);

        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(error_string)?;
        let gpu = Gpu::request(&instance, Some(&surface))
            .await
            .map_err(error_string)?;

        let gpu_errors = Arc::new(AtomicU32::new(0));
        let counter = gpu_errors.clone();
        gpu.device
            .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                counter.fetch_add(1, Ordering::Relaxed);
                log::error!("GPU error: {error}");
            }));

        let mut config = surface
            .get_default_config(&gpu.adapter, width, height)
            .ok_or_else(|| JsValue::from_str("this canvas cannot be presented to"))?;
        config.format = config.format.remove_srgb_suffix();
        surface.configure(&gpu.device, &config);

        let renderer = Renderer::new(&gpu, &scene, config.format, RendererSettings::default());
        Ok(WebViewer {
            gpu,
            surface,
            config,
            renderer,
            camera: FlyCamera::new(default_camera()),
            environment: default_environment(),
            gpu_errors,
        })
    }

    /// Updates the canvas size. Pass physical pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.gpu.device, &self.config);
        }
    }

    /// Advances the camera and draws one frame.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        delta_seconds: f32,
        forward: f32,
        right: f32,
        up: f32,
        look_yaw: f32,
        look_pitch: f32,
        fast: bool,
    ) {
        self.camera.update(
            &CameraInput {
                forward,
                right,
                up,
                look_yaw,
                look_pitch,
                fast,
            },
            delta_seconds,
        );
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = texture.texture.create_view(&Default::default());
        self.renderer.render(
            &self.gpu,
            &self.camera.camera,
            &self.environment,
            &view,
            (self.config.width, self.config.height),
            delta_seconds,
        );
        self.gpu.queue.present(texture);
    }

    /// Selects a debug view by name. Returns false if there is no such view.
    pub fn set_debug_view(&mut self, name: &str) -> bool {
        match DebugView::from_name(name) {
            Some(view) => {
                self.renderer.settings_mut().debug_view = view;
                true
            }
            None => false,
        }
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        self.renderer.settings_mut().render_scale = scale.clamp(0.25, 2.0);
    }

    pub fn set_lod_threshold(&mut self, pixels: f32) {
        self.renderer.settings_mut().lod_threshold = pixels.clamp(0.25, 16.0);
    }

    /// Places the sun by compass direction and height above the horizon, in radians.
    pub fn set_sun(&mut self, azimuth: f32, elevation: f32) {
        self.environment.sun_direction = Vec3::new(
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
        );
    }

    pub fn set_camera(&mut self, x: f32, y: f32, z: f32, yaw: f32, pitch: f32) {
        self.camera.camera.position = Vec3::new(x, y, z);
        self.camera.camera.yaw = yaw;
        self.camera.camera.pitch = pitch;
    }

    pub fn triangles(&self) -> u32 {
        self.renderer.stats().triangles
    }

    pub fn clusters(&self) -> u32 {
        self.renderer.stats().clusters
    }

    pub fn software_clusters(&self) -> u32 {
        self.renderer.stats().software_clusters
    }

    pub fn instances(&self) -> u32 {
        self.renderer.stats().instances
    }

    /// Triangles if every instance rendered its finest LOD. A double, since it can exceed 32 bits.
    pub fn source_triangles(&self) -> f64 {
        self.renderer.stats().source_triangles as f64
    }

    pub fn gpu_error_count(&self) -> u32 {
        self.gpu_errors.load(Ordering::Relaxed)
    }
}
