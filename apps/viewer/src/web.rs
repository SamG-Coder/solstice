//! The browser front end: a small API for `web/index.html` to drive.
//!
//! JavaScript owns the page: the animation loop, input and UI. Rust owns the renderer.
//!
//! Built with the `gameplay` feature, it can also play a scene's game: the page passes on keys,
//! the mouse, gamepads and touch, and shows what scripts log. The plain build has none of that,
//! and only says whether a scene has a game to play, so the page can offer the other build.

use crate::{initial_camera, CameraInput, DayClock, FlyCamera, DEFAULT_DAY_SECONDS};
use glam::Vec3;
use solstice_asset::{file, GameFile};
use solstice_render::{
    DebugView, Effect, Gpu, Observation, Quality, Renderer, RendererSettings, StreamingBudget,
    TimeOfDay, Weather,
};
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
    /// The time of day, which places the sun and moon.
    clock: DayClock,
    gpu_errors: Arc<AtomicU32>,
    /// The scene's gameplay files, empty if it has none.
    gameplay: Vec<GameFile>,
    #[cfg(feature = "gameplay")]
    play: Option<crate::play::PlayMode>,
    /// What scripts logged, and errors, for the page to take.
    #[cfg(feature = "gameplay")]
    messages: Vec<crate::play::Message>,
}

#[wasm_bindgen]
impl WebViewer {
    /// Creates a viewer from an unpacked manifest of a scene, which draws what it has while
    /// `requests` asks for more. `data_offset` is where the file's data area begins. Budgets
    /// are in bytes of GPU memory. Rejects if WebGPU is unavailable or the GPU lacks required
    /// capabilities, with a message suitable for showing to the user. `unpack_textures` acts as
    /// if the GPU had no block compression, to test that path.
    pub async fn create(
        canvas: HtmlCanvasElement,
        manifest: Vec<u8>,
        data_offset: f64,
        geometry_budget: f64,
        texture_budget: f64,
        unpack_textures: bool,
    ) -> Result<WebViewer, JsValue> {
        let scene = file::decode_manifest(&manifest, data_offset as u64).map_err(error_string)?;
        drop(manifest);

        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(error_string)?;
        let mut gpu = Gpu::request(
            &instance,
            Some(&surface),
            wgpu::PowerPreference::HighPerformance,
        )
        .await
        .map_err(error_string)?;
        gpu.block_compressed_textures &= !unpack_textures;

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

        let budget = StreamingBudget {
            geometry_bytes: geometry_budget as u64,
            texture_bytes: texture_budget as u64,
        };
        let renderer = Renderer::streaming(
            &gpu,
            &scene,
            config.format,
            RendererSettings::default(),
            budget,
        );
        Ok(WebViewer {
            gpu,
            surface,
            config,
            renderer,
            camera: FlyCamera::new(initial_camera(scene.viewpoint)),
            clock: DayClock::new(TimeOfDay::default(), DEFAULT_DAY_SECONDS),
            gpu_errors,
            gameplay: scene.gameplay,
            #[cfg(feature = "gameplay")]
            play: None,
            #[cfg(feature = "gameplay")]
            messages: Vec::new(),
        })
    }

    /// What to fetch next, as a flat array with four numbers per request: its id, the first
    /// byte in the file, one past the last byte, and 1 if it must be inflated. Each request
    /// must later go to `deliver` or `request_failed`.
    pub fn stream_requests(&mut self, max_fetching: u32) -> Vec<f64> {
        self.renderer
            .stream_requests(max_fetching)
            .iter()
            .flat_map(|request| {
                [
                    f64::from(request.id),
                    request.start as f64,
                    request.end as f64,
                    f64::from(u8::from(request.deflated)),
                ]
            })
            .collect()
    }

    /// Takes the bytes of a request, inflated if it needed that.
    pub fn deliver(&mut self, id: u32, bytes: Vec<u8>) -> bool {
        self.renderer.deliver(id, bytes)
    }

    /// Reports that a request could not be fetched.
    pub fn request_failed(&mut self, id: u32) {
        self.renderer.request_failed(id);
    }

    /// Geometry pages on the GPU.
    pub fn pages_installed(&self) -> u32 {
        self.renderer.stats().streaming.installed
    }

    pub fn pages_total(&self) -> u32 {
        self.renderer.stats().streaming.pages
    }

    /// Size of the page pool, in pages.
    pub fn page_slots(&self) -> u32 {
        self.renderer.stats().streaming.slots
    }

    pub fn pages_evicted(&self) -> f64 {
        self.renderer.stats().streaming.evictions as f64
    }

    /// Wanted pages with no room in the pool.
    pub fn pages_starved(&self) -> u32 {
        self.renderer.stats().streaming.starved
    }

    /// Layers sharpened beyond the base tier, as middle and top counts.
    pub fn texture_layers_sharp(&self) -> Vec<u32> {
        let streaming = self.renderer.stats().streaming;
        vec![streaming.middle_layers, streaming.top_layers]
    }

    /// Bytes of GPU memory for this scene: records, page pool, then textures.
    pub fn scene_memory(&self) -> Vec<f64> {
        let memory = self.renderer.scene_bytes();
        vec![
            memory.records as f64,
            memory.pages as f64,
            memory.textures as f64,
        ]
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
        if !self.playing_as_someone() {
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
        }
        #[cfg(feature = "gameplay")]
        if let Some(play) = &mut self.play {
            let messages = play.frame(&mut self.renderer, &mut self.camera.camera, delta_seconds);
            self.messages.extend(messages);
        }
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
        let environment = self.clock.tick(delta_seconds);
        self.renderer.render(
            &self.gpu,
            &self.camera.camera,
            &environment,
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

    /// Applies a quality preset by name. Returns false if there is no such preset.
    pub fn set_quality(&mut self, name: &str) -> bool {
        match Quality::from_name(name) {
            Some(quality) => {
                self.renderer.settings_mut().apply_quality(quality);
                true
            }
            None => false,
        }
    }

    /// Sets the time of day, in hours from 0 to 24: noon is 12.
    pub fn set_time(&mut self, hours: f32) {
        self.clock.time.hours = hours.rem_euclid(24.0);
    }

    /// The time of day now, in hours, for the page to show as it passes.
    pub fn time_of_day(&self) -> f32 {
        self.clock.time.hours
    }

    /// How many real seconds a whole day takes to pass, or 0 to hold the time still.
    pub fn set_day_length(&mut self, seconds: f32) {
        self.clock.day_seconds = seconds.max(0.0);
    }

    /// Turns the scene to face north at `degrees` clockwise from its -Z axis, which moves the
    /// sun, moon and stars around it together.
    pub fn set_north(&mut self, degrees: f32) {
        self.clock.place.north = -degrees.to_radians();
    }

    /// Names of the built-in weather, separated by commas.
    pub fn weather_names() -> String {
        Weather::names().collect::<Vec<_>>().join(",")
    }

    /// Lets the weather change by itself, starting from the named weather, or stops it.
    pub fn set_weather_changing(&mut self, changing: bool, start: &str) {
        self.renderer.set_weather_changing(changing, start);
    }

    /// Changes to a built-in weather by name, easing into it over `seconds`. Returns false if
    /// there is no such weather.
    pub fn set_weather(&mut self, name: &str, seconds: f32) -> bool {
        match Weather::by_name(name) {
            Some(weather) => {
                self.renderer.set_weather(weather, seconds);
                true
            }
            None => false,
        }
    }

    /// Changes to the weather in a live report, easing into it over `seconds`: cloud cover low,
    /// middle and high in percent, wind near the ground in km/h and the compass direction it
    /// blows from, the WMO weather code, and the last hour's precipitation in millimetres.
    #[allow(clippy::too_many_arguments)]
    pub fn set_observed_weather(
        &mut self,
        cover_low: f32,
        cover_mid: f32,
        cover_high: f32,
        wind_kmh: f32,
        wind_from: f32,
        code: u32,
        precipitation: f32,
        seconds: f32,
    ) {
        let report = Observation {
            cover_low,
            cover_mid,
            cover_high,
            wind_kmh,
            wind_from,
            code,
            precipitation,
        };
        self.renderer
            .set_weather(Weather::observed(&report), seconds);
    }

    /// The weather now, briefly, for the page to show.
    pub fn weather_summary(&self) -> String {
        let w = self.renderer.weather();
        format!(
            "{:.0}% cloud, wind {:.0} m/s from {:.0}°{}",
            w.coverage * 100.0,
            w.wind_speed,
            w.wind_from,
            if w.rain > 0.05 { ", rain" } else { "" }
        )
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

    /// How the scene's textures are held on the GPU.
    pub fn texture_summary(&self) -> String {
        self.renderer.texture_summary().to_owned()
    }

    /// Starts a built-in particle effect by name, standing at (x, y, z). Returns false if there
    /// is no such effect or no room for it.
    pub fn spawn_effect(&mut self, name: &str, x: f32, y: f32, z: f32) -> bool {
        let Some(effect) = Effect::by_name(name) else {
            return false;
        };
        let at = glam::Mat4::from_translation(Vec3::new(x, y, z));
        self.renderer.particles().spawn(&effect, at).is_some()
    }

    /// Particles alive, from a recent frame.
    pub fn particles_alive(&self) -> u32 {
        self.renderer.stats().particles.alive
    }

    pub fn gpu_error_count(&self) -> u32 {
        self.gpu_errors.load(Ordering::Relaxed)
    }

    /// Whether the scene has a game to play.
    pub fn has_gameplay(&self) -> bool {
        !self.gameplay.is_empty()
    }

    /// Whether this build can play one: the gameplay build can, the plain one can't.
    pub fn can_play() -> bool {
        cfg!(feature = "gameplay")
    }

    fn playing_as_someone(&self) -> bool {
        #[cfg(feature = "gameplay")]
        if let Some(play) = &self.play {
            return play.play.has_player();
        }
        false
    }
}

/// Play mode, in the gameplay build only.
#[cfg(feature = "gameplay")]
#[wasm_bindgen]
impl WebViewer {
    /// Starts the scene's game. Rejects with every mistake in its files, one to a line.
    pub fn play(&mut self) -> Result<(), JsValue> {
        if self.gameplay.is_empty() {
            return Err(JsValue::from_str("this scene has no game to play"));
        }
        let play = crate::play::PlayMode::start(&mut self.renderer, &self.gameplay)
            .map_err(|errors| JsValue::from_str(&errors.join("\n")))?;
        self.play = Some(play);
        Ok(())
    }

    /// Whether a game is running, and someone in it has the camera.
    pub fn has_player(&self) -> bool {
        self.playing_as_someone()
    }

    fn devices(&mut self) -> Option<&mut solstice_game::Devices> {
        self.play.as_mut().map(|play| &mut play.devices)
    }

    /// A key, by its `KeyboardEvent.code`, going down or up.
    pub fn key(&mut self, code: &str, down: bool) {
        if let Some(devices) = self.devices() {
            devices.key(code, down);
        }
    }

    /// A mouse button, `left`, `right` or `middle`, going down or up.
    pub fn mouse_button(&mut self, button: &str, down: bool) {
        if let Some(devices) = self.devices() {
            devices.mouse_button(button, down);
        }
    }

    /// The mouse moving by this many pixels, right and down.
    pub fn mouse_motion(&mut self, x: f32, y: f32) {
        if let Some(devices) = self.devices() {
            devices.mouse_motion(x, y);
        }
    }

    /// A gamepad button's value, from 0 to 1, by its name in the standard layout.
    pub fn gamepad_button(&mut self, button: &str, value: f32) {
        if let Some(devices) = self.devices() {
            devices.gamepad_button(button, value);
        }
    }

    /// A gamepad stick's axis, from -1 to 1: `left_x`, `left_y`, `right_x` or `right_y`.
    pub fn gamepad_axis(&mut self, axis: &str, value: f32) {
        if let Some(devices) = self.devices() {
            devices.gamepad_axis(axis, value);
        }
    }

    /// Where the touch stick is pushed, each axis from -1 to 1.
    pub fn touch_stick(&mut self, x: f32, y: f32) {
        if let Some(devices) = self.devices() {
            devices.touch_stick(x, y);
        }
    }

    /// A drag on the look side of the screen, by this many pixels.
    pub fn touch_look(&mut self, x: f32, y: f32) {
        if let Some(devices) = self.devices() {
            devices.touch_look(x, y);
        }
    }

    pub fn touch_tap(&mut self) {
        if let Some(devices) = self.devices() {
            devices.touch_tap();
        }
    }

    /// Lets every key and button go, for when the page loses focus.
    pub fn release_all(&mut self) {
        if let Some(devices) = self.devices() {
            devices.release_all();
        }
    }

    /// The game's scripts, by name.
    pub fn scripts(&self) -> Vec<String> {
        self.play
            .as_ref()
            .map_or_else(Vec::new, |play| play.play.scripts().to_vec())
    }

    /// Loads a new version of a script, keeping every actor's state. Rejects with the reason
    /// if it fails to load, and the old version keeps running.
    pub fn reload_script(&mut self, name: &str, source: &str) -> Result<(), JsValue> {
        let play = self
            .play
            .as_mut()
            .ok_or_else(|| JsValue::from_str("no game is running"))?;
        play.play
            .reload(name, source)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// What scripts have logged, and every error, since this was last asked, oldest first.
    /// Errors start with `error: `.
    pub fn take_messages(&mut self) -> Vec<String> {
        self.messages
            .drain(..)
            .map(|m| {
                if m.error {
                    format!("error: {}", m.text)
                } else {
                    m.text
                }
            })
            .collect()
    }
}

/// Bytes to fetch first, to learn about the rest of a scene file.
#[wasm_bindgen]
pub fn header_bytes() -> u32 {
    file::HEADER_BYTES as u32
}

/// Reads the start of a scene file. Returns where the manifest starts and how large it is.
#[wasm_bindgen]
pub fn read_header(bytes: &[u8]) -> Result<Vec<f64>, JsValue> {
    let header = file::decode_header(bytes).map_err(error_string)?;
    Ok(vec![
        header.manifest_offset as f64,
        header.manifest_bytes as f64,
    ])
}
