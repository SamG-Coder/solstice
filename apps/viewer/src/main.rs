//! Native viewer: an interactive window, or a headless capture for automated checks.
//!
//! ```text
//! viewer [SCENE] [--capture out.png] [--frames N] [--size WxH]
//!        [--camera x,y,z,yaw,pitch] [--debug MODE] [--sun x,y,z]
//! ```

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use glam::Vec3;
    use solstice_asset::SceneAsset;
    use solstice_render::{Camera, DebugView, Environment, Gpu, Renderer, RendererSettings};
    use solstice_viewer::{default_camera, default_environment, CameraInput, FlyCamera};
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;
    use winit::application::ApplicationHandler;
    use winit::dpi::PhysicalSize;
    use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{CursorGrabMode, Window, WindowId};

    const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    struct Options {
        scene: PathBuf,
        capture: Option<PathBuf>,
        frames: u32,
        size: (u32, u32),
        camera: Camera,
        environment: Environment,
        settings: RendererSettings,
    }

    fn parse_args() -> Result<Options, String> {
        let mut options = Options {
            scene: PathBuf::from("assets/temple.solstice"),
            capture: None,
            frames: 60,
            size: (1600, 900),
            camera: default_camera(),
            environment: default_environment(),
            settings: RendererSettings::default(),
        };
        let floats = |text: &str| -> Result<Vec<f32>, String> {
            text.split(',')
                .map(|part| part.parse::<f32>().map_err(|e| format!("{part}: {e}")))
                .collect()
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
            match arg.as_str() {
                "--capture" => options.capture = Some(PathBuf::from(value()?)),
                "--frames" => {
                    options.frames = value()?.parse().map_err(|e| format!("--frames: {e}"))?
                }
                "--size" => {
                    let text = value()?;
                    let (w, h) = text.split_once('x').ok_or("--size expects WxH")?;
                    options.size = (
                        w.parse().map_err(|e| format!("--size: {e}"))?,
                        h.parse().map_err(|e| format!("--size: {e}"))?,
                    );
                }
                "--camera" => {
                    let v = floats(&value()?)?;
                    let [x, y, z, yaw, pitch] = v[..] else {
                        return Err("--camera expects x,y,z,yaw,pitch".into());
                    };
                    options.camera.position = Vec3::new(x, y, z);
                    options.camera.yaw = yaw;
                    options.camera.pitch = pitch;
                }
                "--sun" => {
                    let v = floats(&value()?)?;
                    let [x, y, z] = v[..] else {
                        return Err("--sun expects x,y,z".into());
                    };
                    options.environment.sun_direction = Vec3::new(x, y, z).normalize();
                }
                "--debug" => {
                    let name = value()?;
                    options.settings.debug_view = DebugView::from_name(&name)
                        .ok_or_else(|| format!("unknown debug view: {name}"))?;
                }
                "--lod" => {
                    options.settings.lod_threshold =
                        value()?.parse().map_err(|e| format!("--lod: {e}"))?;
                }
                other if !other.starts_with('-') => options.scene = PathBuf::from(other),
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        Ok(options)
    }

    fn load_scene(path: &PathBuf) -> Result<SceneAsset, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let scene = solstice_asset::file::decode(&bytes)?;
        log::info!(
            "loaded {} ({:.0} MB) in {:.2}s",
            path.display(),
            bytes.len() as f64 / 1.0e6,
            start.elapsed().as_secs_f32()
        );
        Ok(scene)
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
        )
        .init();
        let options = parse_args()?;
        let scene = load_scene(&options.scene)?;
        if options.capture.is_some() {
            capture(&options, &scene)
        } else {
            let event_loop = EventLoop::new()?;
            let mut app = App {
                options,
                scene,
                state: None,
                input: CameraInput::default(),
                keys: HashSet::new(),
                captured_mouse: false,
            };
            event_loop.run_app(&mut app)?;
            Ok(())
        }
    }

    fn create_instance() -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        })
    }

    /// Renders offscreen, saves a PNG, and fails loudly on any GPU validation error.
    fn capture(options: &Options, scene: &SceneAsset) -> Result<(), Box<dyn std::error::Error>> {
        let path = options
            .capture
            .as_ref()
            .expect("capture mode requires a path");
        let gpu = pollster::block_on(Gpu::request(&create_instance(), None))?;

        let errors: Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let sink = errors.clone();
        gpu.device
            .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                sink.lock().expect("error sink").push(error.to_string());
            }));

        let start = Instant::now();
        let mut renderer = Renderer::new(&gpu, scene, CAPTURE_FORMAT, options.settings);
        log::info!("renderer created in {:.2}s", start.elapsed().as_secs_f32());

        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d {
                width: options.size.0,
                height: options.size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CAPTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());

        let start = Instant::now();
        let mut slowest = 0.0_f32;
        let mut last_time = Instant::now();
        for frame in 0..options.frames {
            renderer.render(
                &gpu,
                &options.camera,
                &options.environment,
                &view,
                options.size,
                1.0 / 60.0,
            );
            gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
            if frame > 10 {
                slowest = slowest.max(last_time.elapsed().as_secs_f32());
            }
            last_time = Instant::now();
        }
        let average = start.elapsed().as_secs_f32() / options.frames.max(1) as f32;

        save_png(&gpu, &target, options.size, path)?;

        let stats = renderer.stats();
        println!(
            "avg {:.2} ms, worst {:.2} ms | {} clusters ({} software), {} triangles of {} source ({} unique), {} instances",
            average * 1000.0,
            slowest * 1000.0,
            stats.clusters,
            stats.software_clusters,
            stats.triangles,
            stats.source_triangles,
            stats.unique_triangles,
            stats.instances,
        );

        let errors = errors.lock().expect("error sink");
        if errors.is_empty() {
            println!("saved {} with no GPU errors", path.display());
            Ok(())
        } else {
            Err(format!("{} GPU errors, first:\n{}", errors.len(), errors[0]).into())
        }
    }

    fn save_png(
        gpu: &Gpu,
        texture: &wgpu::Texture,
        size: (u32, u32),
        path: &PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let row_bytes = size.0 * 4;
        let aligned = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture staging"),
            size: u64::from(aligned) * u64::from(size.1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned),
                    rows_per_image: Some(size.1),
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("mapping the capture buffer failed");
        });
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;

        let data = staging.slice(..).get_mapped_range()?;
        let mut pixels = Vec::with_capacity((row_bytes * size.1) as usize);
        for row in data.chunks_exact(aligned as usize) {
            pixels.extend_from_slice(&row[..row_bytes as usize]);
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut encoder = png::Encoder::new(file, size.0, size.1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&pixels)?;
        Ok(())
    }

    struct WindowState {
        window: Arc<Window>,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        gpu: Gpu,
        renderer: Renderer,
        camera: FlyCamera,
        last_frame: Instant,
        title_timer: f32,
        frames_since_title: u32,
    }

    struct App {
        options: Options,
        scene: SceneAsset,
        state: Option<WindowState>,
        input: CameraInput,
        keys: HashSet<KeyCode>,
        captured_mouse: bool,
    }

    impl App {
        fn create_state(
            &self,
            event_loop: &ActiveEventLoop,
        ) -> Result<WindowState, Box<dyn std::error::Error>> {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("Solstice")
                        .with_inner_size(PhysicalSize::new(
                            self.options.size.0,
                            self.options.size.1,
                        )),
                )?,
            );
            let instance = create_instance();
            let surface = instance.create_surface(window.clone())?;
            let gpu = pollster::block_on(Gpu::request(&instance, Some(&surface)))?;

            let size = window.inner_size();
            let mut config = surface
                .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
                .ok_or("this surface cannot be presented to")?;
            config.format = config.format.remove_srgb_suffix();
            config.present_mode = wgpu::PresentMode::AutoVsync;
            surface.configure(&gpu.device, &config);

            let renderer = Renderer::new(&gpu, &self.scene, config.format, self.options.settings);
            Ok(WindowState {
                window,
                surface,
                config,
                gpu,
                renderer,
                camera: FlyCamera::new(self.options.camera),
                last_frame: Instant::now(),
                title_timer: 0.0,
                frames_since_title: 0,
            })
        }

        fn axis(&self, positive: KeyCode, negative: KeyCode) -> f32 {
            f32::from(u8::from(self.keys.contains(&positive)))
                - f32::from(u8::from(self.keys.contains(&negative)))
        }

        fn draw(&mut self) {
            let forward = self.axis(KeyCode::KeyW, KeyCode::KeyS);
            let right = self.axis(KeyCode::KeyD, KeyCode::KeyA);
            let up = self.axis(KeyCode::KeyE, KeyCode::KeyQ);
            let fast = self.keys.contains(&KeyCode::ShiftLeft);
            let environment = self.options.environment;
            let look = std::mem::take(&mut self.input);
            let Some(state) = self.state.as_mut() else {
                return;
            };

            let now = Instant::now();
            let delta = (now - state.last_frame).as_secs_f32();
            state.last_frame = now;
            state.camera.update(
                &CameraInput {
                    forward,
                    right,
                    up,
                    fast,
                    ..look
                },
                delta,
            );

            state.title_timer += delta;
            state.frames_since_title += 1;
            if state.title_timer > 0.5 {
                let stats = state.renderer.stats();
                state.window.set_title(&format!(
                    "Solstice | {:.0} fps | {:.2}M tris of {:.0}M | {} clusters ({} sw)",
                    state.frames_since_title as f32 / state.title_timer,
                    stats.triangles as f32 / 1.0e6,
                    stats.source_triangles as f32 / 1.0e6,
                    stats.clusters,
                    stats.software_clusters,
                ));
                state.title_timer = 0.0;
                state.frames_since_title = 0;
            }

            let texture = match state.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(texture)
                | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    state.surface.configure(&state.gpu.device, &state.config);
                    return;
                }
                _ => return,
            };
            let view = texture.texture.create_view(&Default::default());
            state.renderer.render(
                &state.gpu,
                &state.camera.camera,
                &environment,
                &view,
                (state.config.width, state.config.height),
                delta,
            );
            state.gpu.queue.present(texture);
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.state.is_some() {
                return;
            }
            match self.create_state(event_loop) {
                Ok(state) => self.state = Some(state),
                Err(error) => {
                    log::error!("{error}");
                    event_loop.exit();
                }
            }
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(state) = self.state.as_mut() {
                        if size.width > 0 && size.height > 0 {
                            state.config.width = size.width;
                            state.config.height = size.height;
                            state.surface.configure(&state.gpu.device, &state.config);
                        }
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match event.state {
                            ElementState::Pressed => {
                                self.keys.insert(code);
                                if code == KeyCode::Escape {
                                    self.set_mouse_capture(false);
                                }
                            }
                            ElementState::Released => {
                                self.keys.remove(&code);
                            }
                        }
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => self.set_mouse_capture(true),
                WindowEvent::RedrawRequested => {
                    self.draw();
                    if let Some(state) = &self.state {
                        state.window.request_redraw();
                    }
                }
                _ => {}
            }
        }

        fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
            if let (DeviceEvent::MouseMotion { delta }, true) = (event, self.captured_mouse) {
                self.input.look_yaw -= delta.0 as f32 * 0.002;
                self.input.look_pitch -= delta.1 as f32 * 0.002;
            }
        }
    }

    impl App {
        fn set_mouse_capture(&mut self, capture: bool) {
            let Some(state) = &self.state else { return };
            let mode = if capture {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            };
            if state.window.set_cursor_grab(mode).is_err() && capture {
                let _ = state.window.set_cursor_grab(CursorGrabMode::Confined);
            }
            state.window.set_cursor_visible(!capture);
            self.captured_mouse = capture;
        }
    }
}
