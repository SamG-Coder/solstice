//! The top-level renderer: owns all passes and defines the order of a frame.

use crate::frame::{
    jitter_offset, jitter_projection, Camera, DebugView, Environment, FrameUniform,
};
use crate::gi::{GiVolume, ScreenGi, ScreenGiBindings, LEVEL_COUNT};
use crate::gpu::{Gpu, Target};
use crate::lighting::{environment_group, environment_layout, DeferredLighting};
use crate::post::{PostBindings, PostProcess};
use crate::profiler::{section, Profiler, SECTIONS};
use crate::scene::GpuScene;
use crate::shadow::{ShadowMaps, CASCADE_COUNT};
use crate::sky::Sky;
use crate::streaming::{choose_tiers, StreamRequest, Streamer, StreamingStats};
use crate::vg::{
    CullPhase, CullPipelines, CullWork, GBuffer, Hzb, PassView, RasterBindings, Rasterizer,
    Resolve, ResolveBindings, ViewSlots, VisibilityBuffer,
};
use glam::{Mat4, Vec2};
use solstice_asset::paged::{PagedScene, PAGE_BYTES};
use solstice_asset::LoadedScene;

/// Scales texture gradients, selecting finer mips than one texel per pixel, about 0.75 of a mip
/// here. Temporal filtering averages over several frames' jitter, which softens the image and
/// also suppresses the shimmer that finer mips would otherwise cause.
const TEXTURE_DETAIL: f32 = 0.6;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Capacity of the main view's traversal queue and selected-cluster list.
const MAIN_QUEUE_CAPACITY: u64 = 1 << 22;
const MAIN_VISIBLE_CAPACITY: u64 = 1 << 20;
/// The same for views that need no history: shadow cascades and voxelization.
const AUX_QUEUE_CAPACITY: u64 = 1 << 22;
const AUX_VISIBLE_CAPACITY: u64 = 1 << 20;

/// Assignment of view slots.
const SLOT_MAIN_FIRST: u32 = 0;
const SLOT_MAIN_SECOND: u32 = 1;
const SLOT_SHADOW: u32 = 2;
const SLOT_VOXEL: u32 = SLOT_SHADOW + CASCADE_COUNT as u32;
const SLOT_COUNT: u32 = SLOT_VOXEL + LEVEL_COUNT as u32;

/// A named balance between image quality and GPU cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quality {
    /// Integrated graphics: fewer pixels and triangles, reconstructed by temporal upscaling.
    Low,
    Medium,
    /// Discrete desktop GPUs.
    #[default]
    High,
}

impl Quality {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            _ => return None,
        })
    }
}

/// User-tunable quality settings.
#[derive(Clone, Copy, Debug)]
pub struct RendererSettings {
    /// Largest allowed geometric error on screen, in pixels.
    pub lod_threshold: f32,
    /// The same, in shadow map texels. Shadow maps are the costliest pass on small GPUs, and
    /// a shadow's outline hides far more error than a directly visible surface does.
    pub shadow_lod_threshold: f32,
    /// Internal resolution relative to the output. Temporal upscaling fills in the rest.
    pub render_scale: f32,
    pub debug_view: DebugView,
    /// Blend each frame with reprojected history. Off, aliasing returns, but every frame
    /// shows exactly what was rendered, which is what diagnostics and tests want.
    pub temporal_accumulation: bool,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            lod_threshold: 1.0,
            shadow_lod_threshold: 1.0,
            render_scale: 1.0,
            debug_view: DebugView::Lit,
            temporal_accumulation: true,
        }
    }
}

impl RendererSettings {
    /// Applies a quality preset, leaving diagnostic options alone.
    pub fn apply_quality(&mut self, quality: Quality) {
        (
            self.render_scale,
            self.lod_threshold,
            self.shadow_lod_threshold,
        ) = match quality {
            Quality::Low => (0.6, 2.0, 4.0),
            Quality::Medium => (0.8, 1.5, 2.0),
            Quality::High => (1.0, 1.0, 1.0),
        };
    }
}

/// How much GPU memory a streaming renderer may spend on a scene, in bytes.
#[derive(Clone, Copy, Debug)]
pub struct StreamingBudget {
    pub geometry_bytes: u64,
    pub texture_bytes: u64,
}

impl Default for StreamingBudget {
    /// Enough for a detailed view of the cove in every direction at once.
    fn default() -> Self {
        Self {
            geometry_bytes: 128 << 20,
            texture_bytes: 64 << 20,
        }
    }
}

/// GPU memory holding a scene's own data, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneMemory {
    /// Instances, materials, lights, hierarchy nodes and cluster records. Always resident.
    pub records: u64,
    /// The page pool.
    pub pages: u64,
    pub textures: u64,
}

impl SceneMemory {
    pub fn total(&self) -> u64 {
        self.records + self.pages + self.textures
    }
}

/// Measurements from a recent frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub clusters: u32,
    pub software_clusters: u32,
    pub triangles: u32,
    pub instances: u32,
    pub source_triangles: u64,
    pub unique_triangles: u64,
    pub streaming: StreamingStats,
}

/// Everything whose size depends on the resolution.
struct SizedResources {
    output_size: (u32, u32),
    render_size: (u32, u32),
    visibility: VisibilityBuffer,
    hzb: Hzb,
    _gbuffer: GBuffer,
    cull_scene: wgpu::BindGroup,
    raster: RasterBindings,
    resolve: ResolveBindings,
    screen_gi: ScreenGiBindings,
    deferred: [wgpu::BindGroup; 2],
    post: PostBindings,
    _hdr: Target,
}

/// Previous-frame camera state, for reprojection and occlusion culling.
#[derive(Clone, Copy)]
struct History {
    view: Mat4,
    view_proj: Mat4,
    projection: Mat4,
}

/// Renders a scene with virtualized geometry and global illumination.
pub struct Renderer {
    streamer: Streamer,
    settings: RendererSettings,
    scene: GpuScene,
    frame_uniform: wgpu::Buffer,
    view_slots: ViewSlots,
    cull: CullPipelines,
    main_work: CullWork,
    aux_work: CullWork,
    aux_cull_scene: wgpu::BindGroup,
    aux_raster: RasterBindings,
    rasterizer: Rasterizer,
    resolve: Resolve,
    sky: Sky,
    shadows: ShadowMaps,
    volume: GiVolume,
    screen_gi: ScreenGi,
    environment: wgpu::BindGroup,
    deferred: DeferredLighting,
    post: PostProcess,
    profiler: Option<Profiler>,
    sized: Option<SizedResources>,
    history: Option<History>,
    frame_index: u32,
    time: f32,
}

impl Renderer {
    /// Creates a renderer for a scene held entirely in memory, with all of it on the GPU. It
    /// draws to textures of `output_format`, which must not be sRGB: the renderer applies its
    /// own display encoding.
    pub fn new(
        gpu: &Gpu,
        loaded: &LoadedScene,
        output_format: wgpu::TextureFormat,
        settings: RendererSettings,
    ) -> Self {
        let scene = GpuScene::new(
            gpu,
            &loaded.scene,
            loaded.scene.pages.len() as u32,
            Some(&loaded.upper_textures),
            None,
        );
        let mut streamer = Streamer::new(gpu, &loaded.scene, loaded.scene.pages.len() as u32, None);
        for page in 0..loaded.scene.pages.len() as u32 {
            streamer.install_now(gpu, &scene, page, loaded.page(page));
        }
        streamer.flush(gpu, &scene);
        Self::from_parts(gpu, scene, streamer, output_format, settings)
    }

    /// Creates a renderer that starts with only a scene's pinned pages and smallest textures,
    /// and draws with what it has while the host supplies the rest: see
    /// [`Self::stream_requests`]. GPU memory for the scene stays within `budget`, apart from
    /// what the scene can't do without: its records, pinned pages and smallest textures.
    pub fn streaming(
        gpu: &Gpu,
        paged: &PagedScene,
        output_format: wgpu::TextureFormat,
        settings: RendererSettings,
        budget: StreamingBudget,
    ) -> Self {
        let slots = u32::try_from(budget.geometry_bytes / PAGE_BYTES as u64)
            .unwrap_or(u32::MAX)
            .max(paged.pinned_pages)
            .min(paged.pages.len() as u32);
        let plan = choose_tiers(gpu, paged, budget.texture_bytes);
        let scene = GpuScene::new(gpu, paged, slots, None, Some(plan));
        let mut streamer = Streamer::new(gpu, paged, slots, Some(plan));
        for page in 0..paged.pinned_pages {
            streamer.install_now(gpu, &scene, page, paged.pinned_page(page));
        }
        streamer.flush(gpu, &scene);
        Self::from_parts(gpu, scene, streamer, output_format, settings)
    }

    fn from_parts(
        gpu: &Gpu,
        scene: GpuScene,
        streamer: Streamer,
        output_format: wgpu::TextureFormat,
        settings: RendererSettings,
    ) -> Self {
        let frame_uniform = gpu.uniform_buffer("frame", std::mem::size_of::<FrameUniform>() as u64);
        let view_slots = ViewSlots::new(gpu, SLOT_COUNT);

        let cull = CullPipelines::new(gpu);
        let main_work = CullWork::new(
            gpu,
            &cull,
            "main cull",
            MAIN_QUEUE_CAPACITY,
            MAIN_VISIBLE_CAPACITY,
            streamer.requests_buffer(),
        );
        let aux_work = CullWork::new(
            gpu,
            &cull,
            "aux cull",
            AUX_QUEUE_CAPACITY,
            AUX_VISIBLE_CAPACITY,
            streamer.requests_buffer(),
        );
        let rasterizer = Rasterizer::new(gpu);

        // Views without occlusion culling never read the pyramid, but a binding is required.
        let no_hzb = Target::new(
            gpu,
            "no hzb",
            (1, 1),
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::empty(),
        );
        let aux_cull_scene = cull.scene_bind_group(gpu, &view_slots, &scene, &no_hzb.view);
        let aux_raster = rasterizer.bindings(gpu, &view_slots, &scene, &aux_work);

        let sky = Sky::new(gpu);
        let shadows = ShadowMaps::new(gpu, &rasterizer);
        let environment_layout = environment_layout(gpu);
        let volume = GiVolume::new(gpu, &scene, &aux_work, &environment_layout);
        let environment =
            environment_group(gpu, &environment_layout, &frame_uniform, &volume, &sky);

        Self {
            settings,
            resolve: Resolve::new(gpu),
            screen_gi: ScreenGi::new(gpu, &environment_layout),
            deferred: DeferredLighting::new(gpu, &environment_layout),
            post: PostProcess::new(gpu, output_format),
            profiler: Profiler::new(gpu),
            scene,
            streamer,
            frame_uniform,
            view_slots,
            cull,
            main_work,
            aux_work,
            aux_cull_scene,
            aux_raster,
            rasterizer,
            sky,
            shadows,
            volume,
            environment,
            sized: None,
            history: None,
            frame_index: 0,
            time: 0.0,
        }
    }

    /// The ranges of the scene file to start fetching now, most wanted first. Each must later
    /// be handed to [`Self::deliver`] or [`Self::request_failed`]. `max_fetching` caps the
    /// pages outstanding, and a few texture levels may come on top.
    pub fn stream_requests(&mut self, max_fetching: u32) -> Vec<StreamRequest> {
        self.streamer.take_fetches(max_fetching)
    }

    /// Takes the bytes of a request, unpacked if it was deflated. They reach the GPU within a
    /// few frames. Returns false if they aren't wanted any more or are the wrong size.
    pub fn deliver(&mut self, id: u32, data: Vec<u8>) -> bool {
        self.streamer.deliver_request(id, data)
    }

    /// Reports that a request could not be fetched, so what it holds may be requested again.
    pub fn request_failed(&mut self, id: u32) {
        self.streamer.request_failed(id);
    }

    /// Bytes of GPU memory holding this scene's own data, not counting render targets.
    pub fn scene_bytes(&self) -> SceneMemory {
        let buffer_bytes = self.scene.instances.size()
            + self.scene.nodes.size()
            + self.scene.clusters.size()
            + self.scene.materials.size()
            + self.scene.lights.size();
        SceneMemory {
            records: buffer_bytes,
            pages: self.scene.vertices.size(),
            textures: self.scene.textures.bytes,
        }
    }

    /// How the scene's textures ended up on the GPU, for logs and on-screen readouts.
    pub fn texture_summary(&self) -> &str {
        &self.scene.texture_summary
    }

    pub fn settings_mut(&mut self) -> &mut RendererSettings {
        &mut self.settings
    }

    /// Current counters. They lag a few frames, since readback is asynchronous.
    pub fn stats(&self) -> FrameStats {
        let stats = self.main_work.stats();
        FrameStats {
            clusters: stats.clusters(),
            software_clusters: stats.software_clusters(),
            triangles: stats.triangles(),
            instances: self.scene.instance_count,
            source_triangles: self.scene.source_triangles,
            unique_triangles: self.scene.unique_triangles,
            streaming: self.streamer.stats(),
        }
    }

    /// GPU milliseconds per entry of [`SECTIONS`], where the platform can measure them.
    pub fn section_times(&self) -> Option<[f32; SECTIONS.len()]> {
        self.profiler.as_ref().and_then(Profiler::latest_ms)
    }

    /// Renders one frame to `output`, a view of a texture of the size given.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        camera: &Camera,
        environment: &Environment,
        output: &wgpu::TextureView,
        output_size: (u32, u32),
        delta_seconds: f32,
    ) {
        let scale = self.settings.render_scale;
        let render_size = (
            ((output_size.0 as f32 * scale) as u32).max(16),
            ((output_size.1 as f32 * scale) as u32).max(16),
        );
        self.ensure_sized(gpu, render_size, output_size);
        self.time += delta_seconds;

        // Everything below draws what's installed from here on.
        self.streamer.update(gpu, &self.scene);
        if self.streamer.should_refresh_cached_views() {
            self.shadows.mark_stale();
            self.volume.mark_stale();
        }

        let aspect = render_size.0 as f32 / render_size.1 as f32;
        let parity = (self.frame_index & 1) as usize;

        self.write_frame_data(gpu, camera, environment, render_size);
        let due_cascades = self.shadows.prepare(
            gpu,
            &self.view_slots,
            SLOT_SHADOW,
            camera,
            aspect,
            environment.sun_direction,
            self.settings.shadow_lod_threshold,
        );
        let volume_work = self
            .volume
            .prepare(gpu, &self.view_slots, SLOT_VOXEL, camera.position);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        self.streamer.begin_frame(gpu, &self.scene, &mut encoder);

        mark(&self.profiler, &mut encoder, section::SKY);
        self.sky.record(
            gpu,
            &mut encoder,
            environment.sun_direction,
            camera.position.y,
        );

        mark(&self.profiler, &mut encoder, section::SHADOWS);
        self.shadows.record(
            &mut encoder,
            &self.cull,
            &self.scene,
            &self.aux_cull_scene,
            &self.aux_work,
            &self.aux_raster,
            SLOT_SHADOW,
            &due_cascades,
        );

        self.volume.record(
            &mut encoder,
            &self.cull,
            &self.scene,
            &self.aux_cull_scene,
            &self.aux_work,
            &self.environment,
            SLOT_VOXEL,
            &volume_work,
            self.profiler.as_ref(),
        );

        let sized = self.sized.as_ref().expect("resources were just created");

        // Geometry: two occlusion phases into one visibility buffer, then its G-buffer.
        sized.visibility.clear(&mut encoder);
        for (phase, slot, cull_section, raster_section) in [
            (CullPhase::First, SLOT_MAIN_FIRST, section::CULL_FIRST, section::RASTER_FIRST),
            (CullPhase::Second, SLOT_MAIN_SECOND, section::CULL_SECOND, section::RASTER_SECOND),
        ] {
            mark(&self.profiler, &mut encoder, cull_section);
            self.cull.record(
                &mut encoder,
                &self.scene,
                &sized.cull_scene,
                slot,
                &self.main_work,
                phase,
            );
            mark(&self.profiler, &mut encoder, raster_section);
            self.rasterizer.record(
                &mut encoder,
                &sized.raster,
                slot,
                &self.main_work,
                &sized.visibility,
            );
            sized.hzb.record(&mut encoder);
        }
        mark(&self.profiler, &mut encoder, section::RESOLVE);
        self.resolve
            .record(&mut encoder, &sized.resolve, render_size);

        // Lighting.
        mark(&self.profiler, &mut encoder, section::SCREEN_GI);
        self.screen_gi
            .record(&mut encoder, &sized.screen_gi, &self.environment, parity);
        mark(&self.profiler, &mut encoder, section::LIGHTING);
        self.deferred.record(
            &mut encoder,
            &sized.deferred[parity],
            &self.environment,
            render_size,
        );

        // From the lit scene to the display.
        mark(&self.profiler, &mut encoder, section::POST);
        let debug = self.settings.debug_view;
        let passthrough = !matches!(
            debug,
            DebugView::Lit | DebugView::DiffuseGi | DebugView::Reflections | DebugView::Voxels
        );
        self.post.record(
            gpu,
            &mut encoder,
            &sized.post,
            output,
            parity,
            self.frame_index,
            delta_seconds,
            environment.sun_illuminance,
            passthrough,
        );

        mark(&self.profiler, &mut encoder, section::END);
        if let Some(profiler) = &mut self.profiler {
            profiler.resolve(&mut encoder);
        }
        self.main_work.capture_stats(&mut encoder);
        self.streamer.capture_requests(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        self.main_work.map_stats();
        self.streamer.map_requests();
        if let Some(profiler) = &mut self.profiler {
            profiler.request_map();
        }
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    fn ensure_sized(&mut self, gpu: &Gpu, render_size: (u32, u32), output_size: (u32, u32)) {
        let unchanged = self
            .sized
            .as_ref()
            .is_some_and(|s| s.render_size == render_size && s.output_size == output_size);
        if unchanged {
            return;
        }
        let visibility = VisibilityBuffer::new(gpu, &self.rasterizer, render_size);
        let hzb = Hzb::new(gpu, &visibility);
        let gbuffer = GBuffer::new(gpu, render_size);
        let hdr = Target::new(
            gpu,
            "hdr",
            render_size,
            HDR_FORMAT,
            wgpu::TextureUsages::empty(),
        );
        let cull_scene = self
            .cull
            .scene_bind_group(gpu, &self.view_slots, &self.scene, &hzb.view);
        let raster = self
            .rasterizer
            .bindings(gpu, &self.view_slots, &self.scene, &self.main_work);
        let resolve = self.resolve.bindings(
            gpu,
            &self.frame_uniform,
            &self.scene,
            &visibility,
            &self.main_work,
            &gbuffer,
        );

        let history = PostProcess::history_targets(gpu, output_size);
        // On each frame the GI passes read the history written on the frame before, which is
        // the other half of the pair.
        let screen_gi = self.screen_gi.bindings(
            gpu,
            render_size,
            &gbuffer,
            [&history[1].view, &history[0].view],
        );
        let deferred = self.deferred.bind_groups(
            gpu,
            &gbuffer,
            &screen_gi,
            &hdr.view,
            &self.shadows,
            &self.scene,
        );
        let post = self.post.bindings(
            gpu,
            &self.frame_uniform,
            &gbuffer,
            &hdr.view,
            history,
            render_size,
            output_size,
        );

        self.history = None;
        self.frame_index = 0;
        self.sized = Some(SizedResources {
            output_size,
            render_size,
            visibility,
            hzb,
            _gbuffer: gbuffer,
            cull_scene,
            raster,
            resolve,
            screen_gi,
            deferred,
            post,
            _hdr: hdr,
        });
    }

    fn write_frame_data(
        &mut self,
        gpu: &Gpu,
        camera: &Camera,
        environment: &Environment,
        size: (u32, u32),
    ) {
        let viewport = Vec2::new(size.0 as f32, size.1 as f32);
        let projection = camera.projection(viewport.x / viewport.y);
        let view = camera.view();
        let unjittered = projection * view;

        let jitter_ndc = jitter_offset(self.frame_index) * 2.0 / viewport;
        let jittered = jitter_projection(projection, jitter_ndc) * view;

        let current = History {
            view,
            view_proj: unjittered,
            projection,
        };
        let previous = self.history.unwrap_or(current);

        let frame = FrameUniform {
            view_proj: jittered,
            inv_view_proj: jittered.inverse(),
            unjittered_view_proj: unjittered,
            prev_view_proj: previous.view_proj,
            view,
            camera_position: camera.position,
            frame_index: self.frame_index,
            viewport,
            inv_viewport: viewport.recip(),
            jitter_ndc,
            near: camera.near,
            debug_mode: self.settings.debug_view as u32,
            sun_direction: environment.sun_direction,
            sun_illuminance: environment.sun_illuminance,
            sun_color: environment.sun_color,
            sky_intensity: environment.sky_intensity,
            light_count: self.scene.light_count,
            time: self.time,
            gi_intensity: environment.gi_intensity,
            temporal_accumulation: u32::from(self.settings.temporal_accumulation),
            mip_scale: self.settings.render_scale * TEXTURE_DETAIL,
            _pad: [0.0; 3],
        };
        gpu.queue
            .write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&frame));

        let sized = self
            .sized
            .as_ref()
            .expect("resources exist before frame data");
        let hzb_size = Vec2::new(sized.hzb.size.0 as f32, sized.hzb.size.1 as f32);
        let main_view = |occlusion: &History, phase: u32, occlusion_enabled: bool| PassView {
            raster_view_proj: jittered,
            occlusion_view: occlusion.view,
            planes: PassView::frustum_planes(unjittered),
            camera_position: camera.position,
            lod_threshold: self.settings.lod_threshold,
            viewport,
            proj_scale: Vec2::new(occlusion.projection.x_axis.x, occlusion.projection.y_axis.y),
            hzb_size,
            near: camera.near,
            lod_scale: 0.5 * viewport.y * projection.y_axis.y,
            is_ortho: 0,
            phase,
            occlusion_enabled: u32::from(occlusion_enabled),
            allow_software: 1,
        };
        self.view_slots.write(
            gpu,
            SLOT_MAIN_FIRST,
            &main_view(&previous, 0, self.history.is_some()),
        );
        self.view_slots
            .write(gpu, SLOT_MAIN_SECOND, &main_view(&current, 1, true));

        self.history = Some(current);
    }
}

fn mark(profiler: &Option<Profiler>, encoder: &mut wgpu::CommandEncoder, section: usize) {
    if let Some(profiler) = profiler {
        profiler.mark(encoder, section);
    }
}
