//! The top-level renderer: owns all passes and defines the order of a frame.

use crate::clouds::{CloudBindings, Clouds, Weather};
use crate::frame::{
    jitter_offset, jitter_projection, Camera, DebugView, Environment, FrameUniform, NO_SEA,
};
use crate::gi::{GiVolume, ScreenGi, ScreenGiBindings, LEVEL_COUNT};
use crate::gpu::{Gpu, Target};
use crate::lighting::{
    environment_group, environment_layout, DeferredLighting, LightShadows, LightTiles,
};
use crate::ocean::grid::PATCH_CELLS;
use crate::ocean::surface::{SceneViews, WaterSurface, WaterTargets};
use crate::ocean::waterline::{SeaViews, Waterline};
use crate::ocean::{Ocean, Sea, SeaState};
use crate::particles::{
    LightInputs, ParticleStats, ParticleTargets, Particles, POOL_HIGH, POOL_LOW, POOL_MEDIUM,
};
use crate::post::{PostBindings, PostProcess};
use crate::profiler::{section, Profiler, SECTIONS};
use crate::scene::GpuScene;
use crate::shadow::{ShadowMaps, CASCADE_COUNT, MOVING_SLOTS};
use crate::sky::Sky;
use crate::streaming::{choose_tiers, StreamRequest, Streamer, StreamingStats};
use crate::vg::{
    CullPhase, CullPipelines, CullWork, GBuffer, Hzb, InstanceFilter, PassView, RasterBindings,
    Rasterizer, Resolve, ResolveBindings, ViewSlots, VisibilityBuffer,
};
use glam::{Mat4, Vec2, Vec3, Vec4};
use solstice_asset::paged::{PagedScene, PAGE_BYTES};
use solstice_asset::LoadedScene;

/// Scales texture gradients, selecting finer mips than one texel per pixel, about 0.75 of a mip
/// here. Temporal filtering averages over several frames' jitter, which softens the image and
/// also suppresses the shimmer that finer mips would otherwise cause.
const TEXTURE_DETAIL: f32 = 0.6;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Capacity of the main view's traversal queue and selected-cluster list.
const MAIN_QUEUE_CAPACITY: u64 = 1 << 22;
const MAIN_VISIBLE_CAPACITY: u64 = 1 << 21;
/// The same for views that need no history: shadow cascades and voxelization.
const AUX_QUEUE_CAPACITY: u64 = 1 << 22;
const AUX_VISIBLE_CAPACITY: u64 = 1 << 21;

/// Assignment of view slots.
const SLOT_MAIN_FIRST: u32 = 0;
const SLOT_MAIN_SECOND: u32 = 1;
const SLOT_SHADOW: u32 = 2;
const SLOT_VOXEL: u32 = SLOT_SHADOW + CASCADE_COUNT as u32;
const SLOT_SEABED: u32 = SLOT_VOXEL + LEVEL_COUNT as u32;
const SLOT_MOVING_SHADOW: u32 = SLOT_SEABED + 1;
const SLOT_COUNT: u32 = SLOT_MOVING_SHADOW + MOVING_SLOTS;

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
    /// Draw volumetric clouds, their shadows and their light in the sky.
    pub clouds: bool,
    /// Texels across each of the ocean's wave tiles: a power of two, at most 256.
    pub ocean_resolution: u32,
    /// Rays each GI probe traces on a side: one of `PROBE_RAY_OPTIONS`. More rays mean less
    /// boiling in indirect light, for more tracing.
    pub probe_rays: u32,
    /// Cells across each of the sea surface's patches: a multiple of four.
    pub ocean_patch_cells: u32,
    /// Particles the pool holds, across every effect.
    pub particle_pool: u32,
    /// Scales every effect's spawn rate: fewer particles for smaller GPUs.
    pub particle_detail: f32,
    /// Each particle's light is worked out afresh once in this many frames, with ambient rays
    /// of this many steps through the GI volume.
    pub particle_light_interval: u32,
    pub particle_ray_steps: u32,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            lod_threshold: 1.0,
            shadow_lod_threshold: 1.0,
            render_scale: 1.0,
            debug_view: DebugView::Lit,
            temporal_accumulation: true,
            clouds: true,
            ocean_resolution: 256,
            probe_rays: 8,
            ocean_patch_cells: PATCH_CELLS,
            particle_pool: POOL_HIGH,
            particle_detail: 1.0,
            particle_light_interval: 1,
            particle_ray_steps: 16,
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
        self.ocean_resolution = if quality == Quality::Low { 128 } else { 256 };
        self.probe_rays = if quality == Quality::Low { 6 } else { 8 };
        self.ocean_patch_cells = if quality == Quality::Low {
            PATCH_CELLS / 2
        } else {
            PATCH_CELLS
        };
        (
            self.particle_pool,
            self.particle_detail,
            self.particle_light_interval,
            self.particle_ray_steps,
        ) = match quality {
            Quality::Low => (POOL_LOW, 0.35, 4, 8),
            Quality::Medium => (POOL_MEDIUM, 0.6, 2, 12),
            Quality::High => (POOL_HIGH, 1.0, 1, 16),
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
    pub particles: ParticleStats,
    /// What the most recently read back shadow cascade redraw selected. Counters take a few
    /// frames to arrive, and a redraw is skipped over while the last is still in flight.
    pub shadow_clusters: u32,
    pub shadow_triangles: u32,
    /// How many of those clusters went to the software rasterizer.
    pub shadow_software_clusters: u32,
    /// Queue entries and clusters the main view, and that cascade, lost for want of room in
    /// their lists. Zero unless the scene has outgrown them, in which case the image, or the
    /// shadow, has holes.
    pub dropped: u32,
    pub shadow_dropped: u32,
}

/// Everything whose size depends on the resolution.
struct SizedResources {
    output_size: (u32, u32),
    render_size: (u32, u32),
    visibility: VisibilityBuffer,
    hzb: Hzb,
    gbuffer: GBuffer,
    cull_scene: wgpu::BindGroup,
    raster: RasterBindings,
    resolve: ResolveBindings,
    screen_gi: ScreenGiBindings,
    clouds: CloudBindings,
    deferred: [wgpu::BindGroup; 2],
    /// The bindings of the pass that lists which lights reach each tile of the screen; the
    /// list itself is held by these and the deferred bind groups.
    light_tiles_group: wgpu::BindGroup,
    post: PostBindings,
    hdr: Target,
    /// Where the water pass draws, when there is a sea.
    water: Option<WaterTargets>,
    /// The depth particles test against.
    particles: ParticleTargets,
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
    /// Where the sea has been, for how wet surfaces are.
    waterline: Waterline,
    sky: Sky,
    clouds: Clouds,
    /// The lighting bind group as cloud passes see it: without the clouds they are making.
    cloud_environment: wgpu::BindGroup,
    shadows: ShadowMaps,
    volume: GiVolume,
    screen_gi: ScreenGi,
    environment: wgpu::BindGroup,
    deferred: DeferredLighting,
    light_tiles: LightTiles,
    light_shadows: LightShadows,
    post: PostProcess,
    profiler: Option<Profiler>,
    /// The open sea, if the scene has one, and an ocean waiting to be eased into, with the
    /// seconds to take. The sea is made on the next frame, which has the GPU to hand.
    sea: Option<Sea>,
    ocean_request: Option<(Option<Ocean>, f32)>,
    /// The calm sea's level, from the scene.
    sea_level: f32,
    /// Draws and shades the sea's surface, when there is a sea.
    water: Option<WaterSurface>,
    /// Effects, and the pool their particles live in.
    particles: Particles,
    /// The layout of the lighting bind group, for passes made after the renderer.
    environment_layout: wgpu::BindGroupLayout,
    sized: Option<SizedResources>,
    history: Option<History>,
    frame_index: u32,
    time: f32,
    /// Seconds since the previous frame.
    frame_seconds: f32,
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
        Self::from_parts(gpu, scene, streamer, output_format, settings).with_sea(&loaded.scene.sea)
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
        Self::from_parts(gpu, scene, streamer, output_format, settings).with_sea(&paged.sea)
    }

    /// Gives the renderer the scene's sea, if it has one.
    fn with_sea(mut self, sea: &Option<solstice_asset::Sea>) -> Self {
        if let Some(sea) = sea {
            self.sea_level = sea.level;
            let ocean =
                Ocean::by_name(sea.ocean.trim()).map_or_else(|| Ocean::parse(&sea.ocean), Ok);
            let ocean = ocean.unwrap_or_else(|error| {
                log::warn!("the scene's ocean is unreadable ({error}); using coastal water");
                Ocean::default()
            });
            self.ocean_request = Some((Some(ocean), 0.0));
        }
        self
    }

    /// Sets the calm sea's level, in metres.
    pub fn set_sea_level(&mut self, level: f32) {
        self.sea_level = level;
    }

    /// How many instances the scene has. Instances are named by their index in the scene file,
    /// from 0 to this.
    pub fn instance_count(&self) -> u32 {
        self.scene.instance_table.len() as u32
    }

    /// Moves an instance to `world` from the next frame on, as motion over the frame, which
    /// motion blur and temporal anti-aliasing follow. An instance that moves becomes movable, as
    /// [`Self::set_instance_movable`] makes it, if it was not already. Returns false, and does
    /// nothing, for an instance that has no geometry or does not exist.
    ///
    /// `world` must be invertible. To make something vanish, hide it with
    /// [`Self::set_instance_hidden`] rather than scaling it to nothing.
    pub fn move_instance(&mut self, instance: u32, world: Mat4) -> bool {
        self.scene
            .instance_table
            .set_transform(instance, world, false)
    }

    /// Puts an instance at `world` from the next frame on, as a jump rather than a motion, as
    /// for a respawn: it shows no motion getting there. Otherwise as [`Self::move_instance`].
    pub fn place_instance(&mut self, instance: u32, world: Mat4) -> bool {
        self.scene
            .instance_table
            .set_transform(instance, world, true)
    }

    /// Makes an instance movable, or static again. A movable instance is drawn afresh every
    /// frame, with its shadow drawn separately, and so costs more; a static one is drawn into
    /// views kept from frame to frame. Either change redraws those views once. Set an instance
    /// movable before it starts moving, for example when a scene loads, to have that happen at
    /// a quiet moment.
    pub fn set_instance_movable(&mut self, instance: u32, movable: bool) -> bool {
        self.scene.instance_table.set_movable(instance, movable)
    }

    /// Keeps a movable instance in the GI voxels, where it stood when they were last drawn, or
    /// leaves it out (the default). Movable instances are left out because the voxels are kept
    /// from frame to frame, and would hold them where they were. Something that moves only a
    /// little, such as a ship riding at anchor, is better kept in: its shade and the light it
    /// bounces, and its reflection in the water, would otherwise be missing.
    pub fn set_instance_stays_in_gi(&mut self, instance: u32, stays: bool) -> bool {
        self.scene.instance_table.set_stays_in_gi(instance, stays)
    }

    /// Hides an instance, or shows it again.
    pub fn set_instance_hidden(&mut self, instance: u32, hidden: bool) -> bool {
        self.scene.instance_table.set_hidden(instance, hidden)
    }

    /// Scales how brightly an instance's material glows, as a lamp is lit or put out: 1 as the
    /// material says, 0 dark.
    pub fn set_instance_glow(&mut self, instance: u32, glow: f32) -> bool {
        self.scene.instance_table.set_glow(instance, glow)
    }

    /// Where an instance will be drawn next frame.
    pub fn instance_transform(&self, instance: u32) -> Option<Mat4> {
        self.scene.instance_table.transform(instance)
    }

    /// An instance's bounding sphere in the world, as its centre and radius.
    pub fn instance_bounds(&self, instance: u32) -> Option<(Vec3, f32)> {
        self.scene.instance_table.world_sphere(instance)
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
        let waterline = Waterline::new(gpu);
        let environment_layout = environment_layout(gpu);
        let light_shadows = LightShadows::new(gpu, &environment_layout, &scene);
        let volume = GiVolume::new(
            gpu,
            &scene,
            &aux_work,
            &environment_layout,
            &light_shadows.view,
        );
        let mut clouds = Clouds::new(gpu, &environment_layout);
        clouds.set_enabled(settings.clouds);
        let environment = environment_group(
            gpu,
            &environment_layout,
            &frame_uniform,
            &volume,
            &sky,
            clouds.sky(),
        );
        let cloud_environment = environment_group(
            gpu,
            &environment_layout,
            &frame_uniform,
            &volume,
            &sky,
            clouds.placeholders(),
        );

        Self {
            settings,
            resolve: Resolve::new(gpu, waterline.resolve_layout()),
            waterline,
            screen_gi: ScreenGi::new(gpu, &environment_layout),
            deferred: DeferredLighting::new(gpu, &environment_layout),
            light_tiles: LightTiles::new(gpu, &environment_layout),
            light_shadows,
            post: PostProcess::new(gpu, output_format),
            particles: Particles::new(
                gpu,
                &frame_uniform,
                settings.particle_pool,
                &environment_layout,
                LightInputs {
                    lights: scene.lights.clone(),
                    shadow_uniform: shadows.uniform.clone(),
                    shadow_maps: shadows.array_view.clone(),
                    shadow_sampler: shadows.comparison_sampler.clone(),
                },
            ),
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
            clouds,
            cloud_environment,
            shadows,
            volume,
            environment,
            sea: None,
            ocean_request: None,
            sea_level: 0.0,
            water: None,
            environment_layout,
            sized: None,
            history: None,
            frame_index: 0,
            time: 0.0,
            frame_seconds: 1.0 / 60.0,
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

    /// The effects running, to start, steer and stop them.
    pub fn particles(&mut self) -> &mut Particles {
        &mut self.particles
    }

    /// Gives the scene an open sea, easing from the current one over `seconds`, or at once for
    /// 0; `None` takes the sea away. Its waves follow the weather's wind.
    pub fn set_ocean(&mut self, ocean: Option<Ocean>, seconds: f32) {
        self.ocean_request = Some((ocean, seconds));
    }

    /// The ocean as it is now, part way to any new one, or `None` without a sea.
    pub fn ocean(&self) -> Option<Ocean> {
        match (self.ocean_request, &self.sea) {
            // A change not yet begun: none, or one that is at once, or the first sea, is already
            // as good as made; one that eases has not moved yet.
            (Some((None, _)), _) => None,
            (Some((Some(ocean), seconds)), sea) if sea.is_none() || seconds <= 0.0 => Some(ocean),
            (_, sea) => sea.as_ref().map(Sea::ocean),
        }
    }

    /// The wind and swell the waves are drawn from now, which lag the weather's wind.
    pub fn sea_state(&self) -> Option<SeaState> {
        self.sea.as_ref().and_then(Sea::state)
    }

    /// Current counters. They lag a few frames, since readback is asynchronous.
    /// Changes the weather, easing into it over about `seconds`, or at once for 0. Clouds
    /// drift with its wind and change shape as they go.
    pub fn set_weather(&mut self, weather: Weather, seconds: f32) {
        self.clouds.set_weather(weather, seconds);
    }

    /// Lets the weather change by itself, starting from the named weather, or stops it: the
    /// sky drifts, grows and clears through the day.
    pub fn set_weather_changing(&mut self, changing: bool, start: &str) {
        self.clouds.set_changing(changing, start);
    }

    /// Moves the weather on by `seconds` at once, as if that much time had passed: for tools,
    /// tests, and catching up after a pause.
    pub fn advance_weather(&mut self, seconds: f32) {
        self.clouds.advance(seconds);
    }

    /// The weather now, part way to its target while it changes.
    pub fn weather(&self) -> Weather {
        self.clouds.weather()
    }

    pub fn stats(&self) -> FrameStats {
        let stats = self.main_work.stats();
        FrameStats {
            clusters: stats.clusters(),
            software_clusters: stats.software_clusters(),
            triangles: stats.triangles(),
            instances: self.scene.gpu_instance_count,
            source_triangles: self.scene.source_triangles,
            unique_triangles: self.scene.unique_triangles,
            streaming: self.streamer.stats(),
            particles: self.particles.stats(),
            shadow_clusters: self.aux_work.stats().clusters(),
            shadow_software_clusters: self.aux_work.stats().software_clusters(),
            shadow_triangles: self.aux_work.stats().triangles(),
            dropped: self.main_work.stats().dropped(),
            shadow_dropped: self.aux_work.stats().dropped(),
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
        // The sea first: whether there is one decides what the frame's textures are.
        if let Some((ocean, seconds)) = self.ocean_request.take() {
            match (ocean, &mut self.sea) {
                (None, _) => self.sea = None,
                (Some(ocean), Some(sea)) => sea.set_ocean(ocean, seconds),
                (Some(ocean), None) => {
                    self.sea = Some(Sea::new(gpu, ocean, self.settings.ocean_resolution))
                }
            }
        }
        let cells = self.settings.ocean_patch_cells;
        if self.sea.is_none() {
            self.water = None;
        } else if self.water.as_ref().is_none_or(|w| w.cells() != cells) {
            self.water = Some(WaterSurface::new(
                gpu,
                &self.environment_layout,
                &self.rasterizer,
                cells,
            ));
        }
        self.ensure_sized(gpu, render_size, output_size);
        self.time += delta_seconds;
        self.frame_seconds = delta_seconds.clamp(1.0 / 1000.0, 0.25);
        self.particles.resize(gpu, self.settings.particle_pool);
        self.particles
            .set_light_interval(self.settings.particle_light_interval);
        self.particles
            .set_ray_steps(self.settings.particle_ray_steps);
        // Particles take the frame's own time, so a host that pauses by passing 0 holds them
        // still, where the passes that filter over time need a step above zero.
        self.particles.prepare(
            gpu,
            camera.position,
            delta_seconds.clamp(0.0, 0.25),
            self.settings.particle_detail,
        );

        // Everything below draws what's installed from here on, with instances where they are
        // this frame.
        self.streamer.update(gpu, &self.scene);
        let changes = self.scene.update_instances(gpu);
        let streamed = self.streamer.should_refresh_cached_views();
        if streamed {
            self.shadows.mark_stale();
        }
        if changes.shadows_changed {
            self.shadows.mark_stale_now();
        }
        if streamed || changes.shadows_changed {
            if let Some(water) = &mut self.water {
                water.seabed.mark_stale();
            }
        }
        if streamed || changes.gi_changed {
            self.volume.mark_stale();
        }

        let aspect = render_size.0 as f32 / render_size.1 as f32;
        let parity = (self.frame_index & 1) as usize;

        self.write_frame_data(gpu, camera, environment, render_size);
        let shadow_work = self.shadows.prepare(
            gpu,
            &self.view_slots,
            SLOT_SHADOW,
            SLOT_MOVING_SHADOW,
            camera,
            aspect,
            environment.sun_direction,
            self.settings.shadow_lod_threshold,
            0.5 * render_size.1 as f32 * camera.projection(aspect).y_axis.y,
            &self.scene.instance_table.moving_spheres(),
        );
        let volume_work = self
            .volume
            .prepare(gpu, &self.view_slots, SLOT_VOXEL, camera.position);
        self.waterline.prepare(
            gpu,
            camera.position,
            delta_seconds,
            self.sea.is_some() && self.water.is_some(),
        );
        let seabed_due = self.water.as_mut().is_some_and(|water| {
            water.seabed.prepare(
                gpu,
                &self.view_slots,
                SLOT_SEABED,
                camera.position,
                self.sea_level,
                self.settings.shadow_lod_threshold,
            )
        });

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

        // Clouds come before anything lit: their shadows and their light in the sky feed it all.
        mark(&self.profiler, &mut encoder, section::CLOUDS);
        self.clouds.update(gpu, delta_seconds, camera.position);
        self.clouds
            .record_lighting(&mut encoder, &self.cloud_environment);

        // The sea's waves, raised by the weather's wind.
        mark(&self.profiler, &mut encoder, section::OCEAN);
        if let Some(sea) = &mut self.sea {
            let weather = self.clouds.weather();
            sea.update(gpu, &weather, delta_seconds, self.settings.ocean_resolution);
            sea.record(&mut encoder);
        }

        mark(&self.profiler, &mut encoder, section::SHADOWS);
        self.shadows.record(
            &mut encoder,
            &self.cull,
            &self.rasterizer,
            &self.scene,
            &self.aux_cull_scene,
            &self.aux_work,
            &self.aux_raster,
            SLOT_SHADOW,
            SLOT_MOVING_SHADOW,
            &shadow_work,
        );
        // The seabed and voxel views share this work, so the counters are read now, while
        // they are the last cascade's.
        if shadow_work.due.iter().any(|&due| due) {
            self.aux_work.capture_stats(&mut encoder);
        }

        if let Some(water) = self.water.as_ref().filter(|_| seabed_due) {
            water.seabed.record(
                &mut encoder,
                &self.cull,
                &self.scene,
                &self.aux_cull_scene,
                &self.aux_work,
                &self.aux_raster,
                SLOT_SEABED,
            );
        }
        // Where the sea stands and has stood, from the waves over the seabed, before the
        // surfaces it wets are resolved.
        mark(&self.profiler, &mut encoder, section::WATERLINE);
        if let (Some(sea), Some(water)) = (&self.sea, &self.water) {
            let waves = sea.waves();
            self.waterline.record(
                gpu,
                &mut encoder,
                SeaViews {
                    key: water.id() ^ (waves.generation() << 32),
                    surface: water.uniform(),
                    displacement: &waves.displacement.view,
                    sampler: water.wave_sampler(),
                    seabed: &water.seabed.view,
                },
            );
        }

        self.volume.record(
            &mut encoder,
            &self.cull,
            &self.scene,
            &self.aux_cull_scene,
            &self.aux_work,
            &self.environment,
            &self.light_shadows,
            SLOT_VOXEL,
            &volume_work,
            self.profiler.as_ref(),
        );

        let sized = self.sized.as_ref().expect("resources were just created");

        // Geometry: two occlusion phases into one visibility buffer, then its G-buffer.
        sized.visibility.clear(&mut encoder);
        for (phase, slot, cull_section, raster_section) in [
            (
                CullPhase::First,
                SLOT_MAIN_FIRST,
                section::CULL_FIRST,
                section::RASTER_FIRST,
            ),
            (
                CullPhase::Second,
                SLOT_MAIN_SECOND,
                section::CULL_SECOND,
                section::RASTER_SECOND,
            ),
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
        self.resolve.record(
            &mut encoder,
            &sized.resolve,
            self.waterline.resolve_group(),
            render_size,
        );

        // Lighting.
        mark(&self.profiler, &mut encoder, section::CLOUDS_ON_SCREEN);
        self.clouds
            .record_screen(&mut encoder, &sized.clouds, &self.cloud_environment, parity);

        mark(&self.profiler, &mut encoder, section::SCREEN_GI);
        self.screen_gi
            .record(&mut encoder, &sized.screen_gi, &self.environment, parity);
        mark(&self.profiler, &mut encoder, section::LIGHTING);
        if self.scene.light_count > 0 {
            self.light_tiles.record(
                &mut encoder,
                &sized.light_tiles_group,
                &self.environment,
                render_size,
            );
        }
        self.deferred.record(
            &mut encoder,
            &sized.deferred[parity],
            &self.environment,
            render_size,
        );

        // The sea's surface, over the lit scene.
        mark(&self.profiler, &mut encoder, section::WATER);
        if let (Some(sea), Some(water), Some(targets)) = (&self.sea, &mut self.water, &sized.water)
        {
            water.prepare(
                gpu,
                camera.position,
                self.sea_level,
                &sea.ocean(),
                sea.waves(),
                sea.slopes(),
            );
            water.record(
                gpu,
                &mut encoder,
                &self.frame_uniform,
                &self.environment,
                sea.waves(),
                targets,
                SceneViews {
                    lit: &sized.hdr.view,
                    depth: &sized.gbuffer.depth.view,
                    motion: &sized.gbuffer.motion.view,
                    depth_pyramid: sized.screen_gi.depth_pyramid_view(),
                    normals: &sized.gbuffer.normal.view,
                },
                &self.shadows,
            );
        }

        // Particles, over the lit scene and the water.
        mark(&self.profiler, &mut encoder, section::PARTICLES);
        self.particles.record_simulation(&mut encoder);
        self.particles.record_light(&mut encoder, &self.environment);
        mark(&self.profiler, &mut encoder, section::PARTICLE_DRAW);
        let debug = self.settings.debug_view;
        let scene_color = sized
            .water
            .as_ref()
            .map_or(&sized.hdr.view, |water| &water.color.view);
        self.particles.record_draw(
            &mut encoder,
            scene_color,
            &sized.particles,
            debug == DebugView::Particles,
        );

        // From the lit scene to the display.
        mark(&self.profiler, &mut encoder, section::POST);
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
            environment.daylight(),
            passthrough,
        );

        mark(&self.profiler, &mut encoder, section::END);
        if let Some(profiler) = &mut self.profiler {
            profiler.resolve(&mut encoder);
        }
        self.main_work.capture_stats(&mut encoder);
        self.particles.capture_stats(&mut encoder);
        self.streamer.capture_requests(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        self.main_work.map_stats();
        self.particles.map_stats();
        self.aux_work.map_stats();
        self.streamer.map_requests();
        if let Some(profiler) = &mut self.profiler {
            profiler.request_map();
        }
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    fn ensure_sized(&mut self, gpu: &Gpu, render_size: (u32, u32), output_size: (u32, u32)) {
        let unchanged = self.sized.as_ref().is_some_and(|s| {
            s.render_size == render_size
                && s.output_size == output_size
                && s.screen_gi.probe_rays() == self.settings.probe_rays
                && s.water.is_some() == self.sea.is_some()
        });
        if unchanged {
            return;
        }
        let visibility = VisibilityBuffer::new(gpu, &self.rasterizer, render_size);
        let hzb = Hzb::new(gpu, &visibility);
        let gbuffer = GBuffer::new(gpu, render_size);
        // Particles are drawn into whichever of this and the water's colour post reads.
        let hdr = Target::new(
            gpu,
            "hdr",
            render_size,
            HDR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
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
            self.settings.probe_rays,
            &gbuffer,
            [&history[1].view, &history[0].view],
        );
        let clouds = self.clouds.bindings(gpu, render_size, &gbuffer.depth.view);
        let light_tiles = LightTiles::buffer(gpu, render_size);
        let light_tiles_group =
            self.light_tiles
                .bind_group(gpu, &gbuffer.depth.view, &self.scene, &light_tiles);
        let deferred = self.deferred.bind_groups(
            gpu,
            &gbuffer,
            &screen_gi,
            &hdr.view,
            &self.shadows,
            &self.scene,
            &clouds,
            &light_tiles,
            &self.light_shadows.view,
        );
        let water = self
            .sea
            .is_some()
            .then(|| WaterTargets::new(gpu, render_size));
        let (scene_color, scene_depth, scene_motion) = match &water {
            Some(water) => (
                &water.color.view,
                &water.scene_depth.view,
                &water.motion.view,
            ),
            None => (&hdr.view, &gbuffer.depth.view, &gbuffer.motion.view),
        };
        let particles = self.particles.targets(gpu, render_size, scene_depth);
        let post = self.post.bindings(
            gpu,
            &self.frame_uniform,
            scene_color,
            scene_depth,
            scene_motion,
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
            gbuffer,
            cull_scene,
            raster,
            resolve,
            screen_gi,
            clouds,
            deferred,
            light_tiles_group,
            post,
            hdr,
            water,
            particles,
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
            night_sky: environment.night_sky,
            stars: environment.stars,
            moon_radiance: if environment.moon_disc.is_some() {
                environment.moon_radiance
            } else {
                0.0
            },
            star_pole: environment.star_pole,
            star_equinox: environment.star_equinox,
            frame_seconds: self.frame_seconds,
            sun_disc: environment.sun_disc.unwrap_or(environment.sun_direction),
            sea_level: if self.sea.is_some() {
                self.sea_level
            } else {
                NO_SEA
            },
            moon_disc: environment.moon_disc.unwrap_or(Vec3::NEG_Y),
            _pad2: 0.0,
            daylight_illuminance: environment.daylight(),
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
            eye_position: Vec3::ZERO,
            eye_lod_scale: 0.0,
            eye_near: 0.0,
            receiver_plane_count: 0,
            instance_filter: InstanceFilter::All as u32,
            stamp: 0,
            receiver_planes: [Vec4::ZERO; 6],
            eye_light: Vec3::Y,
            _pad2: 0.0,
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
