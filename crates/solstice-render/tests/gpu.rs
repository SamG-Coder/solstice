//! End-to-end tests on a real GPU. They pass trivially, with a note, if none is available.

use glam::{Mat4, Vec2, Vec3};
use solstice_asset::{Instance, LoadedScene, Material, SceneAsset, TextureLayer};
use solstice_builder::{
    build_virtual_mesh, compress_textures, mip_chain, paginate, BuildOptions, Mesh, Vertex,
};
use solstice_render::{
    Camera, DebugView, Environment, Gpu, Ocean, Place, Quality, Renderer, RendererSettings,
    StreamRequest, StreamingBudget, TimeOfDay, Weather,
};
use std::sync::{Arc, Mutex};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn request_gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    match pollster::block_on(Gpu::request(
        &instance,
        None,
        wgpu::PowerPreference::HighPerformance,
    )) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipping GPU test: {error}");
            None
        }
    }
}

/// A unit icosphere: closed, so every pixel inside its silhouette must be covered.
fn sphere(subdivisions: u32) -> Mesh {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut positions: Vec<Vec3> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .map(|p| Vec3::from_array(p).normalize())
    .to_vec();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut midpoints = std::collections::HashMap::new();
        let mut next = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let mut midpoint = |i: u32, j: u32| {
                *midpoints.entry((i.min(j), i.max(j))).or_insert_with(|| {
                    positions.push((positions[i as usize] + positions[j as usize]).normalize());
                    positions.len() as u32 - 1
                })
            };
            let (ab, bc, ca) = (midpoint(a, b), midpoint(b, c), midpoint(c, a));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }
    Mesh {
        vertices: positions
            .iter()
            .map(|&p| Vertex {
                position: p,
                normal: p,
                uv: Vec2::new(p.x, p.y),
            })
            .collect(),
        indices: faces.into_iter().flatten().collect(),
    }
}

fn white_material() -> Material {
    Material {
        base_color: [1.0; 4],
        ..Default::default()
    }
}

fn instance(mesh: u32, transform: Mat4) -> Instance {
    Instance {
        mesh,
        material: 0,
        transform,
    }
}

/// No sun and no sky: the background is black, whatever the debug view shows is pure.
fn darkness() -> Environment {
    Environment {
        sun_illuminance: 0.0,
        sky_intensity: 0.0,
        ..Default::default()
    }
}

struct Run {
    pixels: Vec<u8>,
    size: u32,
    triangles: u32,
    clusters: u32,
    software_clusters: u32,
}

/// Renders a few frames, fails on any validation error, and reads back the result.
fn render(
    gpu: &Gpu,
    scene: &SceneAsset,
    camera: &Camera,
    settings: RendererSettings,
    size: u32,
) -> Run {
    let errors: Arc<Mutex<Vec<String>>> = Default::default();
    let sink = errors.clone();
    gpu.device
        .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            sink.lock().unwrap().push(error.to_string());
        }));

    let mut renderer = Renderer::new(gpu, &paginate(scene), FORMAT, settings);
    let target = test_target(gpu, size);
    let view = target.create_view(&Default::default());

    // Enough frames for the depth history to exist and the stats readback to cycle.
    for _ in 0..8 {
        renderer.render(gpu, camera, &darkness(), &view, (size, size), 1.0 / 60.0);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    let pixels = read_pixels(gpu, &target, size);
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "GPU errors: {}", errors.join("\n"));
    let stats = renderer.stats();
    Run {
        pixels,
        size,
        triangles: stats.triangles,
        clusters: stats.clusters,
        software_clusters: stats.software_clusters,
    }
}

fn test_target(gpu: &Gpu, size: u32) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn read_pixels(gpu: &Gpu, target: &wgpu::Texture, size: u32) -> Vec<u8> {
    let row = (size * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: u64::from(row) * u64::from(size),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let data = staging.slice(..).get_mapped_range().unwrap();
    let mut pixels = Vec::new();
    for line in data.chunks_exact(row as usize) {
        pixels.extend_from_slice(&line[..(size * 4) as usize]);
    }
    pixels
}

/// Uncovered pixels inside the silhouette of a unit sphere `distance` away, in a white-on-black
/// image of a sphere centred on the view.
fn holes_in_sphere(pixels: &[u8], size: u32, distance: f32, fov_y: f32) -> u32 {
    let angular = (1.0 / distance).asin();
    let half = size as f32 * 0.5;
    let radius = angular.tan() / (fov_y * 0.5).tan() * half;
    let inner = (radius * 0.95).min(half - 2.0);
    let mut holes = 0;
    for y in 0..size {
        for x in 0..size {
            let (dx, dy) = (x as f32 + 0.5 - half, y as f32 + 0.5 - half);
            if dx * dx + dy * dy < inner * inner && pixels[((y * size + x) * 4) as usize] < 200 {
                holes += 1;
            }
        }
    }
    holes
}

/// The sphere fills most of the view, mixing LOD levels, and both rasterizers draw parts of
/// it. A crack in the DAG or a seam between rasterizers would show the black background.
#[test]
fn closed_surface_has_no_holes() {
    let Some(gpu) = request_gpu() else { return };
    let mesh = build_virtual_mesh(&sphere(7), &BuildOptions::default());
    let scene = SceneAsset {
        meshes: vec![mesh],
        materials: vec![white_material()],
        instances: vec![instance(0, Mat4::IDENTITY)],
        ..Default::default()
    };
    const SIZE: u32 = 512;

    // From far to close, so that both rasterizers and many LOD levels are exercised.
    let mut used_software = false;
    let mut used_hardware = false;
    for distance in [12.0_f32, 5.0, 2.5, 1.6] {
        let camera = Camera {
            position: Vec3::new(0.0, 0.0, distance),
            fov_y: 50.0_f32.to_radians(),
            near: 0.05,
            ..Default::default()
        };
        // No temporal accumulation: jittered history would smooth pinholes out of sight.
        let settings = RendererSettings {
            debug_view: DebugView::Albedo,
            temporal_accumulation: false,
            ..Default::default()
        };
        let run = render(&gpu, &scene, &camera, settings, SIZE);
        assert!(run.clusters > 0, "nothing was drawn at distance {distance}");
        used_software |= run.software_clusters > 0;
        used_hardware |= run.clusters > run.software_clusters;

        let holes = holes_in_sphere(&run.pixels, SIZE, distance, camera.fov_y);
        assert_eq!(holes, 0, "{holes} uncovered pixels at distance {distance}");
    }
    assert!(
        used_software,
        "this test should have exercised the software rasterizer"
    );
    assert!(
        used_hardware,
        "this test should have exercised the hardware rasterizer"
    );
}

fn generous_budget() -> StreamingBudget {
    StreamingBudget {
        geometry_bytes: 1 << 40,
        texture_bytes: 1 << 40,
    }
}

/// One step of a streaming host: hands over a few pages and starts what's wanted, like a slow
/// network with pages arriving out of order.
fn pump(
    renderer: &mut Renderer,
    loaded: &LoadedScene,
    queue: &mut Vec<StreamRequest>,
    per_frame: usize,
) {
    queue.extend(renderer.stream_requests(16));
    for _ in 0..per_frame.min(queue.len()) {
        // Popping the newest first delivers children before their parents.
        let request = queue.pop().unwrap();
        renderer.deliver(request.id, requested_bytes(loaded, &request));
    }
}

/// What a request's bytes come to once unpacked, from a scene in memory.
fn requested_bytes(loaded: &LoadedScene, request: &StreamRequest) -> Vec<u8> {
    if request.deflated {
        return loaded.page(request.id).to_vec();
    }
    let layer = ((request.id & 0x7fff_ffff) >> 8) as usize;
    let mip = (request.id & 0xff) as usize;
    let upper = &loaded.upper_textures[layer];
    [
        upper.color_mips[mip].as_slice(),
        upper.normal_mips[mip].as_slice(),
        upper.material_mips[mip].as_slice(),
    ]
    .concat()
}

/// The scene starts with only its coarsest pages and fills in slowly. Every frame along the way
/// must cover the whole sphere, with detail only ever being added.
#[test]
fn a_loading_scene_never_shows_a_hole() {
    let Some(gpu) = request_gpu() else { return };
    let mesh = build_virtual_mesh(&sphere(7), &BuildOptions::default());
    let scene = SceneAsset {
        meshes: vec![mesh],
        materials: vec![white_material()],
        instances: vec![instance(0, Mat4::IDENTITY)],
        ..Default::default()
    };
    let loaded = paginate(&scene);
    assert!(loaded.scene.pages.len() > 8);
    const SIZE: u32 = 512;
    const DISTANCE: f32 = 1.6;
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, DISTANCE),
        fov_y: 50.0_f32.to_radians(),
        near: 0.05,
        ..Default::default()
    };
    let settings = RendererSettings {
        debug_view: DebugView::Albedo,
        temporal_accumulation: false,
        ..Default::default()
    };
    let mut renderer =
        Renderer::streaming(&gpu, &loaded.scene, FORMAT, settings, generous_budget());
    let target = test_target(&gpu, SIZE);
    let view = target.create_view(&Default::default());

    let mut queue = Vec::new();
    let mut first_triangles = None;
    let mut last_stats = renderer.stats();
    for frame in 0..240 {
        pump(&mut renderer, &loaded, &mut queue, 1);
        renderer.render(&gpu, &camera, &darkness(), &view, (SIZE, SIZE), 1.0 / 60.0);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        last_stats = renderer.stats();
        if frame % 4 == 0 {
            let pixels = read_pixels(&gpu, &target, SIZE);
            let holes = holes_in_sphere(&pixels, SIZE, DISTANCE, camera.fov_y);
            assert_eq!(
                holes, 0,
                "{holes} uncovered pixels at frame {frame}, with {} pages in",
                last_stats.streaming.installed
            );
        }
        if frame == 6 {
            first_triangles = Some(last_stats.triangles);
        }
    }
    let streaming = last_stats.streaming;
    assert!(streaming.installed > loaded.scene.pinned_pages);
    assert!(
        (streaming.installed as usize) < loaded.scene.pages.len(),
        "a view of one side shouldn't need every page"
    );
    assert!(last_stats.triangles > first_triangles.unwrap());
}

/// A pool too small for the scene: visiting one sphere then another must evict the first, stay
/// within the pool and leave both whole.
#[test]
fn a_small_pool_evicts_what_is_out_of_view_and_shows_no_holes() {
    let Some(gpu) = request_gpu() else { return };
    let mesh = build_virtual_mesh(&sphere(7), &BuildOptions::default());
    const APART: f32 = 100.0;
    let scene = SceneAsset {
        meshes: vec![mesh.clone(), mesh],
        materials: vec![white_material()],
        instances: vec![
            instance(0, Mat4::IDENTITY),
            instance(1, Mat4::from_translation(Vec3::X * APART)),
        ],
        ..Default::default()
    };
    let loaded = paginate(&scene);
    const SIZE: u32 = 256;
    const DISTANCE: f32 = 1.6;
    let settings = RendererSettings {
        debug_view: DebugView::Albedo,
        temporal_accumulation: false,
        ..Default::default()
    };

    // Room for the coarsest pages and a part of what one close view needs.
    let pinned = loaded.scene.pinned_pages;
    let slots = pinned + 10;
    assert!(slots < loaded.scene.pages.len() as u32);
    let budget = StreamingBudget {
        geometry_bytes: u64::from(slots) * solstice_asset::paged::PAGE_BYTES as u64,
        ..generous_budget()
    };
    let mut renderer = Renderer::streaming(&gpu, &loaded.scene, FORMAT, settings, budget);
    let target = test_target(&gpu, SIZE);
    let view = target.create_view(&Default::default());

    let mut queue = Vec::new();
    for (visit, x) in [0.0, APART, 0.0, APART].into_iter().enumerate() {
        let camera = Camera {
            position: Vec3::new(x, 0.0, DISTANCE),
            fov_y: 50.0_f32.to_radians(),
            near: 0.05,
            ..Default::default()
        };
        for frame in 0..90 {
            pump(&mut renderer, &loaded, &mut queue, 4);
            renderer.render(&gpu, &camera, &darkness(), &view, (SIZE, SIZE), 1.0 / 60.0);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let streaming = renderer.stats().streaming;
            assert!(streaming.installed <= slots, "the pool overflowed");
            if frame % 8 == 0 {
                let pixels = read_pixels(&gpu, &target, SIZE);
                let holes = holes_in_sphere(&pixels, SIZE, DISTANCE, camera.fov_y);
                assert_eq!(holes, 0, "visit {visit}, frame {frame}");
            }
        }
    }
    assert!(
        renderer.stats().streaming.evictions > 0,
        "the second sphere should have pushed out the first"
    );
}

/// A close view of a texture whose largest level alone is green. It must start out using the
/// red levels in the manifest, then fetch the green one, by itself.
#[test]
fn a_texture_sharpens_once_a_view_needs_it() {
    let Some(gpu) = request_gpu() else { return };
    const EDGE: u32 = 1024;
    let level = |edge: u32, texel: [u8; 4]| -> Vec<u8> {
        texel
            .iter()
            .copied()
            .cycle()
            .take((edge * edge) as usize * 4)
            .collect()
    };
    let flat = |edge: u32, value: u8| vec![value; (edge * edge) as usize * 2];
    let chain = (0..=EDGE.ilog2()).map(|m| EDGE >> m);
    let layer = TextureLayer {
        color_mips: chain
            .clone()
            .map(|edge| {
                let texel = if edge == EDGE {
                    [0, 255, 0, 255]
                } else {
                    [255, 0, 0, 255]
                };
                level(edge, texel)
            })
            .collect(),
        normal_mips: chain.clone().map(|edge| flat(edge, 128)).collect(),
        material_mips: chain.map(|edge| flat(edge, 128)).collect(),
    };
    let scene = SceneAsset {
        meshes: vec![build_virtual_mesh(&sphere(5), &BuildOptions::default())],
        materials: vec![Material {
            texture_layer: 0,
            uv_scale: 8.0,
            ..white_material()
        }],
        instances: vec![instance(0, Mat4::IDENTITY)],
        texture_size: EDGE,
        textures: vec![layer],
        ..Default::default()
    };
    let loaded = paginate(&scene);
    const SIZE: u32 = 256;
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 1.05),
        fov_y: 50.0_f32.to_radians(),
        near: 0.01,
        ..Default::default()
    };
    let settings = RendererSettings {
        debug_view: DebugView::Albedo,
        temporal_accumulation: false,
        ..Default::default()
    };
    let mut renderer =
        Renderer::streaming(&gpu, &loaded.scene, FORMAT, settings, generous_budget());
    let target = test_target(&gpu, SIZE);
    let view = target.create_view(&Default::default());

    let center = |pixels: &[u8]| {
        let at = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        [pixels[at], pixels[at + 1]]
    };
    renderer.render(&gpu, &camera, &darkness(), &view, (SIZE, SIZE), 1.0 / 60.0);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let [red, green] = center(&read_pixels(&gpu, &target, SIZE));
    assert!(red > 200 && green < 50, "starts red, not {red}, {green}");

    let mut queue = Vec::new();
    for _ in 0..60 {
        pump(&mut renderer, &loaded, &mut queue, 8);
        renderer.render(&gpu, &camera, &darkness(), &view, (SIZE, SIZE), 1.0 / 60.0);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
    let stats = renderer.stats().streaming;
    assert_eq!(stats.middle_layers, 1);
    assert_eq!(stats.top_layers, 1);
    let [red, green] = center(&read_pixels(&gpu, &target, SIZE));
    assert!(green > 200 && red < 50, "ends green, not {red}, {green}");
}

/// A wall hides a crowd of spheres. Occlusion culling must skip them, leaving far fewer
/// triangles than the same crowd with the wall out of the way.
#[test]
fn occlusion_culling_skips_hidden_geometry() {
    let Some(gpu) = request_gpu() else { return };
    let ball = build_virtual_mesh(&sphere(5), &BuildOptions::default());
    let wall = build_virtual_mesh(&sphere(3), &BuildOptions::default());

    let mut instances = vec![];
    for x in -5..=5 {
        for y in -5..=5 {
            let position = Vec3::new(x as f32 * 2.2, y as f32 * 2.2, -30.0);
            instances.push(instance(0, Mat4::from_translation(position)));
        }
    }
    let crowd_only = SceneAsset {
        meshes: vec![ball.clone(), wall.clone()],
        materials: vec![white_material()],
        instances: instances.clone(),
        ..Default::default()
    };
    // A huge flattened sphere right in front of the camera covers the whole view.
    instances.push(instance(
        1,
        Mat4::from_scale_rotation_translation(
            Vec3::new(200.0, 200.0, 1.0),
            Default::default(),
            Vec3::new(0.0, 0.0, 4.0),
        ),
    ));
    let with_wall = SceneAsset {
        meshes: vec![ball, wall],
        materials: vec![white_material()],
        instances,
        ..Default::default()
    };

    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 10.0),
        ..Default::default()
    };
    let settings = RendererSettings {
        debug_view: DebugView::Albedo,
        ..Default::default()
    };
    let open = render(&gpu, &crowd_only, &camera, settings, 256);
    let blocked = render(&gpu, &with_wall, &camera, settings, 256);
    assert!(
        open.triangles > 1000,
        "crowd drew only {} triangles",
        open.triangles
    );
    assert!(
        blocked.triangles * 5 < open.triangles,
        "hidden crowd still cost {} triangles, against {} when visible",
        blocked.triangles,
        open.triangles
    );
}

/// Collapsed layouts and hidden frames give canvases of a pixel or two. Nothing useful can be
/// drawn, but every pass must still be valid, or the errors never stop.
#[test]
fn tiny_outputs_render_without_errors() {
    let Some(gpu) = request_gpu() else { return };
    let mesh = build_virtual_mesh(&sphere(2), &BuildOptions::default());
    let scene = SceneAsset {
        meshes: vec![mesh],
        materials: vec![white_material()],
        instances: vec![instance(0, Mat4::IDENTITY)],
        ..Default::default()
    };
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 5.0),
        ..Default::default()
    };
    // `render` itself fails the test on any validation error.
    for size in [1, 2, 3, 17, 33] {
        render(&gpu, &scene, &camera, RendererSettings::default(), size);
    }
}

/// Every probe ray count has its own pipelines, and switching quality at run time picks
/// another, with targets of another size. Each must render, alone and after a switch.
#[test]
fn every_probe_ray_count_renders_without_errors() {
    let Some(gpu) = request_gpu() else { return };
    let mesh = build_virtual_mesh(&sphere(2), &BuildOptions::default());
    let scene = SceneAsset {
        meshes: vec![mesh],
        materials: vec![white_material()],
        instances: vec![instance(0, Mat4::IDENTITY)],
        ..Default::default()
    };
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 5.0),
        ..Default::default()
    };
    for quality in [Quality::Low, Quality::Medium, Quality::High] {
        let mut settings = RendererSettings::default();
        settings.apply_quality(quality);
        let run = render(&gpu, &scene, &camera, settings, 64);
        assert!(run.clusters > 0, "nothing was drawn at {quality:?}");
    }

    let errors: Arc<Mutex<Vec<String>>> = Default::default();
    let sink = errors.clone();
    gpu.device
        .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            sink.lock().unwrap().push(error.to_string());
        }));
    let mut renderer = Renderer::new(&gpu, &paginate(&scene), FORMAT, RendererSettings::default());
    let target = test_target(&gpu, 64);
    let view = target.create_view(&Default::default());
    for rays in [8, 6, 8] {
        renderer.settings_mut().probe_rays = rays;
        for _ in 0..3 {
            renderer.render(&gpu, &camera, &darkness(), &view, (64, 64), 1.0 / 60.0);
        }
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let errors = errors.lock().unwrap();
    assert!(
        errors.is_empty(),
        "GPU errors: {}",
        errors.join(
            "
"
        )
    );
}

/// The same flat texture must look the same stored raw, as blocks the GPU samples directly,
/// and as blocks unpacked for a GPU without BC. 1024 is large enough that unpacking also
/// drops the largest mip.
#[test]
fn every_texture_path_renders_the_same_colour() {
    let Some(mut gpu) = request_gpu() else { return };
    const EDGE: u32 = 1024;
    let n = EDGE as usize * EDGE as usize;
    let layer = TextureLayer {
        color_mips: mip_chain([200, 100, 50, 255].repeat(n), EDGE, 4),
        normal_mips: mip_chain([128, 128].repeat(n), EDGE, 2),
        material_mips: mip_chain([128, 255].repeat(n), EDGE, 2),
    };
    let mesh = build_virtual_mesh(&sphere(4), &BuildOptions::default());
    let mut scene = SceneAsset {
        meshes: vec![mesh],
        materials: vec![Material {
            base_color: [1.0; 4],
            texture_layer: 0,
            ..Default::default()
        }],
        instances: vec![instance(0, Mat4::IDENTITY)],
        texture_size: EDGE,
        textures: vec![layer],
        ..Default::default()
    };
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 3.0),
        ..Default::default()
    };
    let settings = RendererSettings {
        debug_view: DebugView::Albedo,
        temporal_accumulation: false,
        ..Default::default()
    };
    let centre = |run: &Run| {
        let at = ((run.size / 2 * run.size + run.size / 2) * 4) as usize;
        run.pixels[at..at + 3].to_vec()
    };

    let raw = centre(&render(&gpu, &scene, &camera, settings, 64));
    assert!(
        raw[0] > raw[1] && raw[1] > raw[2],
        "texture missing: {raw:?}"
    );

    compress_textures(&mut scene);
    let paths = [("blocks", true), ("unpacked", false)];
    let supported = gpu.block_compressed_textures;
    for (name, keep_blocks) in paths {
        if keep_blocks && !supported {
            continue;
        }
        gpu.block_compressed_textures = keep_blocks;
        let result = centre(&render(&gpu, &scene, &camera, settings, 64));
        for (channel, (a, b)) in raw.iter().zip(&result).enumerate() {
            assert!(
                a.abs_diff(*b) <= 2,
                "{name}: channel {channel} is {b}, raw is {a}"
            );
        }
    }
}

/// The centre pixel of a square whose triangles face away from the camera, `distance` away, in
/// the normals view, and whether software rasterization drew any of it.
fn facing_away(gpu: &Gpu, flags: u32, distance: f32) -> ([u8; 3], bool) {
    const CELLS: u32 = 8;
    let mut mesh = Mesh::default();
    for j in 0..=CELLS {
        for i in 0..=CELLS {
            let (x, y) = (i as f32 / CELLS as f32 - 0.5, j as f32 / CELLS as f32 - 0.5);
            mesh.vertices.push(Vertex {
                position: Vec3::new(x, y, 0.0),
                normal: Vec3::NEG_Z,
                uv: Vec2::new(x, y),
            });
        }
    }
    for j in 0..CELLS {
        for i in 0..CELLS {
            let a = j * (CELLS + 1) + i;
            let (b, c, d) = (a + 1, a + CELLS + 1, a + CELLS + 2);
            // Clockwise as seen from +Z, where the camera is.
            mesh.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let scene = SceneAsset {
        meshes: vec![build_virtual_mesh(&mesh, &BuildOptions::default())],
        materials: vec![Material {
            flags,
            ..white_material()
        }],
        instances: vec![instance(0, Mat4::IDENTITY)],
        ..Default::default()
    };
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, distance),
        ..Default::default()
    };
    let settings = RendererSettings {
        debug_view: DebugView::Normals,
        temporal_accumulation: false,
        ..Default::default()
    };
    let run = render(gpu, &scene, &camera, settings, 128);
    let at = ((run.size / 2 * run.size + run.size / 2) * 4) as usize;
    (
        [run.pixels[at], run.pixels[at + 1], run.pixels[at + 2]],
        run.software_clusters > 0,
    )
}

/// Back faces are culled, unless the material is two-sided, when they are drawn with their normal
/// turned to the viewer. Checked close up, where hardware rasterizes, and far away, where software
/// does.
#[test]
fn two_sided_materials_draw_their_back_faces() {
    let Some(gpu) = request_gpu() else { return };
    for (distance, software) in [(1.0, false), (20.0, true)] {
        let (one_sided, used_software) = facing_away(&gpu, 0, distance);
        assert_eq!(
            used_software, software,
            "at {distance} m the wrong rasterizer ran"
        );
        assert_eq!(
            one_sided,
            [0, 0, 0],
            "a back face was drawn at {distance} m"
        );

        let (two_sided, _) = facing_away(&gpu, Material::TWO_SIDED, distance);
        // Encoded as n * 0.5 + 0.5 and then sRGB, so a normal towards the camera, +Z, is
        // (187, 187, 255).
        assert!(
            two_sided[2] > 240
                && two_sided[0].abs_diff(187) < 20
                && two_sided[1].abs_diff(187) < 20,
            "at {distance} m the back face shows {two_sided:?}, not a normal facing the camera"
        );
    }
}

/// The world normal at the centre of a flat, camera-facing tile, encoded as red and green.
fn tile_normal(gpu: &Gpu, uv: impl Fn(f32, f32) -> Vec2, stored: [u8; 2]) -> (i32, i32) {
    const CELLS: u32 = 8;
    const EDGE: u32 = 16;
    let n = (EDGE * EDGE) as usize;
    let mut mesh = Mesh::default();
    for j in 0..=CELLS {
        for i in 0..=CELLS {
            let (x, y) = (
                i as f32 / CELLS as f32 * 4.0 - 2.0,
                j as f32 / CELLS as f32 * 4.0 - 2.0,
            );
            mesh.vertices.push(Vertex {
                position: Vec3::new(x, y, 0.0),
                normal: Vec3::Z,
                uv: uv(x, y),
            });
        }
    }
    for j in 0..CELLS {
        for i in 0..CELLS {
            let a = j * (CELLS + 1) + i;
            let (b, c, d) = (a + 1, a + CELLS + 1, a + CELLS + 2);
            mesh.indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    let scene = SceneAsset {
        meshes: vec![build_virtual_mesh(&mesh, &BuildOptions::default())],
        materials: vec![Material {
            base_color: [1.0; 4],
            texture_layer: 0,
            ..Default::default()
        }],
        instances: vec![instance(0, Mat4::IDENTITY)],
        texture_size: EDGE,
        textures: vec![TextureLayer {
            color_mips: mip_chain([200, 200, 200, 255].repeat(n), EDGE, 4),
            normal_mips: mip_chain(stored.repeat(n), EDGE, 2),
            material_mips: mip_chain([128, 255].repeat(n), EDGE, 2),
        }],
        ..Default::default()
    };
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 3.0),
        ..Default::default()
    };
    let settings = RendererSettings {
        debug_view: DebugView::Normals,
        temporal_accumulation: false,
        ..Default::default()
    };
    let run = render(gpu, &scene, &camera, settings, 64);
    let at = ((run.size / 2 * run.size + run.size / 2) * 4) as usize;
    (i32::from(run.pixels[at]), i32::from(run.pixels[at + 1]))
}

/// A stored normal leaning towards increasing U must lean that way in the world, or every bump in
/// every texture lights as a dent. Tested with UVs in both handednesses, which models use freely.
#[test]
fn normal_maps_lean_the_way_their_texture_says() {
    let Some(gpu) = request_gpu() else { return };
    const TOWARDS_U: [u8; 2] = [220, 128];
    const TOWARDS_V: [u8; 2] = [128, 220];
    const FLAT: [u8; 2] = [128, 128];

    let plain = |x: f32, y: f32| Vec2::new(x, y);
    let (flat_x, flat_y) = tile_normal(&gpu, plain, FLAT);
    let (x, y) = tile_normal(&gpu, plain, TOWARDS_U);
    assert!(
        x > flat_x + 20,
        "+U should lean +X: {x} against flat {flat_x}"
    );
    assert!(
        y.abs_diff(flat_y) < 20,
        "+U should not lean in Y: {y} against {flat_y}"
    );
    let (x, y) = tile_normal(&gpu, plain, TOWARDS_V);
    assert!(
        y > flat_y + 20,
        "+V should lean +Y: {y} against flat {flat_y}"
    );
    assert!(
        x.abs_diff(flat_x) < 20,
        "+V should not lean in X: {x} against {flat_x}"
    );

    // Mirrored, as a model's far side often is: increasing U now runs towards -X.
    let mirrored = |x: f32, y: f32| Vec2::new(-x, y);
    let (x, _) = tile_normal(&gpu, mirrored, TOWARDS_U);
    assert!(
        x < flat_x - 20,
        "mirrored +U should lean -X: {x} against flat {flat_x}"
    );
    let (_, y) = tile_normal(&gpu, mirrored, TOWARDS_V);
    assert!(
        y > flat_y + 20,
        "mirrored +V should still lean +Y: {y} against flat {flat_y}"
    );
}

/// Baking compresses every scene, including those with no textures to compress.
#[test]
fn compressed_scene_without_textures_renders() {
    let Some(gpu) = request_gpu() else { return };
    let mut scene = SceneAsset {
        meshes: vec![build_virtual_mesh(&sphere(3), &BuildOptions::default())],
        materials: vec![white_material()],
        instances: vec![instance(0, Mat4::IDENTITY)],
        texture_size: 1024,
        ..Default::default()
    };
    compress_textures(&mut scene);
    let camera = Camera {
        position: Vec3::new(0.0, 0.0, 3.0),
        ..Default::default()
    };
    let run = render(&gpu, &scene, &camera, RendererSettings::default(), 64);
    assert!(run.clusters > 0);
}

/// A white ball on a wide floor, lit by the sun from +X, for judging how clouds light a scene.
fn ball_on_floor() -> SceneAsset {
    let mut floor = Mesh::default();
    for (x, z) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        floor.vertices.push(Vertex {
            position: Vec3::new(x * 400.0, 0.0, z * 400.0),
            normal: Vec3::Y,
            uv: Vec2::new(x, z),
        });
    }
    floor.indices = vec![0, 2, 1, 0, 3, 2];
    let options = BuildOptions::default();
    SceneAsset {
        meshes: vec![
            build_virtual_mesh(&sphere(4), &options),
            build_virtual_mesh(&floor, &options),
        ],
        materials: vec![white_material()],
        instances: vec![
            instance(0, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))),
            instance(1, Mat4::IDENTITY),
        ],
        ..Default::default()
    }
}

/// Renders the ball on its floor under `weather`, after `seconds` of it, lit by a sun from +X,
/// and reads back the image.
fn render_weather(
    gpu: &Gpu,
    camera: &Camera,
    weather: Weather,
    seconds: f32,
    size: u32,
) -> Vec<u8> {
    let environment = Environment {
        sun_direction: Vec3::new(1.0, 0.7, 0.2).normalize(),
        ..Default::default()
    };
    render_lit(gpu, camera, weather, seconds, &environment, size, 60)
}

/// Renders the ball on its floor under `weather` and `environment` for `frames` frames, and
/// reads back the last.
fn render_lit(
    gpu: &Gpu,
    camera: &Camera,
    weather: Weather,
    seconds: f32,
    environment: &Environment,
    size: u32,
    frames: u32,
) -> Vec<u8> {
    let errors: Arc<Mutex<Vec<String>>> = Default::default();
    let sink = errors.clone();
    gpu.device
        .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            sink.lock().unwrap().push(error.to_string());
        }));
    // SOLSTICE_TEST_DEBUG=NAME shows a debug view instead, to see where a fault comes from.
    let mut settings = RendererSettings::default();
    if let Ok(name) = std::env::var("SOLSTICE_TEST_DEBUG") {
        settings.debug_view = DebugView::from_name(&name).expect("a debug view");
    }
    if std::env::var("SOLSTICE_TEST_NO_TAA").is_ok() {
        settings.temporal_accumulation = false;
    }
    let mut renderer = Renderer::new(gpu, &paginate(&ball_on_floor()), FORMAT, settings);
    renderer.set_weather(weather, 0.0);
    renderer.advance_weather(seconds);
    let target = test_target(gpu, size);
    let view = target.create_view(&Default::default());
    // Enough frames for the clouds' history and the exposure to settle.
    for _ in 0..frames {
        renderer.render(gpu, camera, environment, &view, (size, size), 1.0 / 60.0);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
    let pixels = read_pixels(gpu, &target, size);
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "GPU errors: {}", errors.join("\n"));
    // SOLSTICE_TEST_IMAGES=DIR keeps each image, to see what a failing test saw.
    if let Ok(dir) = std::env::var("SOLSTICE_TEST_IMAGES") {
        let name = format!(
            "{dir}/weather_{:.0}_{:.0}_{:.0}_{:.0}.ppm",
            camera.pitch * 100.0,
            camera.position.z,
            weather.coverage * 100.0,
            seconds
        );
        let mut ppm = format!("P6\n{size} {size}\n255\n").into_bytes();
        ppm.extend(pixels.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(name, ppm).unwrap();
    }
    pixels
}

/// Mean colour of a square of pixels centred at (x, y), `half` pixels either side.
fn patch(pixels: &[u8], size: u32, x: u32, y: u32, half: u32) -> [f32; 3] {
    let mut sum = [0.0; 3];
    let mut count = 0.0;
    for py in y - half..=y + half {
        for px in x - half..=x + half {
            let i = ((py * size + px) * 4) as usize;
            for c in 0..3 {
                sum[c] += pixels[i + c] as f32;
            }
            count += 1.0;
        }
    }
    sum.map(|s| s / count)
}

fn brightness(c: [f32; 3]) -> f32 {
    (c[0] + c[1] + c[2]) / 3.0
}

#[test]
fn clouds_fill_the_sky_and_grey_it() {
    let Some(gpu) = request_gpu() else { return };
    let camera = Camera {
        position: Vec3::new(0.0, 2.0, 0.0),
        pitch: 1.2,
        ..Default::default()
    };
    let sky = |name: &str| {
        let pixels = render_weather(&gpu, &camera, Weather::by_name(name).unwrap(), 0.0, 96);
        patch(&pixels, 96, 48, 30, 20)
    };
    let clear = sky("clear");
    let overcast = sky("overcast");
    // A clear sky is blue; an overcast one is grey.
    assert!(clear[2] > clear[0] * 1.3, "clear sky {clear:?}");
    assert!(
        overcast[2] < overcast[0] * 1.15,
        "overcast sky {overcast:?}"
    );
}

#[test]
fn overcast_light_is_soft_and_direct_sun_is_not() {
    let Some(gpu) = request_gpu() else { return };
    // Looking at the ball from the side, with the sun on its right: its shadow falls to the
    // left across the floor. Exposure adapts to each image, so compare the floor in the ball's
    // shadow with open floor, not either alone.
    let camera = Camera {
        position: Vec3::new(0.0, 1.0, 5.0),
        ..Default::default()
    };
    let contrast = |name: &str| {
        let pixels = render_weather(&gpu, &camera, Weather::by_name(name).unwrap(), 0.0, 128);
        let open = brightness(patch(&pixels, 128, 104, 87, 2));
        let shadow = brightness(patch(&pixels, 128, 22, 87, 2));
        open / shadow.max(1.0)
    };
    let clear = contrast("clear");
    let overcast = contrast("overcast");
    assert!(
        clear > 1.6,
        "open floor against the ball's shadow in clear weather: {clear}"
    );
    assert!(
        overcast < 1.25,
        "under overcast the shadow should all but vanish: {overcast}"
    );
}

#[test]
fn the_wind_carries_the_clouds_along() {
    let Some(gpu) = request_gpu() else { return };
    let camera = Camera {
        position: Vec3::new(0.0, 2.0, 0.0),
        pitch: 1.0,
        ..Default::default()
    };
    let weather = Weather::by_name("broken").unwrap();
    let before = render_weather(&gpu, &camera, weather, 0.0, 96);
    let after = render_weather(&gpu, &camera, weather, 300.0, 96);
    let changed = before
        .chunks_exact(4)
        .zip(after.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 12))
        .count();
    let share = changed as f32 / (96.0 * 96.0);
    assert!(
        share > 0.1,
        "only {share} of the sky changed in five minutes of wind"
    );
}

/// A camera at the ball's floor looking along `direction`, `fov_degrees` high.
fn looking_at(direction: Vec3, fov_degrees: f32) -> Camera {
    Camera {
        position: Vec3::new(0.0, 2.0, 0.0),
        yaw: (-direction.x).atan2(-direction.z),
        pitch: direction.y.asin(),
        fov_y: fov_degrees.to_radians(),
        ..Default::default()
    }
}

fn mean_brightness(pixels: &[u8]) -> f32 {
    let sum: f32 = pixels
        .chunks_exact(4)
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .sum();
    sum / (pixels.len() / 4) as f32
}

#[test]
fn twilight_fades_steadily_into_night() {
    let Some(gpu) = request_gpu() else { return };
    // Towards the sunset over a summer evening, from just before sunset into full night. The
    // scene may only ever grow darker: once it dipped below night in civil twilight.
    let place = Place::default();
    let daylight = Environment::default();
    let start = TimeOfDay {
        day: 172.0,
        hours: 19.0,
    };
    let camera = looking_at(
        start.sky(&place).sun * Vec3::new(1.0, 0.0, 1.0) + Vec3::Y * 0.1,
        60.0,
    );
    let clear = Weather::by_name("clear").unwrap();
    let mut previous = f32::MAX;
    for step in 0..12 {
        let time = TimeOfDay {
            hours: 19.0 + step as f32 * 0.25,
            ..start
        };
        let environment = time.environment(&place, &daylight);
        let pixels = render_lit(&gpu, &camera, clear, 0.0, &environment, 64, 240);
        let mean = mean_brightness(&pixels);
        assert!(
            mean <= previous * 1.04 + 1.0,
            "brightened from {previous} to {mean} at {:.2} h",
            time.hours
        );
        previous = mean;
    }
}

#[test]
fn the_moon_is_drawn_without_a_dark_halo() {
    let Some(gpu) = request_gpu() else { return };
    // A waxing gibbous moon at midnight in June, filling the middle of a narrow view.
    let place = Place::default();
    let time = TimeOfDay {
        day: 172.0,
        hours: 0.0,
    };
    let camera = looking_at(time.sky(&place).moon, 10.0);
    let environment = time.environment(&place, &Environment::default());
    let size = 192;
    let pixels = render_lit(
        &gpu,
        &camera,
        Weather::by_name("clear").unwrap(),
        0.0,
        &environment,
        size,
        120,
    );
    let at = |x: f32, y: f32| brightness(patch(&pixels, size, x as u32, y as u32, 0));
    let centre = size as f32 / 2.0;
    let moon = brightness(patch(&pixels, size, size / 2, size / 2, 2));
    // Pixels just outside the disc, which is about 8 pixels across its radius, and open sky
    // further out, all the way round.
    let (mut rim, mut sky) = (f32::MAX, 0.0_f32);
    for i in 0..32 {
        let angle = i as f32 / 32.0 * std::f32::consts::TAU;
        let (s, c) = angle.sin_cos();
        rim = rim.min(at(centre + c * 10.0, centre + s * 10.0));
        sky = sky.max(at(centre + c * 20.0, centre + s * 20.0));
    }
    assert!(
        moon > sky * 2.0,
        "the moon {moon} hardly stands out from the sky {sky}"
    );
    assert!(
        rim >= sky * 0.85 - 2.0,
        "a dark ring round the moon: {rim} against sky {sky}"
    );
}

#[test]
fn a_moonless_night_is_full_of_stars() {
    let Some(gpu) = request_gpu() else { return };
    // Early June, after the crescent moon has set: looking high into the southern sky.
    let place = Place::default();
    let time = TimeOfDay {
        day: 157.0,
        hours: 1.0,
    };
    let camera = looking_at(Vec3::new(0.0, 0.8, 0.6).normalize(), 55.0);
    let environment = time.environment(&place, &Environment::default());
    let size = 256;
    let pixels = render_lit(
        &gpu,
        &camera,
        Weather::by_name("clear").unwrap(),
        0.0,
        &environment,
        size,
        120,
    );
    let mut levels: Vec<f32> = pixels
        .chunks_exact(4)
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .collect();
    let stars = levels.iter().filter(|&&v| v > 0.0).count();
    levels.sort_by(f32::total_cmp);
    let background = levels[levels.len() / 2];
    let bright = levels.iter().filter(|&&v| v > background + 30.0).count();
    assert!(stars > 0);
    assert!(
        bright > 40,
        "only {bright} star pixels stand out from a sky of {background}"
    );
    assert!(
        background < 60.0,
        "the night sky is too bright for stars to show: {background}"
    );
}

/// The ball on its floor with an open sea at `sea_level`, seen through `debug` (the sea's
/// normals for coverage, or the final image), after enough frames for its waves to run.
fn render_sea(
    gpu: &Gpu,
    camera: &Camera,
    ocean: &str,
    sea_level: f32,
    debug: DebugView,
    quality: Quality,
    size: u32,
) -> Vec<u8> {
    let errors: Arc<Mutex<Vec<String>>> = Default::default();
    let sink = errors.clone();
    gpu.device
        .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            sink.lock().unwrap().push(error.to_string());
        }));
    let mut settings = RendererSettings::default();
    settings.apply_quality(quality);
    settings.debug_view = debug;
    // Temporal accumulation would average thin cracks away, hiding what these tests look for.
    settings.temporal_accumulation = false;
    let mut renderer = Renderer::new(gpu, &paginate(&ball_on_floor()), FORMAT, settings);
    renderer.set_ocean(Some(Ocean::by_name(ocean).unwrap()), 0.0);
    renderer.set_sea_level(sea_level);
    let environment = Environment {
        sun_direction: Vec3::new(1.0, 0.7, 0.2).normalize(),
        ..Default::default()
    };
    let target = test_target(gpu, size);
    let view = target.create_view(&Default::default());
    for _ in 0..30 {
        renderer.render(gpu, camera, &environment, &view, (size, size), 1.0 / 60.0);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
    let pixels = read_pixels(gpu, &target, size);
    let errors = errors.lock().unwrap();
    assert!(errors.is_empty(), "GPU errors: {}", errors.join("\n"));
    // SOLSTICE_TEST_IMAGES=DIR keeps each image, to see what a failing test saw.
    if let Ok(dir) = std::env::var("SOLSTICE_TEST_IMAGES") {
        let name = format!(
            "{dir}/sea_{:.0}_{:.0}_{:.0}_{debug:?}.ppm",
            camera.position.x, camera.position.y, camera.position.z
        );
        let mut ppm = format!(
            "P6
{size} {size}
255
"
        )
        .into_bytes();
        ppm.extend(pixels.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(name, ppm).unwrap();
    }
    pixels
}

/// Whether a pixel of the ocean debug view shows the sea: a normal leaning up, so green
/// high, and red and blue within a tilt of 0.75 of the middle, as steep wave faces have. The
/// sky and the white ball clip to white, which this is not.
fn shows_sea(pixel: &[u8]) -> bool {
    let near_middle = |c: u8| (32..=223).contains(&c);
    pixel[1] > 180 && near_middle(pixel[0]) && near_middle(pixel[2])
}

#[test]
fn the_sea_covers_everything_below_the_horizon() {
    let Some(gpu) = request_gpu() else { return };
    // The sea stands 5 m up. From just above the water, from a cliff, from far out at sea and from high above: every
    // pixel below the horizon shows sea, so the surface has no cracks or holes, whatever the
    // patches' sizes and however far they reach. The ball stands near the origin, so these
    // look away from it.
    for (size, quality, position, pitch) in [
        (128, Quality::High, Vec3::new(30.0, 6.5, 0.0), -0.15),
        (128, Quality::High, Vec3::new(40.0, 30.0, 3.0), -0.5),
        (128, Quality::Low, Vec3::new(5_000.0, 11.0, -3_000.0), -0.1),
        (128, Quality::Medium, Vec3::new(60.0, 400.0, 0.0), -1.2),
        // Finely, looking along the water where many levels meet within the view.
        (384, Quality::High, Vec3::new(12.0, 8.0, -5.0), -0.08),
    ] {
        let camera = Camera {
            position,
            yaw: -1.57,
            pitch,
            ..Default::default()
        };
        // The open ocean's swell, well above the floor, which its troughs must not reach.
        let pixels = render_sea(&gpu, &camera, "ocean", 5.0, DebugView::Ocean, quality, size);
        // Rows below the horizon, which sits at the screen row of the direction straight
        // out, ignoring the few rows the waves themselves raise and lower.
        let tan_half = (camera.fov_y * 0.5).tan();
        // Pitched down, the horizon is above the middle of the view.
        let horizon = ((0.5 - 0.5 * (-pitch).tan() / tan_half) * size as f32).max(0.0) as u32;
        let first = (horizon + 6).min(size);
        let mut missing = 0;
        for y in first..size {
            for x in 0..size {
                let i = ((y * size + x) * 4) as usize;
                if !shows_sea(&pixels[i..i + 4]) {
                    missing += 1;
                }
            }
        }
        let checked = (size - first) * size;
        assert!(
            checked > 1000,
            "{position}: only {checked} pixels below the horizon"
        );
        assert_eq!(
            missing, 0,
            "{position}, {quality:?}: {missing} of {checked} pixels show no sea"
        );
    }
}

#[test]
fn what_stands_out_of_the_sea_is_not_covered_by_it() {
    let Some(gpu) = request_gpu() else { return };
    // The ball's top stands out of calm water half way up it, a harbour's, without swell: its
    // upper half shows the ball, not water.
    let camera = Camera {
        position: Vec3::new(0.0, 1.6, 4.0),
        ..Default::default()
    };
    let size = 128;
    let pixels = render_sea(
        &gpu,
        &camera,
        "murky",
        1.0,
        DebugView::Ocean,
        Quality::High,
        size,
    );
    let at = |x: u32, y: u32| {
        let i = ((y * size + x) * 4) as usize;
        pixels[i..i + 4].to_vec()
    };
    // The ball's upper half, above the water, and the sea in front of it, low in the view.
    assert!(
        !shows_sea(&at(64, 52)),
        "the ball is drawn as water: {:?}",
        at(64, 52)
    );
    assert!(
        shows_sea(&at(64, 120)),
        "the sea in front is missing: {:?}",
        at(64, 120)
    );
}

#[test]
fn the_bed_shows_through_shallow_water_and_not_through_deep() {
    let Some(gpu) = request_gpu() else { return };
    // Looking down into a clear sea over the white floor, away from the ball: under 30 cm of
    // water the floor shows bright; under 30 m the water absorbs what it reflects, and the
    // sea is dark. Water that hid its bed at any depth, or showed it at every depth, fails.
    let size = 64;
    let brightness = |depth: f32| {
        let camera = Camera {
            position: Vec3::new(40.0, depth + 4.0, 0.0),
            yaw: -1.57,
            pitch: -1.3,
            ..Default::default()
        };
        let pixels = render_sea(
            &gpu,
            &camera,
            "ocean",
            depth,
            DebugView::Lit,
            Quality::High,
            size,
        );
        // The middle of the view, clear of the sun's highlight at the edges.
        let mut sum = 0.0;
        let mut count = 0.0;
        for y in size / 4..size * 3 / 4 {
            for x in size / 4..size * 3 / 4 {
                let i = ((y * size + x) * 4) as usize;
                sum += pixels[i..i + 3].iter().map(|&c| c as f32).sum::<f32>() / 3.0;
                count += 1.0;
            }
        }
        sum / count
    };
    let shallow = brightness(0.3);
    let deep = brightness(30.0);
    assert!(
        shallow > deep * 1.5 + 10.0,
        "the bed under 30 cm of water ({shallow}) is not brighter than 30 m of sea ({deep})"
    );
}

#[test]
fn waves_do_not_rise_through_low_land() {
    let Some(gpu) = request_gpu() else { return };
    // The floor stands just 20 cm above the open ocean, whose swell rises far higher. Waves die
    // as the water shoals, and over dry land there are none, so no crest ever shows through the
    // floor, however low it lies. Seen from high above, the view spans more than a whole
    // wavelength of the swell, so crests are always in it, and all of it is floor.
    let size = 128;
    let camera = Camera {
        position: Vec3::new(20.0, 250.0, 10.0),
        yaw: -1.57,
        pitch: -1.55,
        ..Default::default()
    };
    let pixels = render_sea(
        &gpu,
        &camera,
        "ocean",
        -0.2,
        DebugView::Ocean,
        Quality::High,
        size,
    );
    let wet = pixels.chunks_exact(4).filter(|p| shows_sea(p)).count();
    assert_eq!(wet, 0, "{wet} pixels of sea show through the floor");
}

/// The ball on its floor, both of `material`, with a calm sea at `sea_level` or none, seen
/// through the base-colour debug view: the albedo the G-buffer holds, wetness included.
fn render_albedo(gpu: &Gpu, material: Material, sea_level: Option<f32>) -> Vec<u8> {
    render_wet(
        gpu,
        material,
        sea_level,
        DebugView::Albedo,
        Vec3::new(0.0, 1.6, 4.0),
        0.0,
    )
}

/// The ball on its floor, both of `material`, with a calm sea at `sea_level` or none, seen
/// through `debug` from `position` pitched by `pitch`, under a low sun straight ahead.
fn render_wet(
    gpu: &Gpu,
    material: Material,
    sea_level: Option<f32>,
    debug: DebugView,
    position: Vec3,
    pitch: f32,
) -> Vec<u8> {
    let size = 128;
    let mut scene = ball_on_floor();
    scene.materials = vec![material];
    let settings = RendererSettings {
        debug_view: debug,
        temporal_accumulation: false,
        ..Default::default()
    };
    let mut renderer = Renderer::new(gpu, &paginate(&scene), FORMAT, settings);
    if let Some(level) = sea_level {
        renderer.set_ocean(Some(Ocean::by_name("murky").unwrap()), 0.0);
        renderer.set_sea_level(level);
    }
    let camera = Camera {
        position,
        pitch,
        ..Default::default()
    };
    let environment = Environment {
        sun_direction: Vec3::new(0.0, 0.35, -1.0).normalize(),
        ..Default::default()
    };
    let target = test_target(gpu, size);
    let view = target.create_view(&Default::default());
    for _ in 0..4 {
        renderer.render(gpu, &camera, &environment, &view, (size, size), 1.0 / 60.0);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
    read_pixels(gpu, &target, size)
}

fn brightness_at(pixels: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * 128 + x) * 4) as usize;
    pixels[i..i + 3].iter().map(|&c| c as f32).sum::<f32>() / 3.0
}

#[test]
fn water_darkens_porous_surfaces_but_not_metal_or_what_stands_above_it() {
    let Some(gpu) = request_gpu() else { return };
    // The sea lies 5 cm below the floor, so the floor is wet with it. The ball rises well
    // above the water, out of its reach.
    let (floor, ball) = ((64, 120), (64, 52));
    let sand = Material {
        base_color: [0.45, 0.38, 0.28, 1.0],
        roughness: 0.95,
        ..Default::default()
    };
    let dry = render_albedo(&gpu, sand, None);
    let wet = render_albedo(&gpu, sand, Some(-0.05));
    let (dry_floor, wet_floor) = (
        brightness_at(&dry, floor.0, floor.1),
        brightness_at(&wet, floor.0, floor.1),
    );
    // Wet sand reflects about half what it does dry; in the debug view's sRGB, darker by
    // well over a tenth.
    assert!(
        wet_floor < dry_floor * 0.85,
        "wet sand ({wet_floor}) is not darker than dry ({dry_floor})"
    );
    let (dry_ball, wet_ball) = (
        brightness_at(&dry, ball.0, ball.1),
        brightness_at(&wet, ball.0, ball.1),
    );
    assert!(
        (dry_ball - wet_ball).abs() < 2.0,
        "the ball's upper half, far above the water, changed: {dry_ball} dry, {wet_ball} wet"
    );

    // Metal has no pores to fill: water only films it, which leaves its colour alone.
    let metal = Material {
        base_color: [0.6, 0.55, 0.5, 1.0],
        roughness: 0.6,
        metallic: 1.0,
        ..Default::default()
    };
    let dry = brightness_at(&render_albedo(&gpu, metal, None), floor.0, floor.1);
    let wet = brightness_at(&render_albedo(&gpu, metal, Some(-0.05)), floor.0, floor.1);
    assert!(
        (dry - wet).abs() < 2.0,
        "wet metal changed colour: {dry} dry, {wet} wet"
    );
}

#[test]
fn wet_ground_shines_where_dry_ground_does_not() {
    let Some(gpu) = request_gpu() else { return };
    // A low sun straight ahead, mirrored by the floor a few metres in front of the camera, to
    // one side of the ball. Dry sand is too rough to show it as more than a faint sheen; wet,
    // a film of water turns it into a bright glint.
    let sand = Material {
        base_color: [0.45, 0.38, 0.28, 1.0],
        roughness: 0.95,
        ..Default::default()
    };
    let position = Vec3::new(10.0, 1.6, 4.0);
    let peak = |pixels: &[u8]| {
        (70..128)
            .flat_map(|y| (0..128).map(move |x| (x, y)))
            .map(|(x, y)| brightness_at(pixels, x, y))
            .fold(0.0, f32::max)
    };
    let dry = peak(&render_wet(
        &gpu,
        sand,
        None,
        DebugView::Lit,
        position,
        -0.25,
    ));
    let wet = peak(&render_wet(
        &gpu,
        sand,
        Some(-0.05),
        DebugView::Lit,
        position,
        -0.25,
    ));
    assert!(
        wet > dry * 1.3 + 10.0,
        "the wet floor's glint ({wet}) is no brighter than the dry floor's sheen ({dry})"
    );
}
