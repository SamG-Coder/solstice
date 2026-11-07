//! End-to-end tests on a real GPU. They pass trivially, with a note, if none is available.

use glam::{Mat4, Vec2, Vec3};
use solstice_asset::{Instance, Material, SceneAsset};
use solstice_builder::{build_virtual_mesh, BuildOptions, Mesh, Vertex};
use solstice_render::{Camera, DebugView, Environment, Gpu, Renderer, RendererSettings};
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

impl Run {
    fn red(&self, x: u32, y: u32) -> u8 {
        self.pixels[((y * self.size + x) * 4) as usize]
    }
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

    let mut renderer = Renderer::new(gpu, scene, FORMAT, settings);
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
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
    });
    let view = target.create_view(&Default::default());

    // Enough frames for the depth history to exist and the stats readback to cycle.
    for _ in 0..8 {
        renderer.render(gpu, camera, &darkness(), &view, (size, size), 1.0 / 60.0);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

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

        // Silhouette radius in pixels, for a sphere of radius 1 under perspective.
        let angular = (1.0 / distance).asin();
        let half = SIZE as f32 * 0.5;
        let radius = angular.tan() / (camera.fov_y * 0.5).tan() * half;
        let inner = (radius * 0.95).min(half - 2.0);

        let mut holes = 0;
        for y in 0..SIZE {
            for x in 0..SIZE {
                let (dx, dy) = (x as f32 + 0.5 - half, y as f32 + 0.5 - half);
                if dx * dx + dy * dy < inner * inner && run.red(x, y) < 200 {
                    holes += 1;
                }
            }
        }
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
