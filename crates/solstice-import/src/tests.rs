//! Each test writes a small glTF file by hand, so every value checked is known exactly.

use super::*;
use gltf::image::{Data as ImageData, Format};
use solstice_asset::gpu::unpack_triangle;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

/// A scratch directory removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("solstice-import-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A closed unit cube, centred on the origin, with outward winding.
fn cube() -> (Vec<[f32; 3]>, Vec<u16>) {
    let positions = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            ]
        })
        .collect();
    let indices = vec![
        0, 2, 1, 1, 2, 3, // -Z
        4, 5, 6, 5, 7, 6, // +Z
        0, 1, 4, 1, 5, 4, // -Y
        2, 6, 3, 3, 6, 7, // +Y
        0, 4, 2, 2, 4, 6, // -X
        1, 3, 5, 3, 7, 5, // +X
    ];
    (positions, indices)
}

/// Writes `model.gltf` plus the cube's `model.bin`. `nodes` and `materials` are JSON, and
/// `primitive_extra` is added to the mesh's primitive. Accessor 0 is positions, 1 indices.
fn write_model(
    dir: &TempDir,
    nodes: &str,
    primitive_extra: &str,
    materials_and_images: &str,
) -> PathBuf {
    let (positions, indices) = cube();
    let mut bin = Vec::new();
    for p in &positions {
        for value in p {
            bin.extend_from_slice(&value.to_le_bytes());
        }
    }
    let positions_len = bin.len();
    for index in &indices {
        bin.extend_from_slice(&index.to_le_bytes());
    }
    std::fs::write(dir.0.join("model.bin"), &bin).unwrap();

    let mut json = String::new();
    write!(
        json,
        r#"{{
        "asset": {{"version": "2.0"}},
        "scene": 0,
        "scenes": [{{"nodes": [0]}}],
        "nodes": {nodes},
        "meshes": [{{"primitives": [{{
            "attributes": {{"POSITION": 0}},
            "indices": 1
            {primitive_extra}
        }}]}}],
        "buffers": [{{"uri": "model.bin", "byteLength": {total}}}],
        "bufferViews": [
            {{"buffer": 0, "byteOffset": 0, "byteLength": {positions_len}}},
            {{"buffer": 0, "byteOffset": {positions_len}, "byteLength": {indices_len}}}
        ],
        "accessors": [
            {{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
              "min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}},
            {{"bufferView": 1, "componentType": 5123, "count": {index_count}, "type": "SCALAR"}}
        ]
        {materials_and_images}
    }}"#,
        total = bin.len(),
        indices_len = bin.len() - positions_len,
        index_count = indices.len(),
    )
    .unwrap();
    let path = dir.0.join("model.gltf");
    std::fs::write(&path, json).unwrap();
    path
}

/// A 2x2 RGBA PNG holding the given pixels.
fn write_png(dir: &TempDir, name: &str, pixels: [[u8; 4]; 4]) {
    let file = std::fs::File::create(dir.0.join(name)).unwrap();
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 2, 2);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let data: Vec<u8> = pixels.into_iter().flatten().collect();
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&data)
        .unwrap();
}

fn import(path: &Path) -> Imported {
    import_gltf(
        path,
        &ImportOptions {
            texture_size: 4,
            ..Default::default()
        },
    )
    .unwrap()
}

/// Triangles of the finest level of a mesh, in object space.
fn finest_triangles(mesh: &VirtualMesh) -> Vec<[Vec3; 3]> {
    let mut result = Vec::new();
    for cluster in mesh.clusters.iter().filter(|c| c.lod_error == 0.0) {
        for t in 0..cluster.triangle_count() {
            let packed = mesh.triangles[(cluster.triangle_offset + t) as usize];
            result.push(unpack_triangle(packed).map(|local| {
                let data = mesh.vertices[cluster.vertex_offset as usize + local as usize].data;
                mesh.quant_origin
                    + Vec3::new(f32::from(data[0]), f32::from(data[1]), f32::from(data[2]))
                        * mesh.quant_step
            }));
        }
    }
    result
}

/// Outward-facing closed surfaces have positive volume.
fn signed_volume(triangles: &[[Vec3; 3]], transform: Mat4) -> f32 {
    triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|p| transform.transform_point3(p));
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

#[test]
fn imports_geometry_with_exact_triangle_count() {
    let dir = TempDir::new();
    let path = write_model(&dir, r#"[{"mesh": 0}]"#, "", "");
    let scene = import(&path).scene;
    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.instances.len(), 1);
    assert_eq!(scene.meshes[0].source_triangles, 12);
    assert!(
        (signed_volume(&finest_triangles(&scene.meshes[0]), Mat4::IDENTITY) - 1.0).abs() < 1.0e-3
    );
}

#[test]
fn node_transforms_compose_through_the_hierarchy() {
    let dir = TempDir::new();
    let nodes = r#"[
        {"translation": [10, 0, 0], "children": [1]},
        {"translation": [0, 5, 0], "scale": [2, 2, 2], "mesh": 0}
    ]"#;
    let scene = import(&write_model(&dir, nodes, "", "")).scene;
    let expected =
        Mat4::from_translation(Vec3::new(10.0, 5.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0));
    assert!(scene.instances[0].transform.abs_diff_eq(expected, 1.0e-5));
}

#[test]
fn repeated_placements_share_one_mesh() {
    let dir = TempDir::new();
    let nodes = r#"[
        {"children": [1, 2]},
        {"translation": [1, 0, 0], "mesh": 0},
        {"translation": [2, 0, 0], "mesh": 0}
    ]"#;
    let scene = import(&write_model(&dir, nodes, "", "")).scene;
    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.instances.len(), 2);
}

/// A mirrored transform turns a surface inside out, and back-face culling would then hide
/// all of its front faces. The import must compensate so it stays outward-facing.
#[test]
fn mirrored_instances_stay_outward_facing() {
    let dir = TempDir::new();
    let nodes = r#"[
        {"children": [1, 2]},
        {"mesh": 0},
        {"scale": [-1, 1, 1], "translation": [3, 0, 0], "mesh": 0}
    ]"#;
    let scene = import(&write_model(&dir, nodes, "", "")).scene;
    assert_eq!(
        scene.meshes.len(),
        2,
        "mirrored placements need their own winding"
    );
    for instance in &scene.instances {
        let triangles = finest_triangles(&scene.meshes[instance.mesh as usize]);
        let volume = signed_volume(&triangles, instance.transform);
        assert!(volume > 0.9, "instance is inside out: volume {volume}");
    }
}

#[test]
fn fit_size_scales_centres_and_grounds_the_model() {
    let dir = TempDir::new();
    let nodes = r#"[{"translation": [100, 50, -20], "scale": [10, 20, 10], "mesh": 0}]"#;
    let path = write_model(&dir, nodes, "", "");
    let options = ImportOptions {
        texture_size: 4,
        fit_size: Some(4.0),
        ground: false,
        skinned: false,
    };
    let scene = import_gltf(&path, &options).unwrap().scene;
    let transform = scene.instances[0].transform;

    let mut bounds = Bounds::default();
    for triangle in finest_triangles(&scene.meshes[0]) {
        for p in triangle {
            bounds.add_box(p, p, transform);
        }
    }
    assert!(
        (bounds.longest_side() - 4.0).abs() < 1.0e-3,
        "side {}",
        bounds.longest_side()
    );
    assert!(
        bounds.min.y.abs() < 1.0e-3,
        "stands on the ground, was at {}",
        bounds.min.y
    );
    assert!(bounds.center().x.abs() < 1.0e-3 && bounds.center().z.abs() < 1.0e-3);
}

#[test]
fn ground_plane_lies_under_the_model_and_faces_up() {
    let dir = TempDir::new();
    let path = write_model(&dir, r#"[{"translation": [0, 3, 0], "mesh": 0}]"#, "", "");
    let options = ImportOptions {
        texture_size: 4,
        fit_size: None,
        ground: true,
        skinned: false,
    };
    let scene = import_gltf(&path, &options).unwrap().scene;
    assert_eq!(scene.instances.len(), 2);
    let ground = &scene.instances[1];
    assert!(
        (ground.transform.w_axis.y - 2.5).abs() < 1.0e-4,
        "at the model's lowest point"
    );
    for [a, b, c] in finest_triangles(&scene.meshes[ground.mesh as usize]) {
        assert!((b - a).cross(c - a).y > 0.0);
    }
}

#[test]
fn a_model_authored_around_y_zero_keeps_its_buried_base() {
    assert_eq!(
        ground_height(-0.1, 1.0),
        0.0,
        "origin near the base is the authored ground"
    );
    assert_eq!(ground_height(0.0, 1.0), 0.0);
    assert_eq!(
        ground_height(2.0, 3.0),
        2.0,
        "floating above the origin: use the lowest point"
    );
    assert_eq!(
        ground_height(-1.0, 1.0),
        -1.0,
        "origin at the centre says nothing about the ground"
    );
    assert_eq!(ground_height(-3.0, -2.0), -3.0);
}

#[test]
fn ground_plane_uses_the_authored_ground() {
    let dir = TempDir::new();
    // Spans Y from -0.1 to 0.9, like a scan with its base slightly buried.
    let path = write_model(&dir, r#"[{"translation": [0, 0.4, 0], "mesh": 0}]"#, "", "");
    let options = ImportOptions {
        texture_size: 4,
        fit_size: None,
        ground: true,
        skinned: false,
    };
    let scene = import_gltf(&path, &options).unwrap().scene;
    assert!(scene.instances[1].transform.w_axis.y.abs() < 1.0e-4);
}

/// `--fit` and `--ground` must agree, or the model would sink into or hover over the floor.
#[test]
fn fit_and_ground_plane_agree_on_the_ground_height() {
    let dir = TempDir::new();
    let path = write_model(&dir, r#"[{"translation": [0, 0.4, 0], "mesh": 0}]"#, "", "");
    let options = ImportOptions {
        texture_size: 4,
        fit_size: Some(2.0),
        ground: true,
        skinned: false,
    };
    let scene = import_gltf(&path, &options).unwrap().scene;
    let (model, floor) = (scene.instances[0].transform, scene.instances[1].transform);
    assert!(floor.w_axis.y.abs() < 1.0e-4, "floor at {}", floor.w_axis.y);
    // The cube's lowest point is 0.1 below the authored ground, scaled up by two.
    let lowest = model.transform_point3(Vec3::new(0.0, -0.5, 0.0)).y;
    assert!((lowest + 0.2).abs() < 1.0e-4, "model's base at {lowest}");
}

#[test]
fn viewpoint_looks_at_the_model() {
    let dir = TempDir::new();
    let path = write_model(&dir, r#"[{"translation": [5, 2, -3], "mesh": 0}]"#, "", "");
    let viewpoint = import(&path).scene.viewpoint.unwrap();
    let position = Vec3::from(viewpoint.position);
    let (sy, cy) = viewpoint.yaw.sin_cos();
    let (sp, cp) = viewpoint.pitch.sin_cos();
    let forward = Vec3::new(-sy * cp, sp, -cy * cp);
    let to_model = (Vec3::new(5.0, 2.0, -3.0) - position).normalize();
    assert!(
        forward.dot(to_model) > 0.999,
        "camera looks along {forward}, model is along {to_model}"
    );
}

#[test]
fn material_factors_carry_over() {
    let dir = TempDir::new();
    let materials = r#", "materials": [{
        "pbrMetallicRoughness": {"baseColorFactor": [0.1, 0.2, 0.3, 1.0], "metallicFactor": 0.4, "roughnessFactor": 0.6},
        "emissiveFactor": [1.0, 0.5, 0.25]
    }]"#;
    let path = write_model(&dir, r#"[{"mesh": 0}]"#, r#", "material": 0"#, materials);
    let scene = import(&path).scene;
    let material = scene.materials[scene.instances[0].material as usize];
    assert_eq!(material.base_color, [0.1, 0.2, 0.3, 1.0]);
    assert_eq!(material.metallic, 0.4);
    assert_eq!(material.roughness, 0.6);
    assert_eq!(material.emissive, [1.0, 0.5, 0.25]);
    assert_eq!(material.texture_layer, -1, "no textures means no layer");
    assert!(scene.textures.is_empty());
}

/// Builds a model with textures of uniform colour, and returns the first texel of its layer.
fn textured_layer_texel(pixel_name: &str, pixel: [u8; 4], material: &str) -> (Vec<u8>, Vec<u8>) {
    let dir = TempDir::new();
    write_png(&dir, pixel_name, [pixel; 4]);
    let materials = format!(
        r#", "images": [{{"uri": "{pixel_name}"}}], "textures": [{{"source": 0}}], "materials": [{material}]"#
    );
    let path = write_model(&dir, r#"[{"mesh": 0}]"#, r#", "material": 0"#, &materials);
    let scene = import(&path).scene;
    assert_eq!(scene.texture_size, 4);
    let layer = &scene.textures[0];
    assert_eq!(
        layer.color_mips.len(),
        3,
        "a 4x4 texture has three mip levels"
    );
    let surface = [&layer.normal_mips[0][..2], &layer.material_mips[0][..2]].concat();
    (layer.color_mips[0][..4].to_vec(), surface)
}

#[test]
fn base_color_texture_reaches_the_color_channels_with_default_surface() {
    let material = r#"{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}"#;
    let (color, surface) = textured_layer_texel("c.png", [10, 20, 30, 255], material);
    assert_eq!(color, vec![10, 20, 30, 255]);
    assert_eq!(surface[0], 128, "flat normal X");
    assert_eq!(surface[1], 127, "flat normal Y");
    assert_eq!(
        surface[3], 255,
        "metallic multiplier leaves the factor unchanged"
    );
}

/// This is the classic import bug: with green the wrong way, all relief looks inverted.
#[test]
fn normal_map_green_channel_is_flipped() {
    let material = r#"{"normalTexture": {"index": 0, "scale": 0.5}}"#;
    let (_, surface) = textured_layer_texel("n.png", [100, 200, 255, 255], material);
    assert_eq!(surface[0], 100);
    assert_eq!(surface[1], 55);
}

#[test]
fn metallic_roughness_channels_map_to_the_surface_layer() {
    let material = r#"{"pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}}}"#;
    let (_, surface) = textured_layer_texel("mr.png", [0, 200, 77, 255], material);
    assert_eq!(surface[2], 100, "roughness from green, halved");
    assert_eq!(surface[3], 77, "metallic from blue");
}

#[test]
fn occlusion_map_becomes_cavity() {
    let material = r#"{"occlusionTexture": {"index": 0}}"#;
    let (color, _) = textured_layer_texel("o.png", [90, 0, 0, 255], material);
    assert_eq!(color[3], 90);
}

#[test]
fn large_images_are_averaged_when_shrunk() {
    // Alternating black and white must average to grey, not pick one of them.
    let image = ImageData {
        pixels: (0..16u32 * 16)
            .flat_map(|i| {
                let value = if (i % 16 + i / 16) % 2 == 0 { 0 } else { 255 };
                [value, value, value, 255]
            })
            .collect(),
        format: Format::R8G8B8A8,
        width: 16,
        height: 16,
    };
    let sampled = material_sample(&image);
    assert!((100..=155).contains(&sampled), "got {sampled}");
}

fn material_sample(image: &ImageData) -> u8 {
    material::sample(image, 0.3, 0.3, 4)[0]
}

#[test]
fn unsupported_features_are_reported_not_hidden() {
    let dir = TempDir::new();
    let materials = r#", "materials": [{"alphaMode": "BLEND", "doubleSided": true}]"#;
    let path = write_model(&dir, r#"[{"mesh": 0}]"#, r#", "material": 0"#, materials);
    let warnings = import(&path).warnings.join("\n");
    assert!(warnings.contains("Blend"), "{warnings}");
}

/// Writes a model with two images, each a flat 2x2 colour, and the given materials over them,
/// with the cube using material 0.
fn masked_model(dir: &TempDir, first: [u8; 4], second: [u8; 4], materials: &str) -> PathBuf {
    write_png(dir, "a.png", [first; 4]);
    write_png(dir, "b.png", [second; 4]);
    let images = r#""images": [{"uri": "a.png"}, {"uri": "b.png"}],
        "textures": [{"source": 0}, {"source": 1}]"#;
    let extras = format!(r#", {images}, "materials": [{materials}]"#);
    write_model(dir, r#"[{"mesh": 0}]"#, r#", "material": 0"#, &extras)
}

fn textured(texture: u32, extra: &str) -> String {
    format!(r#"{{"pbrMetallicRoughness": {{"baseColorTexture": {{"index": {texture}}}}}{extra}}}"#)
}

/// The alpha of the largest level of a layer's colour map.
fn first_alpha(scene: &SceneAsset, layer: usize) -> u8 {
    scene.textures[layer].color_mips[0][3]
}

#[test]
fn masked_materials_keep_coverage_at_the_cut_off() {
    let dir = TempDir::new();
    let material = textured(0, r#", "alphaMode": "MASK", "alphaCutoff": 0.25"#);
    // Alpha 64 is just above a quarter, so it must pass the renderer's test at 0.5.
    let path = masked_model(&dir, [10, 20, 30, 64], [0; 4], &material);
    let scene = import(&path).scene;
    assert_eq!(scene.materials[0].flags, Material::MASKED);
    assert!(first_alpha(&scene, 0) >= 128, "{}", first_alpha(&scene, 0));

    let dir = TempDir::new();
    let material = textured(0, r#", "alphaMode": "MASK", "alphaCutoff": 0.75"#);
    let path = masked_model(&dir, [10, 20, 30, 64], [0; 4], &material);
    assert!(first_alpha(&import(&path).scene, 0) < 128);
}

#[test]
fn a_tint_changes_colour_but_not_what_is_cut_out() {
    let dir = TempDir::new();
    let material = r#"{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
        "baseColorFactor": [0.2, 0.9, 0.1, 1.0]}, "alphaMode": "MASK"}"#;
    let path = masked_model(&dir, [200, 200, 200, 200], [0; 4], material);
    assert!(first_alpha(&import(&path).scene, 0) >= 128);
}

/// Converts every material of a model, used or not, and returns what that made.
fn convert_all(path: &Path, count: usize) -> (Vec<Material>, Vec<solstice_asset::TextureLayer>) {
    let (document, _, images) = gltf::import(path).unwrap();
    let mut table = material::MaterialTable::new(&document, &images, 4);
    for index in 0..count {
        table.slot(&document, Some(index), &mut Vec::new());
    }
    table.finish()
}

#[test]
fn opaque_and_masked_materials_on_one_atlas_share_a_layer() {
    let dir = TempDir::new();
    let opaque = textured(0, "");
    let masked = textured(0, r#", "alphaMode": "MASK", "alphaCutoff": 0.75"#);
    // 64 of 255 is below the cut-off, so coverage in the layer must read below 0.5.
    let materials = format!("{opaque}, {masked}");
    let path = masked_model(&dir, [1, 2, 3, 64], [0; 4], &materials);
    let (materials, layers) = convert_all(&path, 2);
    assert_eq!(layers.len(), 1, "they must share");
    assert_eq!(materials[0].flags & Material::MASKED, 0);
    assert_eq!(materials[1].flags, Material::MASKED);
    assert!(layers[0].color_mips[0][3] < 128, "the layer holds coverage");
}

#[test]
fn an_opaque_material_with_a_cavity_of_its_own_keeps_it() {
    let dir = TempDir::new();
    let opaque = textured(0, r#", "occlusionTexture": {"index": 1}"#);
    let masked = textured(0, r#", "alphaMode": "MASK""#);
    let materials = format!("{opaque}, {masked}");
    let path = masked_model(&dir, [1, 2, 3, 40], [99, 0, 0, 255], &materials);
    let (materials, layers) = convert_all(&path, 2);
    assert_eq!(layers.len(), 2, "a layer each");
    let layer = materials[0].texture_layer as usize;
    assert_eq!(layers[layer].color_mips[0][3], 99);
}

#[test]
fn a_factor_alpha_raises_the_threshold_in_the_texture() {
    let material = r#"{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
        "baseColorFactor": [1, 1, 1, 0.5]}, "alphaMode": "MASK", "alphaCutoff": 0.25}"#;
    // Multiplied by the factor, 127 of 255 falls just short of the cut-off and 128 reaches it.
    let dir = TempDir::new();
    let path = masked_model(&dir, [9, 9, 9, 127], [0; 4], material);
    assert!(first_alpha(&import(&path).scene, 0) < 128);
    let dir = TempDir::new();
    let path = masked_model(&dir, [9, 9, 9, 128], [0; 4], material);
    assert!(first_alpha(&import(&path).scene, 0) >= 128);
}

#[test]
fn a_tiny_cut_off_still_leaves_holes_where_alpha_is_zero() {
    let dir = TempDir::new();
    let material = textured(0, r#", "alphaMode": "MASK", "alphaCutoff": 0.001"#);
    let path = masked_model(&dir, [9, 9, 9, 0], [0; 4], &material);
    assert!(first_alpha(&import(&path).scene, 0) < 128);
}

#[test]
fn shrinking_a_mask_keeps_the_colour_of_its_strands() {
    // Half a strand, half a black hole, shrunk to one texel.
    let image = ImageData {
        pixels: vec![200, 100, 50, 255, 0, 0, 0, 0],
        format: Format::R8G8B8A8,
        width: 2,
        height: 2,
    };
    let image = ImageData {
        pixels: [image.pixels.clone(), image.pixels].concat(),
        ..image
    };
    assert!(material::sample_masked(&image, 0.5, 0.5, 1)[0] > 190);
}

#[test]
fn enlarging_a_mask_keeps_the_colour_of_its_strands() {
    // A column of strand and a column of black hole, read three quarters of the way to the hole's
    // side of the strand: mostly solid, so it passes, and it must not be darkened.
    let image = ImageData {
        pixels: [[200, 100, 50, 255, 0, 0, 0, 0]; 2].concat(),
        format: Format::R8G8B8A8,
        width: 2,
        height: 2,
    };
    let texel = material::sample_masked(&image, 0.375, 0.5, 4);
    assert!(texel[3] > 128, "{texel:?}");
    assert!(texel[0] > 190, "{texel:?}");
}

#[test]
fn an_occlusion_map_on_a_masked_material_is_reported_and_left_out() {
    let dir = TempDir::new();
    let material = textured(
        0,
        r#", "alphaMode": "MASK", "occlusionTexture": {"index": 1}"#,
    );
    let path = masked_model(&dir, [9, 9, 9, 255], [7, 0, 0, 255], &material);
    let imported = import(&path);
    assert_eq!(first_alpha(&imported.scene, 0), 255);
    assert!(imported.warnings.join("\n").contains("occlusion"));
}

#[test]
fn a_mask_with_nothing_to_cut_out_of_is_solid() {
    let dir = TempDir::new();
    let materials = r#", "materials": [{"alphaMode": "MASK"}]"#;
    let path = write_model(&dir, r#"[{"mesh": 0}]"#, r#", "material": 0"#, materials);
    assert_eq!(import(&path).scene.materials[0].flags, 0);
}

#[test]
fn a_mask_whose_factor_never_passes_is_left_out() {
    let dir = TempDir::new();
    let invisible =
        r#"{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 0.1]}, "alphaMode": "MASK"}"#;
    let materials = format!(r#", "materials": [{invisible}]"#);
    let path = write_model(&dir, r#"[{"mesh": 0}]"#, r#", "material": 0"#, &materials);
    let result = import_gltf(&path, &ImportOptions::default());
    assert!(matches!(result, Err(ImportError::NothingToDraw)));
}

#[test]
fn double_sided_materials_are_drawn_from_both_sides() {
    let dir = TempDir::new();
    let materials = r#", "materials": [{"doubleSided": true}, {}]"#;
    let path = write_model(&dir, r#"[{"mesh": 0}]"#, r#", "material": 0"#, materials);
    let scene = import(&path).scene;
    assert_eq!(scene.materials[0].flags, Material::TWO_SIDED);
}

#[test]
fn clear_errors_for_bad_input() {
    let dir = TempDir::new();
    let missing = dir.0.join("nope.gltf");
    assert!(matches!(
        import_gltf(&missing, &ImportOptions::default()),
        Err(ImportError::Gltf(_))
    ));

    let path = write_model(&dir, r#"[{"mesh": 0}]"#, "", "");
    let odd = ImportOptions {
        texture_size: 100,
        ..Default::default()
    };
    assert!(matches!(
        import_gltf(&path, &odd),
        Err(ImportError::InvalidOptions(_))
    ));

    let empty = write_model(&dir, "[{}]", "", "");
    assert!(matches!(
        import_gltf(&empty, &ImportOptions::default()),
        Err(ImportError::NothingToDraw)
    ));
}

#[test]
fn texture_transform_follows_the_extension_definition() {
    let transform = material::UvTransform {
        offset: Vec2::new(0.5, 0.25),
        rotation: std::f32::consts::FRAC_PI_2,
        scale: Vec2::new(2.0, 3.0),
    };
    // (1, 1) scales to (2, 3); rotating by +90 degrees in glTF's convention gives (3, -2).
    let result = transform.apply(Vec2::ONE);
    assert!(
        result.distance(Vec2::new(3.5, -1.75)) < 1.0e-5,
        "got {result}"
    );
    assert_eq!(
        material::UvTransform::IDENTITY.apply(Vec2::new(0.3, 0.7)),
        Vec2::new(0.3, 0.7)
    );
}

fn small_scene(meshes: usize, materials: usize, layers: usize) -> SceneAsset {
    SceneAsset {
        meshes: vec![VirtualMesh::default(); meshes],
        materials: (0..materials)
            .map(|i| Material {
                texture_layer: if i < layers { i as i32 } else { -1 },
                ..Default::default()
            })
            .collect(),
        instances: vec![Instance {
            mesh: meshes as u32 - 1,
            material: materials as u32 - 1,
            transform: Mat4::from_translation(Vec3::X),
        }],
        textures: vec![Default::default(); layers],
        texture_size: 4,
        ..Default::default()
    }
}

fn small_model(meshes: usize, materials: usize, layers: usize) -> Imported {
    let scene = small_scene(meshes, materials, layers);
    let piece = Instance {
        mesh: 0,
        material: 0,
        transform: Mat4::IDENTITY,
    };
    Imported {
        scene,
        parts: HashMap::from([("kit_wall_01".to_string(), vec![piece])]),
        warnings: Vec::new(),
    }
}

#[test]
fn appended_models_keep_their_references_valid() {
    let mut scene = small_scene(2, 2, 1);
    let placements = ModelPlacements {
        whole: vec![
            Mat4::from_translation(Vec3::Y),
            Mat4::from_translation(Vec3::Z),
        ],
        parts: Vec::new(),
    };
    append_model(&mut scene, small_model(3, 2, 2), &placements).unwrap();

    assert_eq!(scene.meshes.len(), 5, "shared, not copied per placement");
    assert_eq!(scene.materials.len(), 4);
    assert_eq!(scene.textures.len(), 3);
    assert_eq!(scene.instances.len(), 3);
    for instance in &scene.instances[1..] {
        assert_eq!(
            instance.mesh, 4,
            "the model's last mesh, after the two already present"
        );
        assert_eq!(instance.material, 3);
    }
    assert_eq!(
        scene.materials[2].texture_layer, 1,
        "layers shift by those already present"
    );
    assert_eq!(scene.materials[3].texture_layer, 2);
    assert_eq!(
        scene.materials[0].texture_layer, 0,
        "existing materials are untouched"
    );
    // Placement applies outside the model's own transform.
    let moved = scene.instances[2].transform.transform_point3(Vec3::ZERO);
    assert!(moved.distance(Vec3::new(1.0, 0.0, 1.0)) < 1.0e-6);
}

#[test]
fn parts_are_placed_by_name_without_their_prefix() {
    let mut scene = small_scene(2, 2, 1);
    let at = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
    let placements = ModelPlacements {
        whole: Vec::new(),
        parts: vec![("wall_01".to_string(), at), ("kit_wall_01".to_string(), at)],
    };
    append_model(&mut scene, small_model(3, 2, 2), &placements).unwrap();
    assert_eq!(scene.instances.len(), 3);
    assert_eq!(
        scene.instances[1].mesh, 2,
        "mesh 0 of the model, after the two already present"
    );
    assert!(scene.instances[1].transform.abs_diff_eq(at, 1.0e-6));
}

#[test]
fn an_unknown_part_is_an_error_that_names_it() {
    let mut scene = small_scene(2, 2, 1);
    let placements = ModelPlacements {
        whole: Vec::new(),
        parts: vec![("gate_99".to_string(), Mat4::IDENTITY)],
    };
    let error = append_model(&mut scene, small_model(1, 1, 0), &placements).unwrap_err();
    assert!(error.contains("gate_99"), "{error}");
}

#[test]
fn import_records_named_nodes_as_parts() {
    let dir = TempDir::new();
    let nodes = r#"[
        {"children": [1]},
        {"name": "kit_tower", "translation": [40, 0, 0], "mesh": 0}
    ]"#;
    let imported = import(&write_model(&dir, nodes, "", ""));
    let part = &imported.parts["kit_tower"];
    assert_eq!(part.len(), 1);
    assert_eq!(
        part[0].transform,
        Mat4::IDENTITY,
        "by its own origin, not its place in the file"
    );
    assert!(
        imported.scene.instances[0].transform.w_axis.x > 39.0,
        "the whole model keeps its layout"
    );
}
