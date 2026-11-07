//! A meadow at the edge of a wood, to show procedural foliage.

use crate::meshes;
use glam::{Mat4, Vec3};
use solstice_asset::scene::Viewpoint;
use solstice_asset::{Instance, Material, SceneAsset};
use solstice_builder::{build_virtual_mesh, BuildOptions};
use solstice_procgen::foliage::grass::{blade_atlas, Meadow};
use solstice_procgen::foliage::tree::{bark, leaf_atlas, Species};
use solstice_procgen::foliage::Rng;
use solstice_procgen::noise::fbm;
use solstice_procgen::surface::TILE_METRES;
use solstice_procgen::surfaces;

/// Edge of the ground, in metres.
const GROUND_SIZE: f32 = 160.0;
/// Edge of the square that is covered in grass, centred on the ground.
const MEADOW_SIZE: f32 = 70.0;
/// Different grass patches, laid out at random so no repeat shows.
const PATCH_VARIANTS: u32 = 4;
/// Different trees of each species.
const TREE_VARIANTS: u32 = 2;
const BUSH_VARIANTS: u32 = 2;

mod material {
    pub const GROUND: u32 = 0;
    pub const GRASS: u32 = 1;
    pub const BARK: u32 = 2;
    pub const LEAVES: u32 = 3;
}

/// Gently rolling ground.
fn ground_height(x: f32, z: f32) -> f32 {
    fbm(Vec3::new(x * 0.012, 0.0, z * 0.012), 4, 3) * 5.0
        + fbm(Vec3::new(x * 0.05, 1.0, z * 0.05), 3, 4) * 0.6
}

/// A small, fast, deterministic hash for placement.
fn hash(x: i32, z: i32, channel: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (z as u32).wrapping_mul(0xd816_3841)
        ^ channel.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^ (h >> 12)
}

/// Places a patch whose local square is `0..size` so it lies on the ground: sheared to the ground's
/// slope, so every blade still grows straight up and every root touches the ground, and turned by
/// a quarter turn at random.
fn patch_transform(corner_x: f32, corner_z: f32, size: f32, turn: u32) -> Mat4 {
    let half = size * 0.5;
    let (cx, cz) = (corner_x + half, corner_z + half);
    let h = ground_height(cx, cz);
    let step = half;
    let slope_x = (ground_height(cx + step, cz) - ground_height(cx - step, cz)) / (2.0 * step);
    let slope_z = (ground_height(cx, cz + step) - ground_height(cx, cz - step)) / (2.0 * step);
    let shear = Mat4::from_cols(
        glam::Vec4::new(1.0, slope_x, 0.0, 0.0),
        glam::Vec4::Y,
        glam::Vec4::new(0.0, slope_z, 1.0, 0.0),
        glam::Vec4::W,
    );
    Mat4::from_translation(Vec3::new(cx, h, cz))
        * shear
        * Mat4::from_rotation_y(turn as f32 * std::f32::consts::FRAC_PI_2)
        * Mat4::from_translation(Vec3::new(-half, 0.0, -half))
}

pub fn build(texture_size: u32) -> SceneAsset {
    let options = BuildOptions::default();
    let meadow = Meadow::default();

    println!("growing grass...");
    let mut meshes = vec![build_virtual_mesh(
        &meshes::terrain(GROUND_SIZE, 256, ground_height),
        &options,
    )];
    for seed in 0..PATCH_VARIANTS {
        let mesh = Meadow { seed, ..meadow }.mesh();
        println!("  patch {seed}: {} triangles", mesh.triangle_count());
        meshes.push(build_virtual_mesh(&mesh, &options));
    }

    // Each plant is two meshes, wood then leaves, placed together.
    println!("growing trees...");
    let first_tree = meshes.len() as u32;
    let mut plants = Vec::new();
    for seed in 0..TREE_VARIANTS {
        plants.push(Species::broadleaf().grow(seed));
    }
    let first_bush = first_tree + TREE_VARIANTS * 2;
    for seed in 0..BUSH_VARIANTS {
        plants.push(Species::bush().grow(seed));
    }
    for plant in &plants {
        println!(
            "  {} wood and {} leaf triangles",
            plant.wood.triangle_count(),
            plant.leaves.triangle_count()
        );
        meshes.push(build_virtual_mesh(&plant.wood, &options));
        meshes.push(build_virtual_mesh(&plant.leaves, &options));
    }

    println!("generating textures...");
    let textures = vec![
        surfaces::grass(0).bake(texture_size),
        blade_atlas(0).bake(texture_size),
        bark(0).bake(texture_size),
        leaf_atlas(0).bake(texture_size),
    ];
    let materials = vec![
        Material {
            base_color: [0.8, 0.8, 0.8, 1.0],
            roughness: 1.0,
            uv_scale: 1.0 / TILE_METRES,
            texture_layer: 0,
            macro_variation: 1.0,
            ..Default::default()
        },
        Material {
            base_color: [1.0; 4],
            roughness: 1.0,
            normal_strength: 1.0,
            texture_layer: 1,
            flags: Material::TWO_SIDED | Material::FOLIAGE,
            ..Default::default()
        },
        Material {
            base_color: [1.0; 4],
            roughness: 1.0,
            uv_scale: 1.0 / TILE_METRES,
            normal_strength: 1.0,
            texture_layer: 2,
            ..Default::default()
        },
        Material {
            base_color: [1.0; 4],
            roughness: 1.0,
            normal_strength: 1.0,
            texture_layer: 3,
            flags: Material::TWO_SIDED | Material::FOLIAGE,
            ..Default::default()
        },
    ];

    let mut instances = vec![Instance {
        mesh: 0,
        material: material::GROUND,
        transform: Mat4::IDENTITY,
    }];
    let patches = (MEADOW_SIZE / meadow.size) as i32;
    let start = -MEADOW_SIZE * 0.5;
    for j in 0..patches {
        for i in 0..patches {
            let variant = hash(i, j, 0) % PATCH_VARIANTS;
            let turn = hash(i, j, 1) % 4;
            let x = start + i as f32 * meadow.size;
            let z = start + j as f32 * meadow.size;
            instances.push(Instance {
                mesh: 1 + variant,
                material: material::GRASS,
                transform: patch_transform(x, z, meadow.size, turn),
            });
        }
    }
    println!("{} grass patches", instances.len() - 1);

    let place_plant = |instances: &mut Vec<Instance>, mesh: u32, x: f32, z: f32, k: u32| {
        let turn = (hash(x as i32, z as i32, k) % 1000) as f32 / 1000.0 * std::f32::consts::TAU;
        let size = 0.8 + (hash(x as i32, z as i32, k + 1) % 1000) as f32 / 1000.0 * 0.45;
        let transform = Mat4::from_translation(Vec3::new(x, ground_height(x, z) - 0.05, z))
            * Mat4::from_rotation_y(turn)
            * Mat4::from_scale(Vec3::splat(size));
        instances.push(Instance {
            mesh,
            material: material::BARK,
            transform,
        });
        instances.push(Instance {
            mesh: mesh + 1,
            material: material::LEAVES,
            transform,
        });
    };
    // A wood beyond the meadow, thickening with distance, a few trees standing alone in the
    // meadow, and bushes along the wood's edge. Trees are scattered by dart throwing, which keeps
    // each one its own space as competition for light does, without the rows of a jittered grid.
    let mut rng = Rng::new(7, 3);
    let wood_density = |x: f32, z: f32| ((x.abs().max(z.abs()) - 30.0) / 22.0).clamp(0.0, 1.0);
    let mut stands: Vec<(f32, f32)> = vec![(-9.0, -14.0), (13.0, -22.0), (-20.0, 4.0)];
    for _ in 0..40_000 {
        let (x, z) = (rng.range(-78.0, 78.0), rng.range(-78.0, 78.0));
        let density = wood_density(x, z);
        if density <= 0.0 || rng.unit() > density {
            continue;
        }
        // Closer together deeper in the wood.
        let spacing = 9.0 - 4.0 * density;
        if stands
            .iter()
            .all(|&(sx, sz)| (sx - x).powi(2) + (sz - z).powi(2) > spacing * spacing)
        {
            stands.push((x, z));
        }
    }
    for (index, &(x, z)) in stands.iter().enumerate() {
        let variant = index as u32 % TREE_VARIANTS;
        place_plant(&mut instances, first_tree + variant * 2, x, z, 9);
    }
    let trees = stands.len();
    let mut bushes = 0;
    for _ in 0..3_000 {
        let (x, z) = (rng.range(-60.0, 60.0), rng.range(-60.0, 60.0));
        // Thickest just outside the meadow, in the light at the wood's edge.
        let edge = x.abs().max(z.abs());
        let chance = (1.0 - ((edge - 34.0) / 7.0).abs()).max(0.0);
        if rng.unit() < chance * 0.12 {
            let variant = rng.index(BUSH_VARIANTS as usize) as u32;
            place_plant(&mut instances, first_bush + variant * 2, x, z, 11);
            bushes += 1;
        }
    }
    println!("{trees} trees and {bushes} bushes");

    let eye = Vec3::new(0.0, 0.0, 12.0);
    SceneAsset {
        meshes,
        materials,
        instances,
        texture_size,
        textures,
        viewpoint: Some(Viewpoint {
            position: [eye.x, ground_height(eye.x, eye.z) + 1.6, eye.z],
            yaw: 0.0,
            pitch: -0.12,
        }),
        ..Default::default()
    }
}
