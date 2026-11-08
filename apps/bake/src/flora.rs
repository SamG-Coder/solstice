//! Procedural trees and grass placed over a scene file's terrain.
//!
//! The forest demo grows the same plants over its own ground; this does it for any scene with a
//! `terrain` line, so that a scene file can ask for a wood or a meadow in one line.

use glam::{Mat4, Vec3, Vec4};
use solstice_asset::{Instance, Material, SceneAsset};
use solstice_builder::{build_virtual_mesh, BuildOptions};
use solstice_procgen::foliage::bark::BARK_TILE_METRES;
use solstice_procgen::foliage::grass::Meadow;
use solstice_procgen::foliage::tree::Species;
use solstice_procgen::foliage::Rng;
use solstice_procgen::surface::Surface;
use std::f32::consts::{FRAC_PI_2, TAU};

/// Differently grown copies of each species, so that no two neighbours are the same tree.
const TREE_VARIANTS: u32 = 2;
/// Differently laid out patches of each breed of grass.
const GRASS_VARIANTS: u32 = 4;
/// How far above the sea the ground must be for anything to grow on it.
const DRY_LAND: f32 = 0.3;

/// The textures a scene has made so far, by name, so that each bark, leaf and blade atlas is
/// generated once however many lines ask for it.
#[derive(Default)]
pub struct Layers {
    names: Vec<String>,
}

impl Layers {
    /// The layer called `name`, made with `make` the first time it is asked for.
    pub fn layer(
        &mut self,
        scene: &mut SceneAsset,
        name: &str,
        make: impl FnOnce() -> Surface,
    ) -> i32 {
        // Layers made here follow whatever the imports and the terrain added, in the order they
        // were made, so a name's position among them says where its layer is.
        let first = scene.textures.len() - self.names.len();
        if let Some(index) = self.names.iter().position(|n| n == name) {
            return (first + index) as i32;
        }
        println!("  {name}");
        self.names.push(name.to_string());
        scene.textures.push(make().bake(scene.texture_size));
        scene.textures.len() as i32 - 1
    }
}

/// A rectangle of ground, from one corner to the opposite one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Rect {
    pub fn new(x0: f32, z0: f32, x1: f32, z1: f32) -> Self {
        Self {
            min: [x0.min(x1), z0.min(z1)],
            max: [x0.max(x1), z0.max(z1)],
        }
    }

    /// A point drawn evenly from the rectangle.
    pub fn sample(&self, rng: &mut Rng) -> (f32, f32) {
        (
            rng.range(self.min[0], self.max[0]),
            rng.range(self.min[1], self.max[1]),
        )
    }
}

/// The ground plants grow on: its height at any point, and the level of the sea around it.
#[derive(Clone, Copy)]
pub struct Ground<'a> {
    pub height: &'a dyn Fn(f32, f32) -> f32,
    pub sea_level: f32,
}

impl Ground<'_> {
    /// Whether anything can grow at this spot: above the sea by a margin.
    fn is_dry(&self, x: f32, z: f32) -> bool {
        (self.height)(x, z) >= self.sea_level + DRY_LAND
    }
}

/// A stand of trees to grow: how many, where, how far apart, and from which seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stand {
    pub count: u32,
    pub area: Rect,
    pub spacing: f32,
    pub seed: u32,
}

/// Trees of one species, thrown as darts over the stand's area so that no two stand closer
/// than its spacing, as competition for light keeps them, standing only on dry land. Returns
/// how many were placed.
pub fn add_trees(
    scene: &mut SceneAsset,
    layers: &mut Layers,
    species: &Species,
    name: &str,
    stand: Stand,
    ground: Ground,
) -> u32 {
    let Stand {
        count,
        area,
        spacing,
        seed,
    } = stand;
    let options = BuildOptions::default();
    let first_mesh = scene.meshes.len() as u32;
    for variant in 0..TREE_VARIANTS {
        let plant = species.grow(seed.wrapping_mul(TREE_VARIANTS) + variant);
        println!(
            "  {name} {variant}: {} wood and {} leaf triangles",
            plant.wood.triangle_count(),
            plant.leaves.triangle_count()
        );
        scene.meshes.push(build_virtual_mesh(&plant.wood, &options));
        scene
            .meshes
            .push(build_virtual_mesh(&plant.leaves, &options));
    }
    let bark = species.bark;
    let leaf = species.leaves.kind;
    let bark_layer = layers.layer(scene, bark.name(), || bark.surface(0));
    let leaf_layer = layers.layer(scene, leaf.name(), || leaf.atlas(0));
    scene.materials.push(Material {
        base_color: [1.0; 4],
        roughness: 1.0,
        uv_scale: 1.0 / BARK_TILE_METRES,
        normal_strength: 1.0,
        texture_layer: bark_layer,
        ..Default::default()
    });
    scene.materials.push(Material {
        base_color: [1.0; 4],
        roughness: 1.0,
        normal_strength: 1.0,
        texture_layer: leaf_layer,
        flags: Material::TWO_SIDED | Material::FOLIAGE,
        ..Default::default()
    });
    let wood_material = scene.materials.len() as u32 - 2;

    // Dart throwing: a candidate is kept when it stands clear of every tree so far. A grid of
    // cells one spacing wide keeps the check to a tree's neighbours, so a wood of tens of
    // thousands costs the same per tree as a copse.
    let mut rng = Rng::new(seed, 13);
    let cells_x = ((area.max[0] - area.min[0]) / spacing).ceil().max(1.0) as usize;
    let cells_z = ((area.max[1] - area.min[1]) / spacing).ceil().max(1.0) as usize;
    let mut grid: Vec<Vec<(f32, f32)>> = vec![Vec::new(); cells_x * cells_z];
    let cell_of = |x: f32, z: f32| {
        let i = (((x - area.min[0]) / spacing) as usize).min(cells_x - 1);
        let j = (((z - area.min[1]) / spacing) as usize).min(cells_z - 1);
        (i, j)
    };
    let mut placed = 0;
    for _ in 0..count.saturating_mul(20) {
        if placed >= count {
            break;
        }
        let (x, z) = area.sample(&mut rng);
        if !ground.is_dry(x, z) {
            continue;
        }
        let height = (ground.height)(x, z);
        let (i, j) = cell_of(x, z);
        let mut clear = true;
        'cells: for cj in j.saturating_sub(1)..(j + 2).min(cells_z) {
            for ci in i.saturating_sub(1)..(i + 2).min(cells_x) {
                for &(sx, sz) in &grid[cj * cells_x + ci] {
                    if (sx - x).powi(2) + (sz - z).powi(2) < spacing * spacing {
                        clear = false;
                        break 'cells;
                    }
                }
            }
        }
        if !clear {
            continue;
        }
        grid[j * cells_x + i].push((x, z));
        let variant = rng.index(TREE_VARIANTS as usize) as u32;
        let mesh = first_mesh + variant * 2;
        let transform = Mat4::from_translation(Vec3::new(x, height - 0.05, z))
            * Mat4::from_rotation_y(rng.unit() * TAU)
            * Mat4::from_scale(Vec3::splat(rng.range(0.8, 1.25)));
        scene.instances.push(Instance {
            mesh,
            material: wood_material,
            transform,
        });
        scene.instances.push(Instance {
            mesh: mesh + 1,
            material: wood_material + 1,
            transform,
        });
        placed += 1;
    }
    placed
}

/// Places a patch whose local square is `0..size` on the ground at its centre: sheared to the
/// ground's slope, so every blade still grows straight up and every root touches the ground, and
/// turned by a quarter turn at random.
fn patch_transform(
    centre: (f32, f32),
    size: f32,
    turn: u32,
    ground: &dyn Fn(f32, f32) -> f32,
) -> Mat4 {
    let (cx, cz) = centre;
    let half = size * 0.5;
    let h = ground(cx, cz);
    let slope_x = (ground(cx + half, cz) - ground(cx - half, cz)) / size;
    let slope_z = (ground(cx, cz + half) - ground(cx, cz - half)) / size;
    let shear = Mat4::from_cols(
        Vec4::new(1.0, slope_x, 0.0, 0.0),
        Vec4::Y,
        Vec4::new(0.0, slope_z, 1.0, 0.0),
        Vec4::W,
    );
    Mat4::from_translation(Vec3::new(cx, h, cz))
        * shear
        * Mat4::from_rotation_y(turn as f32 * FRAC_PI_2)
        * Mat4::from_translation(Vec3::new(-half, 0.0, -half))
}

/// Covers the dry land within `area` with patches of one breed of grass. Returns how many.
pub fn add_grass(
    scene: &mut SceneAsset,
    layers: &mut Layers,
    breed: &Meadow,
    name: &str,
    area: Rect,
    ground: Ground,
) -> u32 {
    let options = BuildOptions::default();
    let first_mesh = scene.meshes.len() as u32;
    for seed in 0..GRASS_VARIANTS {
        let mesh = Meadow {
            seed,
            ..breed.clone()
        }
        .mesh();
        println!("  {name} grass {seed}: {} triangles", mesh.triangle_count());
        scene.meshes.push(build_virtual_mesh(&mesh, &options));
    }
    let atlas = layers.layer(scene, &format!("{name} blades"), || breed.atlas(0));
    scene.materials.push(Material {
        base_color: [1.0; 4],
        roughness: 1.0,
        normal_strength: 1.0,
        texture_layer: atlas,
        flags: Material::TWO_SIDED | Material::FOLIAGE,
        ..Default::default()
    });
    let material = scene.materials.len() as u32 - 1;

    let size = breed.size;
    let columns = ((area.max[0] - area.min[0]) / size).ceil() as i32;
    let rows = ((area.max[1] - area.min[1]) / size).ceil() as i32;
    let mut placed = 0;
    for j in 0..rows {
        for i in 0..columns {
            let centre = (
                area.min[0] + (i as f32 + 0.5) * size,
                area.min[1] + (j as f32 + 0.5) * size,
            );
            if !ground.is_dry(centre.0, centre.1) {
                continue;
            }
            let variant = hash(i, j, 0) % GRASS_VARIANTS;
            let turn = hash(i, j, 1) % 4;
            scene.instances.push(Instance {
                mesh: first_mesh + variant,
                material,
                transform: patch_transform(centre, size, turn, ground.height),
            });
            placed += 1;
        }
    }
    placed
}

/// A small, fast, deterministic hash for laying out patches.
fn hash(x: i32, z: i32, channel: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (z as u32).wrapping_mul(0xd816_3841)
        ^ channel.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^ (h >> 12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solstice_procgen::surfaces;

    #[test]
    fn a_rect_takes_its_corners_in_any_order() {
        assert_eq!(
            Rect::new(5.0, -2.0, -5.0, 2.0),
            Rect::new(-5.0, 2.0, 5.0, -2.0)
        );
        let rect = Rect::new(5.0, -2.0, -5.0, 2.0);
        let mut rng = Rng::new(1, 1);
        for _ in 0..100 {
            let (x, z) = rect.sample(&mut rng);
            assert!((-5.0..=5.0).contains(&x) && (-2.0..=2.0).contains(&z));
        }
    }

    #[test]
    fn a_patch_follows_the_slope_and_keeps_its_blades_upright() {
        let ground = |x: f32, _: f32| x * 0.5 + 3.0;
        let transform = patch_transform((10.0, 0.0), 2.0, 0, &ground);
        // Every corner of the patch rests on the ground.
        for (u, v) in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)] {
            let corner = transform.transform_point3(Vec3::new(u, 0.0, v));
            assert!(
                (corner.y - ground(corner.x, corner.z)).abs() < 1.0e-4,
                "{corner} is off the ground"
            );
        }
        // A blade's tip is straight above its root.
        let root = transform.transform_point3(Vec3::new(0.5, 0.0, 0.5));
        let tip = transform.transform_point3(Vec3::new(0.5, 1.0, 0.5));
        assert!((tip - root - Vec3::Y).length() < 1.0e-5);
    }

    #[test]
    fn layers_are_made_once_and_found_after_other_textures() {
        let mut scene = SceneAsset {
            texture_size: 4,
            ..Default::default()
        };
        // Two layers an import left behind.
        scene.textures.push(surfaces::sand(0).bake(4));
        scene.textures.push(surfaces::sand(0).bake(4));
        let mut layers = Layers::default();
        let mut made = 0;
        let mut make = || {
            made += 1;
            surfaces::sand(made)
        };
        assert_eq!(layers.layer(&mut scene, "oak_bark", &mut make), 2);
        assert_eq!(layers.layer(&mut scene, "oak_leaves", &mut make), 3);
        assert_eq!(layers.layer(&mut scene, "oak_bark", &mut make), 2);
        assert_eq!(made, 2);
        assert_eq!(scene.textures.len(), 4);
    }
}
