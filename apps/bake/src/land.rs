//! A coast's land in bands: sand along the water, a strip of dirt, then grass, with a meadow of
//! real grass on it and a wood beyond, as a scene file asks with `land`, `path`, `yard`,
//! `clear`, `meadow` and `wood` (see `scene_file.rs`).
//!
//! The bands follow how far inland a spot is from where the shore begins, so they keep to the
//! coast wherever it wanders, and each border meanders by its own noise, as a beach's edge and
//! a field's do. Paths and yards are dirt wherever they lie.
//!
//! The terrain is drawn as a mesh for each kind of ground. Each border is where a smooth field
//! crosses zero: how far past the sand's wandering edge a spot lies, and how far past the dirt's
//! or a path's. Where a border is near, a cell of the terrain is cut into smaller ones, and each
//! small triangle is cut along the line where the field crosses zero, as contour lines are
//! drawn, so a border is a smooth curve, not a row of teeth. The small cells' corners lie on
//! the big cell's own triangles, and a cut along an edge depends only on that edge's ends, so no
//! crack opens anywhere.
//!
//! Grass patches stand only where all of a patch is grass, leaving the ground's own short turf
//! along the edge of the dirt, as a mown verge; they sway in the wind. Trees and bushes are
//! scattered by dart throwing over grass, clear of paths, yards and the places the file keeps
//! clear, and sway too.

use glam::{Mat4, Vec2, Vec3};
use solstice_asset::{Instance, Material, SceneAsset};
use solstice_builder::{build_virtual_mesh, BuildOptions, Mesh, Vertex};
use solstice_procgen::foliage::bark::BARK_TILE_METRES;
use solstice_procgen::foliage::grass::Meadow;
use solstice_procgen::foliage::tree::Species;
use solstice_procgen::foliage::Rng;
use solstice_procgen::noise::fbm;
use solstice_procgen::surface::TILE_METRES;
use solstice_procgen::surfaces;
use std::collections::HashMap;

/// The kinds of ground, in the order of [`Plan::kind`]'s answers.
pub const SAND: usize = 0;
pub const DIRT: usize = 1;
pub const GRASS: usize = 2;
/// How many smaller cells a side a cell is cut into where kinds meet.
const FINE: u32 = 4;
/// How far each border wanders either way, in metres.
const SAND_WANDER: f32 = 4.0;
const DIRT_WANDER: f32 = 2.5;

/// What the scene file asks of the land.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// Whether the land is banded at all: a `land` line.
    pub banded: bool,
    /// How far inland from where the shore begins sand reaches, and the dirt beyond it, metres.
    pub sand: f32,
    pub dirt: f32,
    /// Dirt paths: from, to, width.
    pub paths: Vec<(Vec2, Vec2, f32)>,
    /// Dirt yards: opposite corners.
    pub yards: Vec<(Vec2, Vec2)>,
    /// Places kept clear of plants: centre and radius.
    pub clear: Vec<(Vec2, f32)>,
    /// A meadow of real grass on the grass within this distance of the origin, if any.
    pub meadow: Option<f32>,
    /// Woods: centre, radius, and the spacing between trees.
    pub woods: Vec<(Vec2, f32, f32)>,
}

/// The distance from `p` to the segment from `a` to `b`.
fn to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let along = b - a;
    let t = ((p - a).dot(along) / along.length_squared().max(1.0e-6)).clamp(0.0, 1.0);
    p.distance(a + along * t)
}

impl Plan {
    /// The two fields whose zeros are the borders at `p`, `inland` metres in from where the
    /// shore begins, in metres: past the sand's edge (sand below zero), and past the dirt's edge
    /// or a path's or a yard's (dirt below zero, where it is not sand).
    pub fn fields(&self, p: Vec2, inland: f32) -> (f32, f32) {
        let sand_edge =
            self.sand + fbm(Vec3::new(p.x * 0.03, 11.0, p.y * 0.03), 4, 703) * SAND_WANDER * 2.0;
        let dirt_edge = sand_edge
            + self.dirt
            + fbm(Vec3::new(p.x * 0.05, 17.0, p.y * 0.05), 4, 705) * DIRT_WANDER * 2.0;
        let wander = fbm(Vec3::new(p.x * 0.4, 31.0, p.y * 0.4), 3, 701) * 0.35;
        let mut dirt = inland - dirt_edge;
        for &(a, b, width) in &self.paths {
            dirt = dirt.min(to_segment(p, a, b) - width * 0.5 - wander);
        }
        for &(a, b) in &self.yards {
            let (low, high) = (a.min(b), a.max(b));
            let centre = (low + high) * 0.5;
            let half = (high - low) * 0.5;
            let d = (p - centre).abs() - half;
            let outside = d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0);
            dirt = dirt.min(outside - wander);
        }
        (inland - sand_edge, dirt)
    }

    /// The kind of ground at `p`, `inland` metres in from where the shore begins.
    pub fn kind(&self, p: Vec2, inland: f32) -> usize {
        kind_of(self.fields(p, inland))
    }

    /// Whether a plant may stand at `p`, `reach` metres across: on grass, not on a path or yard,
    /// nor anywhere kept clear.
    pub fn may_plant(&self, p: Vec2, reach: f32, kind_at: &dyn Fn(Vec2) -> usize) -> bool {
        [Vec2::ZERO, Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y]
            .iter()
            .all(|&d| kind_at(p + d * reach) == GRASS)
            && self.clear.iter().all(|&(c, r)| p.distance(c) > r + reach)
    }
}

/// The kind of ground for a pair of border fields.
fn kind_of((sand, dirt): (f32, f32)) -> usize {
    if sand < 0.0 {
        SAND
    } else if dirt < 0.0 {
        DIRT
    } else {
        GRASS
    }
}

/// A point of the terrain, with its normal and both border fields there.
#[derive(Clone, Copy, Debug)]
struct Point {
    p: Vec3,
    n: Vec3,
    fields: (f32, f32),
}

impl Point {
    fn lerp(self, other: Point, t: f32) -> Point {
        Point {
            p: self.p.lerp(other.p, t),
            n: self.n.lerp(other.n, t).normalize_or(Vec3::Y),
            fields: (
                self.fields.0 + (other.fields.0 - self.fields.0) * t,
                self.fields.1 + (other.fields.1 - self.fields.1) * t,
            ),
        }
    }
}

/// Cuts a convex polygon where `field` crosses zero: the part below it, and the part above.
fn split(polygon: &[Point], field: impl Fn(&Point) -> f32) -> (Vec<Point>, Vec<Point>) {
    let (mut below, mut above) = (Vec::new(), Vec::new());
    for (i, &a) in polygon.iter().enumerate() {
        let b = polygon[(i + 1) % polygon.len()];
        let (fa, fb) = (field(&a), field(&b));
        if fa < 0.0 {
            below.push(a);
        } else {
            above.push(a);
        }
        if (fa < 0.0) != (fb < 0.0) {
            let crossing = a.lerp(b, fa / (fa - fb));
            below.push(crossing);
            above.push(crossing);
        }
    }
    (below, above)
}

/// A mesh being built for one kind of ground, sharing vertices by position.
#[derive(Default)]
struct KindMesh {
    mesh: Mesh,
    at: HashMap<(i64, i64), u32>,
}

impl KindMesh {
    fn vertex(&mut self, p: Vec3, normal: Vec3) -> u32 {
        let key = ((p.x * 4096.0).round() as i64, (p.z * 4096.0).round() as i64);
        *self.at.entry(key).or_insert_with(|| {
            self.mesh.vertices.push(Vertex {
                position: p,
                normal,
                uv: Vec2::new(p.x, p.z),
            });
            self.mesh.vertices.len() as u32 - 1
        })
    }

    fn triangle(&mut self, corners: [(Vec3, Vec3); 3]) {
        let [a, b, c] = corners.map(|(p, n)| self.vertex(p, n));
        if a != b && b != c && a != c {
            self.mesh.indices.extend([a, b, c]);
        }
    }

    /// A convex polygon, as a fan of triangles.
    fn polygon(&mut self, points: &[Point]) {
        for i in 1..points.len().saturating_sub(1) {
            self.triangle([
                (points[0].p, points[0].n),
                (points[i].p, points[i].n),
                (points[i + 1].p, points[i + 1].n),
            ]);
        }
    }
}

/// Lays one small triangle down on the meshes of the kinds it covers, cut along its borders.
fn lay(meshes: &mut [KindMesh; 3], triangle: [Point; 3]) {
    let (sand, rest) = split(&triangle, |p| p.fields.0);
    let (dirt, grass) = split(&rest, |p| p.fields.1);
    meshes[SAND].polygon(&sand);
    meshes[DIRT].polygon(&dirt);
    meshes[GRASS].polygon(&grass);
}

/// The terrain `size` metres across in `cells` a side, a mesh for each kind of ground, from the
/// ground's `height` and the border `fields` at each spot (see [`Plan::fields`]).
pub fn zoned_terrain(
    size: f32,
    cells: u32,
    height: &(dyn Fn(f32, f32) -> f32 + Sync),
    fields: &(dyn Fn(Vec2) -> (f32, f32) + Sync),
) -> [Mesh; 3] {
    let step = size / cells as f32;
    let half = size * 0.5;
    let normal_at = |x: f32, z: f32| {
        let e = step * 0.5;
        Vec3::new(
            height(x - e, z) - height(x + e, z),
            2.0 * e,
            height(x, z - e) - height(x, z + e),
        )
        .normalize()
    };
    let mut meshes: [KindMesh; 3] = Default::default();
    for row in 0..cells {
        for column in 0..cells {
            let x0 = column as f32 * step - half;
            let z0 = row as f32 * step - half;
            let corner = |dx: f32, dz: f32| {
                let (x, z) = (x0 + dx * step, z0 + dz * step);
                Point {
                    p: Vec3::new(x, height(x, z), z),
                    n: normal_at(x, z),
                    fields: fields(Vec2::new(x, z)),
                }
            };
            // The cell's corners: a b on its near row, c d on its far one, as the plain
            // terrain lays them, split a-c-b, b-c-d.
            let (a, b, c, d) = (
                corner(0.0, 0.0),
                corner(1.0, 0.0),
                corner(0.0, 1.0),
                corner(1.0, 1.0),
            );
            let middle = fields(Vec2::new(x0 + step * 0.5, z0 + step * 0.5));
            // Whole, where no border comes within a cell's width.
            let far = |f: (f32, f32)| f.0.abs() > step * 1.5 && f.1.abs() > step * 1.5;
            let kinds = [a.fields, b.fields, c.fields, d.fields, middle].map(kind_of);
            if kinds.iter().all(|&k| k == kinds[0])
                && [a.fields, b.fields, c.fields, d.fields, middle]
                    .into_iter()
                    .all(far)
            {
                let k = kinds[0];
                meshes[k].triangle([(a.p, a.n), (c.p, c.n), (b.p, b.n)]);
                meshes[k].triangle([(b.p, b.n), (c.p, c.n), (d.p, d.n)]);
                continue;
            }
            // Where a border is near: smaller cells on the big cell's own two triangles, so that
            // the big cells around meet them without a crack, each cut along the border.
            let on_cell = |u: f32, v: f32| {
                let y = if u + v <= 1.0 {
                    a.p.y + (b.p.y - a.p.y) * u + (c.p.y - a.p.y) * v
                } else {
                    d.p.y + (c.p.y - d.p.y) * (1.0 - u) + (b.p.y - d.p.y) * (1.0 - v)
                };
                let (x, z) = (x0 + u * step, z0 + v * step);
                Point {
                    p: Vec3::new(x, y, z),
                    n: normal_at(x, z),
                    fields: fields(Vec2::new(x, z)),
                }
            };
            let small = 1.0 / FINE as f32;
            for j in 0..FINE {
                for i in 0..FINE {
                    let (u0, v0) = (i as f32 * small, j as f32 * small);
                    let (u1, v1) = (u0 + small, v0 + small);
                    let (p, q, r, s) = (
                        on_cell(u0, v0),
                        on_cell(u1, v0),
                        on_cell(u0, v1),
                        on_cell(u1, v1),
                    );
                    // The small cells' diagonals follow the big one's, so none crosses it.
                    lay(&mut meshes, [p, r, q]);
                    lay(&mut meshes, [q, r, s]);
                }
            }
        }
    }
    meshes.map(|m| m.mesh)
}

/// A texture layer for `surface`, added to the scene.
fn layer(scene: &mut SceneAsset, surface: solstice_procgen::surface::Surface) -> i32 {
    scene.textures.push(surface.bake(scene.texture_size));
    scene.textures.len() as i32 - 1
}

/// Adds the banded land to `scene`: its three kinds of ground, the meadow and the woods.
/// Returns the sand's texture layer, which water ripples are drawn from.
pub fn add(
    scene: &mut SceneAsset,
    plan: &Plan,
    size: f32,
    cells: u32,
    height: &(dyn Fn(f32, f32) -> f32 + Sync),
    inland: &(dyn Fn(f32, f32) -> f32 + Sync),
) -> i32 {
    let kind_at = |p: Vec2| plan.kind(p, inland(p.x, p.y));
    let fields = |p: Vec2| plan.fields(p, inland(p.x, p.y));
    println!("  zoning the land...");
    let meshes = zoned_terrain(size, cells, height, &fields);
    let sand = layer(scene, surfaces::sand(0));
    let dirt = layer(scene, surfaces::dirt(0));
    let grass = layer(scene, surfaces::grass(0));
    // Darkened as each is seen in the field, so they sit beside the scans: damp sand about 0.35
    // albedo, earth darker, turf darker still.
    for (mesh, (texture, tone)) in meshes
        .iter()
        .zip([(sand, 0.68), (dirt, 0.62), (grass, 0.75)])
    {
        if mesh.indices.is_empty() {
            continue;
        }
        println!("    {} triangles", mesh.indices.len() / 3);
        scene.materials.push(Material {
            base_color: [tone, tone, tone, 1.0],
            roughness: 1.0,
            uv_scale: 1.0 / TILE_METRES,
            normal_strength: 1.0,
            texture_layer: texture,
            macro_variation: 1.0,
            ..Default::default()
        });
        scene.instances.push(Instance {
            mesh: scene.meshes.len() as u32,
            material: scene.materials.len() as u32 - 1,
            transform: Mat4::IDENTITY,
        });
        scene
            .meshes
            .push(build_virtual_mesh(mesh, &BuildOptions::default()));
    }
    if let Some(radius) = plan.meadow {
        add_meadow(scene, plan, radius, height, &kind_at);
    }
    if !plan.woods.is_empty() {
        add_woods(scene, plan, height, &kind_at);
    }
    sand
}

/// Different grass patches, laid at random so no repeat shows.
const PATCH_VARIANTS: u32 = 4;

/// A small, fast hash for placement.
fn hash(x: i32, z: i32, channel: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (z as u32).wrapping_mul(0xd816_3841)
        ^ channel.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^ (h >> 12)
}

/// A patch whose local square is `0..size`, laid on the ground with its corner at (x, z),
/// sheared to the ground's slope so every blade grows straight up from it, and turned a
/// quarter turn at random.
fn patch_transform(x: f32, z: f32, size: f32, turn: u32, height: &dyn Fn(f32, f32) -> f32) -> Mat4 {
    let half = size * 0.5;
    let (cx, cz) = (x + half, z + half);
    let h = height(cx, cz);
    let slope_x = (height(cx + half, cz) - height(cx - half, cz)) / size;
    let slope_z = (height(cx, cz + half) - height(cx, cz - half)) / size;
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

/// Grass patches over the grass within `radius` of the origin, each wholly on grass.
fn add_meadow(
    scene: &mut SceneAsset,
    plan: &Plan,
    radius: f32,
    height: &dyn Fn(f32, f32) -> f32,
    kind_at: &dyn Fn(Vec2) -> usize,
) {
    println!("  growing the meadow...");
    let meadow = Meadow::default();
    let first_mesh = scene.meshes.len() as u32;
    for seed in 0..PATCH_VARIANTS {
        let mesh = Meadow {
            seed,
            ..meadow.clone()
        }
        .mesh();
        scene
            .meshes
            .push(build_virtual_mesh(&mesh, &BuildOptions::default()));
    }
    let blades = layer(scene, meadow.atlas(0));
    scene.materials.push(Material {
        base_color: [1.0; 4],
        roughness: 1.0,
        normal_strength: 1.0,
        texture_layer: blades,
        flags: Material::TWO_SIDED | Material::FOLIAGE | Material::SWAYS,
        ..Default::default()
    });
    let material = scene.materials.len() as u32 - 1;
    let size = meadow.size;
    let cells = (radius * 2.0 / size) as i32;
    let mut placed = 0;
    for j in 0..cells {
        for i in 0..cells {
            let (x, z) = (i as f32 * size - radius, j as f32 * size - radius);
            let centre = Vec2::new(x + size * 0.5, z + size * 0.5);
            if centre.length() > radius {
                continue;
            }
            // All of the patch, its corners and middle, on grass and clear.
            let whole = [(0.0, 0.0), (size, 0.0), (0.0, size), (size, size)]
                .iter()
                .all(|&(dx, dz)| kind_at(Vec2::new(x + dx, z + dz)) == GRASS)
                && plan.may_plant(centre, size * 0.5, kind_at);
            if !whole {
                continue;
            }
            let variant = hash(i, j, 0) % PATCH_VARIANTS;
            let turn = hash(i, j, 1) % 4;
            scene.instances.push(Instance {
                mesh: first_mesh + variant,
                material,
                transform: patch_transform(x, z, size, turn, height),
            });
            placed += 1;
        }
    }
    println!("    {placed} grass patches");
}

/// Trees and bushes over the grass of each wood, scattered by dart throwing: each its own space,
/// as competition for light keeps them, without a grid's rows.
fn add_woods(
    scene: &mut SceneAsset,
    plan: &Plan,
    height: &dyn Fn(f32, f32) -> f32,
    kind_at: &dyn Fn(Vec2) -> usize,
) {
    println!("  growing the woods...");
    // Each plant: its meshes, wood then leaves for each grown copy, and its two materials.
    let mut grown = Vec::new();
    for (name, variants) in [("oak", 2u32), ("birch", 2), ("bush", 2)] {
        let species = Species::by_name(name).expect("a built-in species");
        let first_mesh = scene.meshes.len() as u32;
        for seed in 0..variants {
            let plant = species.grow(seed);
            scene
                .meshes
                .push(build_virtual_mesh(&plant.wood, &BuildOptions::default()));
            scene
                .meshes
                .push(build_virtual_mesh(&plant.leaves, &BuildOptions::default()));
        }
        let bark = layer(scene, species.bark.surface(0));
        let leaf = layer(scene, species.leaves.kind.atlas(0));
        scene.materials.push(Material {
            base_color: [1.0; 4],
            roughness: 1.0,
            uv_scale: 1.0 / BARK_TILE_METRES,
            normal_strength: 1.0,
            texture_layer: bark,
            flags: Material::SWAYS | Material::TREE,
            ..Default::default()
        });
        scene.materials.push(Material {
            base_color: [1.0; 4],
            roughness: 1.0,
            normal_strength: 1.0,
            texture_layer: leaf,
            flags: Material::TWO_SIDED | Material::FOLIAGE | Material::SWAYS | Material::TREE,
            ..Default::default()
        });
        let leaves = scene.materials.len() as u32 - 1;
        grown.push((first_mesh, variants, leaves - 1, leaves));
    }
    let mut rng = Rng::new(11, 5);
    let mut trees: Vec<Vec2> = Vec::new();
    let mut bushes: Vec<Vec2> = Vec::new();
    let place = |scene: &mut SceneAsset, which: usize, at: Vec2, rng: &mut Rng| {
        let (first_mesh, variants, bark, leaves) = grown[which];
        let mesh = first_mesh + rng.index(variants as usize) as u32 * 2;
        let transform = Mat4::from_translation(Vec3::new(at.x, height(at.x, at.y) - 0.05, at.y))
            * Mat4::from_rotation_y(rng.range(0.0, std::f32::consts::TAU))
            * Mat4::from_scale(Vec3::splat(rng.range(0.8, 1.25)));
        for (mesh, material) in [(mesh, bark), (mesh + 1, leaves)] {
            scene.instances.push(Instance {
                mesh,
                material,
                transform,
            });
        }
    };
    for &(centre, radius, spacing) in &plan.woods {
        for _ in 0..(radius * radius * 2.0) as u32 {
            let at = centre
                + Vec2::from_angle(rng.range(0.0, std::f32::consts::TAU))
                    * radius
                    * rng.unit().sqrt();
            // Thicker towards the middle of the wood.
            let middle = 1.0 - at.distance(centre) / radius;
            let gap = spacing * (1.4 - 0.6 * middle);
            if trees.iter().any(|t| t.distance(at) < gap) || !plan.may_plant(at, 2.5, kind_at) {
                continue;
            }
            trees.push(at);
            let which = if rng.unit() < 0.65 { 0 } else { 1 };
            place(scene, which, at, &mut rng);
        }
        // Bushes along the wood's edge, where light reaches.
        for _ in 0..(radius * 6.0) as u32 {
            let at = centre
                + Vec2::from_angle(rng.range(0.0, std::f32::consts::TAU))
                    * radius
                    * rng.range(0.75, 1.1);
            if trees.iter().any(|t| t.distance(at) < 2.5)
                || bushes.iter().any(|b| b.distance(at) < 1.6)
                || !plan.may_plant(at, 1.0, kind_at)
            {
                continue;
            }
            bushes.push(at);
            place(scene, 2, at, &mut rng);
        }
    }
    println!("    {} trees and {} bushes", trees.len(), bushes.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Plan {
        Plan {
            banded: true,
            sand: 10.0,
            dirt: 6.0,
            paths: vec![(Vec2::new(0.0, 0.0), Vec2::new(30.0, 0.0), 3.0)],
            yards: vec![(Vec2::new(-5.0, -5.0), Vec2::new(-2.0, -2.0))],
            clear: vec![(Vec2::new(50.0, 50.0), 10.0)],
            meadow: Some(40.0),
            woods: Vec::new(),
        }
    }

    #[test]
    fn sand_then_dirt_then_grass_inland() {
        let plan = plan();
        // Far from paths and yards, along a line straight inland.
        let at = Vec2::new(100.0, 100.0);
        assert_eq!(plan.kind(at, -5.0), SAND);
        assert_eq!(plan.kind(at, 60.0), GRASS);
        let mut last = SAND;
        for step in 0..200 {
            let k = plan.kind(at, step as f32 * 0.25 - 5.0);
            assert!(k >= last, "the bands go sand, dirt, grass, and never back");
            last = k;
        }
        // Somewhere along the way there is dirt.
        assert!((0..200).any(|s| plan.kind(at, s as f32 * 0.25) == DIRT));
    }

    #[test]
    fn paths_and_yards_are_dirt_wherever_they_lie() {
        let plan = plan();
        assert_eq!(plan.kind(Vec2::new(15.0, 0.0), 80.0), DIRT);
        assert_eq!(plan.kind(Vec2::new(-3.5, -3.5), 80.0), DIRT);
        assert_eq!(plan.kind(Vec2::new(15.0, 6.0), 80.0), GRASS);
    }

    #[test]
    fn plants_keep_off_paths_and_clear_places() {
        let plan = plan();
        let kind_at = |p: Vec2| plan.kind(p, 80.0);
        assert!(plan.may_plant(Vec2::new(15.0, 8.0), 1.0, &kind_at));
        assert!(!plan.may_plant(Vec2::new(15.0, 2.0), 1.0, &kind_at));
        assert!(!plan.may_plant(Vec2::new(52.0, 52.0), 1.0, &kind_at));
    }

    #[test]
    fn the_zoned_terrain_has_no_cracks_and_its_borders_are_fine() {
        let height = |x: f32, z: f32| (x * 0.3).sin() + (z * 0.2).cos();
        // A sand border running diagonally across the square, and a curved dirt one.
        let fields = |p: Vec2| (p.x + 0.5 * p.y - 0.3, p.length() - 6.0);
        let meshes = zoned_terrain(20.0, 20, &height, &fields);
        // Every triangle's area adds up to the square's, so nothing overlaps and nothing is
        // missing: across the whole terrain, the kinds tile it.
        let area: f32 = meshes
            .iter()
            .map(|m| {
                m.indices
                    .chunks_exact(3)
                    .map(|t| {
                        let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.vertices[i as usize].position);
                        let (a, b, c) = (
                            Vec2::new(a.x, a.z),
                            Vec2::new(b.x, b.z),
                            Vec2::new(c.x, c.z),
                        );
                        ((b - a).perp_dot(c - a) * 0.5).abs()
                    })
                    .sum::<f32>()
            })
            .sum();
        assert!((area - 400.0).abs() < 0.01, "{area}");
        // Every triangle faces up.
        for m in &meshes {
            for t in m.indices.chunks_exact(3) {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.vertices[i as usize].position);
                assert!((b - a).cross(c - a).y > 0.0);
            }
        }
        // Each vertex where small cells meet a whole cell's edge lies on that edge's straight
        // line: every vertex lies on the plain terrain's triangles. Checked by the grass and sand
        // meshes sharing their border vertices at the same heights.
        let at = |m: &Mesh| -> HashMap<(i64, i64), f32> {
            m.vertices
                .iter()
                .map(|v| {
                    (
                        (
                            (v.position.x * 1024.0).round() as i64,
                            (v.position.z * 1024.0).round() as i64,
                        ),
                        v.position.y,
                    )
                })
                .collect()
        };
        let (sand, grass) = (at(&meshes[SAND]), at(&meshes[GRASS]));
        let mut shared = 0;
        for (key, y) in &sand {
            if let Some(other) = grass.get(key) {
                assert!((y - other).abs() < 1.0e-4);
                shared += 1;
            }
        }
        assert!(shared > 20);
        // The borders are cut along the fields: every vertex of the sand lies on its side of the
        // straight line, and the grass's outside the circle, to within the chords that draw it.
        for v in &meshes[SAND].vertices {
            assert!(v.position.x + 0.5 * v.position.z - 0.3 <= 1.0e-4);
        }
        for v in &meshes[GRASS].vertices {
            let p = Vec2::new(v.position.x, v.position.z);
            assert!(
                p.length() >= 6.0 - 1.0e-2 && v.position.x + 0.5 * v.position.z - 0.3 >= -1.0e-4
            );
        }
    }
}
