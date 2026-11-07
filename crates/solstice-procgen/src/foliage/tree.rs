//! Trees and bushes, grown from a species description.
//!
//! A plant is a trunk that carries levels of branches, each level spiralling around its parent at
//! the golden angle as real shoots do, and the last level carries leaves. Every stem is a tube of
//! its own, starting inside its parent as branches do in production tree models. That keeps each
//! one a separate piece, which the cluster builder can thin out with distance like a leaf.
//!
//! The crown's outline comes from how long the trunk's branches are at each height, which is how
//! species differ most at a glance: a rounded oak, a narrow birch, a wide bush.

use super::Rng;
use crate::field::{fbm, ridged, smoothstep, Field};
use crate::surface::{Relief, Shade, Surface, Texel, TILE_METRES};
use glam::{Quat, Vec2, Vec3};
use solstice_builder::{Mesh, Vertex};
use std::f32::consts::{PI, TAU};

/// The golden angle, at which successive shoots spiral around a stem.
const GOLDEN_ANGLE: f32 = 2.399_963;
/// Bark repeats around a stem in whole multiples of this, in metres, so its seam never shows.
pub const BARK_REPEAT: f32 = TILE_METRES / 16.0;
/// Leaf colour variants in the leaf atlas, side by side.
pub const LEAF_VARIANTS: u32 = 4;

/// The outline of a crown: how long the trunk's branches are at each height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Crown {
    /// Longest a third of the way up, as on an oak or a beech in the open.
    Round,
    /// Longest at the bottom, as on a young conifer or a narrow birch.
    Cone,
    /// Much the same length all the way up, as on a bush.
    Even,
}

impl Crown {
    /// Branch length at `x`, from the bottom of the crown (0) to the top (1).
    fn envelope(self, x: f32) -> f32 {
        match self {
            Self::Round => (PI * (x * 0.85 + 0.12)).sin().powf(0.7),
            Self::Cone => 1.0 - x * 0.85,
            Self::Even => 1.0 - x * 0.3,
        }
    }
}

/// One level of branching.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    /// Children per metre of the parent's length.
    pub per_metre: f32,
    /// Stretch of the parent they grow from, as fractions of its length.
    pub span: (f32, f32),
    /// Angle from the parent, in degrees.
    pub angle: f32,
    /// Length as a fraction of the parent's.
    pub length: f32,
    /// Radius where it starts, as a fraction of the parent's there.
    pub radius: f32,
    /// How strongly it turns towards the light, per metre.
    pub upturn: f32,
    /// How strongly it sags under its weight, per metre, most towards its tip.
    pub droop: f32,
    /// How much it wanders, per metre.
    pub gnarl: f32,
}

/// How leaves grow.
#[derive(Clone, Copy, Debug)]
pub struct Leaves {
    /// Leaves per metre of the last level of stem.
    pub per_metre: f32,
    /// Range of leaf lengths, in metres.
    pub length: (f32, f32),
    /// Width as a fraction of length.
    pub width: f32,
    /// How likely each atlas variant is.
    pub variants: [f32; LEAF_VARIANTS as usize],
    /// How far leaf normals lean out from the centre of the crown, which lights a crown as one
    /// soft mass rather than thousands of separately facing leaves.
    pub crown_normals: f32,
}

/// Everything that decides how a plant grows.
#[derive(Clone, Debug)]
pub struct Species {
    /// Length of the trunk, in metres.
    pub height: f32,
    pub trunk_radius: f32,
    /// How much wider the trunk spreads at the ground, as a fraction of its radius.
    pub root_flare: f32,
    pub trunk_gnarl: f32,
    pub crown: Crown,
    /// Branching levels, starting with those on the trunk.
    pub levels: Vec<Level>,
    pub leaves: Leaves,
}

impl Species {
    /// A broadleaf tree with a rounded crown, like a young oak or beech.
    pub fn broadleaf() -> Self {
        Self {
            height: 11.0,
            trunk_radius: 0.24,
            root_flare: 0.6,
            trunk_gnarl: 0.05,
            crown: Crown::Round,
            levels: vec![
                Level {
                    per_metre: 4.2,
                    span: (0.28, 0.97),
                    angle: 52.0,
                    length: 0.5,
                    radius: 0.55,
                    upturn: 0.12,
                    droop: 0.05,
                    gnarl: 0.25,
                },
                Level {
                    per_metre: 2.6,
                    span: (0.15, 0.95),
                    angle: 42.0,
                    length: 0.42,
                    radius: 0.6,
                    upturn: 0.25,
                    droop: 0.1,
                    gnarl: 0.4,
                },
                Level {
                    per_metre: 8.0,
                    span: (0.2, 1.0),
                    angle: 48.0,
                    length: 0.35,
                    radius: 0.6,
                    upturn: 0.3,
                    droop: 0.2,
                    gnarl: 0.6,
                },
            ],
            leaves: Leaves {
                per_metre: 42.0,
                length: (0.08, 0.13),
                width: 0.55,
                variants: [0.4, 0.3, 0.2, 0.1],
                crown_normals: 0.45,
            },
        }
    }

    /// A dense bush of many stems from the ground.
    pub fn bush() -> Self {
        Self {
            height: 0.25,
            trunk_radius: 0.05,
            root_flare: 0.2,
            trunk_gnarl: 0.1,
            crown: Crown::Even,
            levels: vec![
                Level {
                    per_metre: 64.0,
                    span: (0.0, 1.0),
                    angle: 38.0,
                    length: 6.0,
                    radius: 0.55,
                    upturn: 0.15,
                    droop: 0.35,
                    gnarl: 0.6,
                },
                Level {
                    per_metre: 12.0,
                    span: (0.2, 1.0),
                    angle: 45.0,
                    length: 0.35,
                    radius: 0.6,
                    upturn: 0.4,
                    droop: 0.2,
                    gnarl: 0.8,
                },
            ],
            leaves: Leaves {
                per_metre: 55.0,
                length: (0.04, 0.065),
                width: 0.6,
                variants: [0.3, 0.4, 0.3, 0.0],
                crown_normals: 0.55,
            },
        }
    }
}

/// A grown plant: wood and leaves, which take different materials.
pub struct Plant {
    pub wood: Mesh,
    pub leaves: Mesh,
}

/// A stem as a curve: points along it, the radius at each, and the direction it grows there.
struct Stem {
    points: Vec<Vec3>,
    radii: Vec<f32>,
    length: f32,
}

impl Stem {
    /// Point, direction and radius at `t` of the way along.
    fn at(&self, t: f32) -> (Vec3, Vec3, f32) {
        let segments = self.points.len() - 1;
        let x = (t.clamp(0.0, 1.0) * segments as f32).min(segments as f32 - 1.0e-4);
        let i = x as usize;
        let f = x - i as f32;
        let point = self.points[i].lerp(self.points[i + 1], f);
        let direction = (self.points[i + 1] - self.points[i]).normalize_or(Vec3::Y);
        let radius = self.radii[i] + (self.radii[i + 1] - self.radii[i]) * f;
        (point, direction, radius)
    }
}

/// How finely a stem is divided along its length, in metres: about three of its widths, so that
/// bends stay smooth, but never so fine that twigs cost more than the leaves they carry.
fn ring_spacing(radius: f32, length: f32) -> f32 {
    (radius * 6.0).clamp(0.12, 0.6).min(length / 2.0)
}

/// Grows one stem by marching along it, turning a little at each step.
#[allow(clippy::too_many_arguments)]
fn grow(
    rng: &mut Rng,
    start: Vec3,
    direction: Vec3,
    length: f32,
    radius: f32,
    tip_radius: f32,
    level: &Level,
) -> Stem {
    let step = ring_spacing(radius, length);
    let segments = ((length / step).ceil() as usize).max(2);
    let step = length / segments as f32;
    let mut points = vec![start];
    let mut radii = vec![radius];
    let mut direction = direction.normalize();
    let mut point = start;
    for i in 1..=segments {
        let t = i as f32 / segments as f32;
        let wander = Vec3::new(rng.signed(), rng.signed() * 0.5, rng.signed()) * level.gnarl;
        direction = (direction
            + (wander + Vec3::Y * level.upturn - Vec3::Y * level.droop * t * 2.0) * step)
            .normalize();
        point += direction * step;
        points.push(point);
        radii.push(radius + (tip_radius - radius) * t.powf(0.8));
    }
    Stem {
        points,
        radii,
        length,
    }
}

impl Species {
    /// Grows one plant. Different seeds give different plants of the same species.
    pub fn grow(&self, seed: u32) -> Plant {
        let mut rng = Rng::new(seed, 7);
        let trunk_level = Level {
            per_metre: 0.0,
            span: (0.0, 1.0),
            angle: 0.0,
            length: 1.0,
            radius: 1.0,
            upturn: 0.02,
            droop: 0.0,
            gnarl: self.trunk_gnarl,
        };
        let lean = Vec3::new(rng.signed(), 0.0, rng.signed()) * 0.05;
        let trunk = grow(
            &mut rng,
            Vec3::ZERO - Vec3::Y * self.trunk_radius,
            Vec3::Y + lean,
            self.height + self.trunk_radius,
            self.trunk_radius,
            self.trunk_radius * 0.25,
            &trunk_level,
        );

        let mut stems = vec![trunk];
        let mut generation = vec![0usize];
        for (depth, level) in self.levels.iter().enumerate() {
            let mut next = Vec::new();
            for &parent_index in &generation {
                let parent_length = stems[parent_index].length;
                let span = level.span.1 - level.span.0;
                let count = (parent_length * span * level.per_metre).round().max(1.0) as usize;
                let mut spin = rng.range(0.0, TAU);
                for k in 0..count {
                    let t = level.span.0 + span * (k as f32 + rng.range(0.2, 0.8)) / count as f32;
                    let (point, along, parent_radius) = stems[parent_index].at(t);
                    spin += GOLDEN_ANGLE;
                    let side = perpendicular(along, spin);
                    let mut angle = level.angle + rng.signed() * 10.0;
                    if depth == 0 {
                        // Lower limbs spread out wide, upper ones reach up, as on an open-grown tree.
                        angle += (1.0 - (t - level.span.0) / span) * 28.0 - 8.0;
                    }
                    let angle = angle.to_radians();
                    let direction = along * angle.cos() + side * angle.sin();
                    let mut length = parent_length * level.length * rng.range(0.75, 1.2);
                    if depth == 0 {
                        length *= self.crown.envelope((t - level.span.0) / span);
                    } else {
                        // Shoots near a branch's tip are shorter, which rounds off its end.
                        length *= 1.0 - 0.5 * (t - level.span.0) / span;
                    }
                    if length < 0.03 {
                        continue;
                    }
                    let radius = (parent_radius * level.radius).max(0.003);
                    let stem = grow(
                        &mut rng,
                        point,
                        direction,
                        length,
                        radius,
                        (radius * 0.2).max(0.0015),
                        level,
                    );
                    next.push(stems.len());
                    stems.push(stem);
                }
            }
            generation = next;
        }

        let mut wood = Mesh::default();
        for (index, stem) in stems.iter().enumerate() {
            let flare = if index == 0 { self.root_flare } else { 0.0 };
            add_tube(&mut wood, stem, flare, &mut rng);
        }

        let crown_center = crown_center(&stems, &generation);
        let mut leaves = Mesh::default();
        for &index in &generation {
            add_leaves(
                &mut leaves,
                &stems[index],
                &self.leaves,
                crown_center,
                &mut rng,
            );
        }
        Plant { wood, leaves }
    }
}

/// A unit vector perpendicular to `axis`, turned `angle` around it.
fn perpendicular(axis: Vec3, angle: f32) -> Vec3 {
    let reference = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let first = axis.cross(reference).normalize();
    Quat::from_axis_angle(axis, angle) * first
}

/// The middle of where the leaves are.
fn crown_center(stems: &[Stem], leafy: &[usize]) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut count = 0.0;
    for &index in leafy {
        for &p in &stems[index].points {
            sum += p;
            count += 1.0;
        }
    }
    if count > 0.0 {
        sum / count
    } else {
        Vec3::ZERO
    }
}

/// A stem's surface: rings of vertices along its curve, closing to a point at its tip.
fn add_tube(mesh: &mut Mesh, stem: &Stem, root_flare: f32, rng: &mut Rng) {
    let sides = ((stem.radii[0] * TAU / 0.035) as usize).clamp(3, 16);
    // Bark wraps a whole number of times, so its seam matches.
    let around = (stem.radii[0] * TAU / BARK_REPEAT).round().max(1.0) * BARK_REPEAT;
    let lobes = rng.range(0.0, TAU);
    let base = mesh.vertices.len() as u32;

    // Frames carried along the curve without twisting.
    let mut normal = perpendicular((stem.points[1] - stem.points[0]).normalize_or(Vec3::Y), 0.0);
    let mut distance = 0.0;
    let rings = stem.points.len();
    for i in 0..rings {
        let tangent = if i + 1 < rings {
            (stem.points[i + 1] - stem.points[i]).normalize_or(Vec3::Y)
        } else {
            (stem.points[i] - stem.points[i - 1]).normalize_or(Vec3::Y)
        };
        if i > 0 {
            distance += stem.points[i].distance(stem.points[i - 1]);
        }
        normal = (normal - tangent * normal.dot(tangent)).normalize_or(perpendicular(tangent, 0.0));
        let binormal = tangent.cross(normal);
        let tip = i + 1 == rings;
        // The trunk spreads into buttresses where it meets the ground.
        let height = stem.points[i].y;
        let flare = root_flare * (1.0 - smoothstep(0.0, 1.2, height)).powi(2);
        for j in 0..=sides {
            let theta = j as f32 / sides as f32 * TAU;
            let buttress = 1.0 + flare * (0.6 + 0.4 * (theta * 5.0 + lobes).cos());
            let outward = normal * theta.cos() + binormal * theta.sin();
            let radius = if tip { 0.0 } else { stem.radii[i] * buttress };
            mesh.vertices.push(Vertex {
                position: stem.points[i] + outward * radius,
                normal: outward,
                uv: Vec2::new(j as f32 / sides as f32 * around, distance),
            });
        }
    }
    let row = sides as u32 + 1;
    for i in 0..rings as u32 - 1 {
        for j in 0..sides as u32 {
            let a = base + i * row + j;
            let (b, c, d) = (a + 1, a + row + 1, a + row);
            mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
}

/// Fractions of a leaf's length with a full cross-section, between its stalk and its tip.
const LEAF_ROWS: [f32; 2] = [0.32, 0.68];

/// Leaves along a stem, alternating around it, with a few at its tip.
fn add_leaves(mesh: &mut Mesh, stem: &Stem, leaves: &Leaves, crown: Vec3, rng: &mut Rng) {
    let count = (stem.length * leaves.per_metre).round().max(1.0) as usize;
    let total: f32 = leaves.variants.iter().sum();
    let mut spin = rng.range(0.0, TAU);
    for k in 0..count {
        // Mostly towards the tip, where the newest growth is.
        let t = 0.25 + 0.75 * ((k as f32 + rng.unit()) / count as f32).powf(0.7);
        let (point, along, radius) = stem.at(t);
        spin += GOLDEN_ANGLE;
        let side = perpendicular(along, spin);
        let out = (along * 0.6 + side).normalize();
        let length = rng.range(leaves.length.0, leaves.length.1);
        let mut pick = rng.unit() * total;
        let mut variant = 0;
        while variant + 1 < LEAF_VARIANTS as usize && pick >= leaves.variants[variant] {
            pick -= leaves.variants[variant];
            variant += 1;
        }
        // Leaves turn their faces to the sky, give or take.
        let roll = rng.signed() * 0.6;
        add_leaf(
            mesh,
            point + side * radius,
            out,
            length,
            leaves.width,
            roll,
            variant as u32,
            crown,
            leaves.crown_normals,
        );
    }
}

/// One leaf: a stalk point, two rows folded along the midrib, and a tip, curling down a little.
#[allow(clippy::too_many_arguments)]
fn add_leaf(
    mesh: &mut Mesh,
    base: Vec3,
    axis: Vec3,
    length: f32,
    width_ratio: f32,
    roll: f32,
    variant: u32,
    crown: Vec3,
    crown_normals: f32,
) {
    let up_ish = (Vec3::Y - axis * axis.dot(Vec3::Y)).normalize_or(perpendicular(axis, 0.0));
    let face = Quat::from_axis_angle(axis, roll) * up_ish;
    let side = axis.cross(face).normalize();
    let width = length * width_ratio;
    // Widest a third of the way along, then tapering to a point.
    let half_width = |s: f32| width * 0.5 * (s.powf(0.6) * (1.0 - s).powf(0.9)) / 0.47;
    let curl = |s: f32| -face * (s * s * length * 0.18);
    let u = |across: f32| (variant as f32 + 0.5 + across * 0.47) / LEAF_VARIANTS as f32;
    let lit = |n: Vec3, at: Vec3| {
        let out = (at - crown).normalize_or(n);
        (n * (1.0 - crown_normals) + out * crown_normals).normalize()
    };

    let first = mesh.vertices.len() as u32;
    mesh.vertices.push(Vertex {
        position: base,
        normal: lit(face, base),
        uv: Vec2::new(u(0.0), 0.0),
    });
    for &s in &LEAF_ROWS {
        let center = base + axis * (s * length) + curl(s);
        let half = half_width(s);
        let fold = face * (half * 0.25);
        for (across, offset, tilt) in [
            (-1.0, -side * half, side),
            (0.0, -fold, Vec3::ZERO),
            (1.0, side * half, -side),
        ] {
            let position = center + offset;
            let normal = (face + tilt * 0.35).normalize();
            mesh.vertices.push(Vertex {
                position,
                normal: lit(normal, position),
                uv: Vec2::new(u(across * half / (width * 0.5)), s),
            });
        }
    }
    let tip = base + axis * length + curl(1.0);
    mesh.vertices.push(Vertex {
        position: tip,
        normal: lit(face, tip),
        uv: Vec2::new(u(0.0), 1.0),
    });

    // Counter-clockwise seen from the face.
    let (stalk, l0, c0, r0) = (first, first + 1, first + 2, first + 3);
    let (l1, c1, r1) = (first + 4, first + 5, first + 6);
    let point = first + 7;
    mesh.indices.extend_from_slice(&[
        stalk, c0, l0, stalk, r0, c0, //
        l0, c0, c1, l0, c1, l1, c0, r0, r1, c0, r1, c1, //
        l1, c1, point, c1, r1, point,
    ]);
}

/// The leaf atlas: [`LEAF_VARIANTS`] columns, each a leaf from stalk (v = 0) to tip (v = 1) and
/// across it (u), with the midrib in the middle. Variants run from deep green to the first yellow
/// of autumn.
pub fn leaf_atlas(seed: u32) -> Surface {
    const COLORS: [Vec3; LEAF_VARIANTS as usize] = [
        Vec3::new(0.040, 0.085, 0.016),
        Vec3::new(0.060, 0.110, 0.020),
        Vec3::new(0.095, 0.135, 0.025),
        Vec3::new(0.200, 0.150, 0.030),
    ];
    let mottle = fbm(24, 4, seed + 1);
    let blotches = fbm(8, 3, seed + 2);
    Surface::new(0.004, move |u, v| {
        let variant = ((u * LEAF_VARIANTS as f32) as u32).min(LEAF_VARIANTS - 1);
        let across = (u * LEAF_VARIANTS as f32).fract() * 2.0 - 1.0;
        let d = across.abs();
        // Side veins leave the midrib towards the tip.
        let phase = v * 7.0 - d * 2.2;
        let side_veins = (-((phase - phase.round()) * 9.0).powi(2)).exp() * (1.0 - d);
        let midrib = (-(across * 18.0).powi(2)).exp();
        let veins = side_veins.max(midrib);

        let mut color = COLORS[variant as usize];
        color *= 1.0 + 0.18 * mottle.at(u, v) + 0.1 * blotches.at(u, v);
        // Veins are paler, and the edge is a little darker.
        color = color.lerp(color * Vec3::new(1.6, 1.5, 1.3), veins * 0.5);
        color *= 1.0 - 0.18 * smoothstep(0.75, 1.0, d);
        Texel {
            color,
            // Veins sit in grooves on a leaf's upper face.
            height: -veins,
            roughness: 0.45,
            cavity: 1.0 - veins * 0.15,
        }
    })
}

/// Furrowed bark, as on an oak: long ridged plates split by deep fissures, grey on the plates and
/// brown in the fissures, with patches of lichen.
pub fn bark(seed: u32) -> Surface {
    // Bark repeats every BARK_REPEAT around a stem, so features across it repeat 16 times per tile.
    let fissures = ridged(5, 4, seed)
        .warped(fbm(6, 3, seed + 1), 0.04)
        .repeated(16, 1);
    let breaks = fbm(4, 3, seed + 2).repeated(16, 3);
    let plates = fbm(14, 4, seed + 3).repeated(16, 1);
    let grain = fbm(60, 2, seed + 4).repeated(16, 1);
    let lichen = fbm(5, 5, seed + 5).unit().smooth(0.62, 0.72);
    let tint = fbm(3, 3, seed + 6).unit();
    Surface::shaped(
        move |u, v| {
            let fissure = smoothstep(0.55, 0.95, fissures.at(u, v));
            let broken = smoothstep(0.3, 0.6, breaks.at(u, v).abs());
            let depth = fissure.max(broken * 0.6);
            let height = -0.018 * depth + 0.003 * plates.at(u, v) + 0.0008 * grain.at(u, v);
            (height, depth)
        },
        move |u, v, depth, relief: Relief| {
            let plate =
                Vec3::new(0.17, 0.155, 0.135).lerp(Vec3::new(0.12, 0.1, 0.085), tint.at(u, v));
            let fissure = Vec3::new(0.05, 0.035, 0.025);
            let mut color = plate.lerp(fissure, smoothstep(0.2, 0.8, depth));
            // Worn high points are paler.
            color *= 1.0 + (relief.fine * 60.0).clamp(-0.15, 0.2);
            let moss = lichen.at(u, v) * (1.0 - depth);
            color = color.lerp(Vec3::new(0.11, 0.13, 0.06), moss * 0.8);
            Shade {
                color,
                roughness: 0.9,
                cavity: 1.0 - depth * 0.6,
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(mesh: &Mesh) -> (Vec3, Vec3) {
        mesh.vertices
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), v| {
                (lo.min(v.position), hi.max(v.position))
            })
    }

    #[test]
    fn a_broadleaf_tree_has_the_shape_and_size_asked_for() {
        let species = Species::broadleaf();
        let plant = species.grow(0);
        let (lo, hi) = bounds(&plant.wood);
        assert!(
            hi.y > species.height * 0.9 && hi.y < species.height * 1.4,
            "top at {}",
            hi.y
        );
        // A rounded crown is wide, but narrower than it is tall.
        let spread = (hi.x - lo.x).max(hi.z - lo.z);
        assert!(
            spread > species.height * 0.35 && spread < species.height * 1.3,
            "spread {spread}"
        );
        let leaves = plant.leaves.triangle_count() / 8;
        assert!((8_000..60_000).contains(&leaves), "{leaves} leaves");
        let wood = plant.wood.triangle_count();
        assert!(wood < 120_000, "{wood} wood triangles");
    }

    #[test]
    fn surfaces_face_the_way_their_triangles_wind() {
        let plant = Species::broadleaf().grow(1);
        for mesh in [&plant.wood, &plant.leaves] {
            let mut wrong = 0;
            let mut total = 0;
            for triangle in mesh.indices.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|i| mesh.vertices[triangle[i] as usize]);
                let face = (b.position - a.position).cross(c.position - a.position);
                if face.length() < 1.0e-12 {
                    continue;
                }
                total += 1;
                if face.dot(a.normal + b.normal + c.normal) <= 0.0 {
                    wrong += 1;
                }
            }
            assert!(
                wrong * 100 < total,
                "{wrong} of {total} triangles face against their normals"
            );
        }
    }

    #[test]
    fn the_same_seed_grows_the_same_plant() {
        let species = Species::bush();
        assert_eq!(
            species.grow(3).leaves.vertices,
            species.grow(3).leaves.vertices
        );
        assert_ne!(
            species.grow(3).leaves.vertices,
            species.grow(4).leaves.vertices
        );
    }

    #[test]
    fn bark_wraps_without_a_seam() {
        // Features across the bark repeat every sixteenth of the tile.
        let surface = bark(0);
        for i in 0..20 {
            let v = i as f32 / 20.0;
            let a = surface.formula_at(0.0, v);
            let b = surface.formula_at(1.0 / 16.0, v);
            assert!(
                (a.height - b.height).abs() < 1.0e-3,
                "{} against {}",
                a.height,
                b.height
            );
        }
    }
}
