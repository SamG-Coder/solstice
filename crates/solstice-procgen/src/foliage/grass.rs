//! Grass: tufts of curved, folded, tapering blades, and the atlas that colours them.
//!
//! A real blade is not a flat strip. It is folded along its midrib into a shallow V, which catches
//! light differently on each half, it tapers to a point, and it arcs over under its own weight
//! towards the face it is folded around. Grass also grows in tufts, with blades fanning outward
//! from a crown, rather than as an even scatter, and that clumping is much of what makes a field
//! read as grass rather than as green noise.

use super::Rng;
use crate::field::smoothstep;
use crate::noise::tiling_fbm;
use crate::surface::{Surface, Texel};
use glam::{Vec2, Vec3};
use solstice_builder::{Mesh, Vertex};
use std::f32::consts::TAU;

/// Colour variants in the blade atlas, side by side: lush, fresh, blue-green and dry.
pub const VARIANTS: u32 = 4;
/// The atlas variant of dead, dry blades.
pub const DRY: u32 = 3;

/// Heights along a blade, as fractions of its length, at which it has a full cross-section. The
/// tip is a single point beyond the last.
const ROWS: [f32; 3] = [0.0, 0.4, 0.75];
/// How deep the fold along the midrib is, relative to the blade's width.
const FOLD: f32 = 0.18;
/// How far blade normals lean towards straight up. Grass in Unreal is usually lit this way, as
/// light through a whole field rather than off each blade, which keeps a meadow from turning to
/// glitter where blades face away from the sun.
const NORMAL_UP: f32 = 0.3;
/// How far below the ground blades start, so they never float where the ground is uneven.
const ROOT_DEPTH: f32 = 0.03;

/// A square patch of grass, which lays edge to edge with copies of itself.
#[derive(Clone, Copy, Debug)]
pub struct Meadow {
    /// Edge of the patch, in metres.
    pub size: f32,
    pub tufts_per_square_metre: f32,
    pub blades_per_tuft: (u32, u32),
    /// Range of tuft heights, in metres. Blades vary around their tuft's height.
    pub height: (f32, f32),
    /// Range of blade widths at the base, in metres.
    pub width: (f32, f32),
    /// How likely a tuft is to be each variant, before dry blades are added.
    pub variants: [f32; VARIANTS as usize],
    /// Fraction of blades that are dead and dry, in any tuft.
    pub dry: f32,
    /// Short single blades between the tufts, per square metre, which hide the soil as the thatch
    /// and young shoots of a real meadow do.
    pub undergrowth_per_square_metre: f32,
    /// Range of undergrowth heights, in metres.
    pub undergrowth_height: (f32, f32),
    pub seed: u32,
}

impl Default for Meadow {
    /// An unmown summer meadow.
    fn default() -> Self {
        Self {
            size: 2.0,
            tufts_per_square_metre: 50.0,
            blades_per_tuft: (8, 20),
            height: (0.22, 0.55),
            width: (0.005, 0.009),
            variants: [0.45, 0.3, 0.25, 0.0],
            dry: 0.08,
            undergrowth_per_square_metre: 450.0,
            undergrowth_height: (0.05, 0.16),
            seed: 0,
        }
    }
}

impl Meadow {
    /// The patch's blades, with the ground at Y = 0 and the patch spanning `0..size` in X and Z.
    pub fn mesh(&self) -> Mesh {
        let mut rng = Rng::new(self.seed, 1);
        let mut mesh = Mesh::default();
        let tufts = (self.size * self.size * self.tufts_per_square_metre).round() as u32;
        let total_weight: f32 = self.variants.iter().sum();
        for _ in 0..tufts {
            let center = Vec3::new(rng.range(0.0, self.size), 0.0, rng.range(0.0, self.size));
            let tuft_height = rng.range(self.height.0, self.height.1);
            let tuft_width = rng.range(self.width.0, self.width.1);
            let mut pick = rng.unit() * total_weight;
            let mut variant = 0;
            while variant + 1 < VARIANTS as usize && pick >= self.variants[variant] {
                pick -= self.variants[variant];
                variant += 1;
            }
            let count = self.blades_per_tuft.0
                + rng.index((self.blades_per_tuft.1 - self.blades_per_tuft.0 + 1) as usize) as u32;
            for _ in 0..count {
                // Blades rise from a crown a few centimetres across and fan outward from it,
                // the outer ones leaning furthest.
                let angle = rng.range(0.0, TAU);
                let reach = rng.unit().sqrt();
                let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
                let root = center + outward * reach * 0.035 - Vec3::Y * ROOT_DEPTH;
                let swing = rng.signed() * 0.7;
                let bend = Vec3::new((angle + swing).cos(), 0.0, (angle + swing).sin());
                let lean = (0.12 + reach * 0.45 + rng.range(0.0, 0.25)).min(0.85);
                let blade_variant = if rng.unit() < self.dry {
                    DRY
                } else {
                    variant as u32
                };
                add_blade(
                    &mut mesh,
                    root,
                    bend,
                    tuft_height * rng.range(0.65, 1.15),
                    tuft_width * rng.range(0.8, 1.2),
                    lean,
                    blade_variant,
                );
            }
        }
        let undergrowth =
            (self.size * self.size * self.undergrowth_per_square_metre).round() as u32;
        for _ in 0..undergrowth {
            let root = Vec3::new(
                rng.range(0.0, self.size),
                -ROOT_DEPTH,
                rng.range(0.0, self.size),
            );
            let angle = rng.range(0.0, TAU);
            let variant = if rng.unit() < self.dry * 2.0 {
                DRY
            } else {
                rng.index(DRY as usize) as u32
            };
            add_blade(
                &mut mesh,
                root,
                Vec3::new(angle.cos(), 0.0, angle.sin()),
                rng.range(self.undergrowth_height.0, self.undergrowth_height.1),
                rng.range(self.width.0, self.width.1) * 0.8,
                rng.range(0.1, 0.7),
                variant,
            );
        }
        mesh
    }
}

/// One blade, arcing over towards `bend` by `lean` of its height at the tip.
fn add_blade(
    mesh: &mut Mesh,
    root: Vec3,
    bend: Vec3,
    length: f32,
    width: f32,
    lean: f32,
    variant: u32,
) {
    // A quadratic curve, rising straight from the crown and arcing over at the top, sized so the
    // tip is `lean` of the blade's length out and the blade keeps roughly its length.
    let tip = root + Vec3::Y * length * (1.0 - lean * lean).sqrt() * 0.97 + bend * length * lean;
    let control = root + Vec3::Y * length * 0.6;
    let point = |t: f32| root * (1.0 - t) * (1.0 - t) + control * 2.0 * t * (1.0 - t) + tip * t * t;
    let direction = |t: f32| ((control - root) * (1.0 - t) + (tip - control) * t).normalize();

    // The blade's flat side faces the way it bends, as real blades fold around their upper face.
    let side = Vec3::Y.cross(bend).normalize();
    let u = |across: f32| (variant as f32 + 0.5 + across * 0.45) / VARIANTS as f32;
    let lit = |n: Vec3| (n + Vec3::Y * NORMAL_UP).normalize();

    let base = mesh.vertices.len() as u32;
    for &t in &ROWS {
        let center = point(t);
        let along = direction(t);
        let face = side.cross(along).normalize();
        let half = width * 0.5 * (1.0 - t.powf(2.2));
        let fold = face * (FOLD * half * 2.0);
        // Each half of the V leans in towards the midrib.
        let tilt = side * (FOLD * 2.0);
        mesh.vertices.push(Vertex {
            position: center - side * half,
            normal: lit((face + tilt).normalize()),
            uv: Vec2::new(u(-1.0), t),
        });
        mesh.vertices.push(Vertex {
            position: center - fold,
            normal: lit(face),
            uv: Vec2::new(u(0.0), t),
        });
        mesh.vertices.push(Vertex {
            position: center + side * half,
            normal: lit((face - tilt).normalize()),
            uv: Vec2::new(u(1.0), t),
        });
    }
    let tip_face = side.cross(direction(1.0)).normalize();
    mesh.vertices.push(Vertex {
        position: tip,
        normal: lit(tip_face),
        uv: Vec2::new(u(0.0), 1.0),
    });

    // Counter-clockwise seen from the face.
    for row in 0..ROWS.len() as u32 - 1 {
        let (l0, c0, r0) = (base + row * 3, base + row * 3 + 1, base + row * 3 + 2);
        let (l1, c1, r1) = (l0 + 3, c0 + 3, r0 + 3);
        mesh.indices
            .extend_from_slice(&[l0, c0, c1, l0, c1, l1, c0, r0, r1, c0, r1, c1]);
    }
    let last = base + (ROWS.len() as u32 - 1) * 3;
    let tip_index = last + 3;
    mesh.indices
        .extend_from_slice(&[last, last + 1, tip_index, last + 1, last + 2, tip_index]);
}

/// The blade atlas: [`VARIANTS`] columns, each a blade's colour from root (v = 0) to tip (v = 1)
/// and across it (u).
pub fn blade_atlas(seed: u32) -> Surface {
    // Linear albedo at mid-blade. Living grass reflects little: about 0.1 in green.
    const COLORS: [Vec3; VARIANTS as usize] = [
        Vec3::new(0.050, 0.105, 0.018),
        Vec3::new(0.085, 0.130, 0.022),
        Vec3::new(0.042, 0.090, 0.034),
        Vec3::new(0.230, 0.180, 0.080),
    ];
    const SHEATH: Vec3 = Vec3::new(0.11, 0.12, 0.05);
    const WITHERED: Vec3 = Vec3::new(0.20, 0.15, 0.07);
    Surface::new(0.0006, move |u, v| {
        let variant = ((u * VARIANTS as f32) as u32).min(VARIANTS - 1);
        let across = (u * VARIANTS as f32).fract();
        let base = COLORS[variant as usize];

        // Parallel veins, and a paler midrib.
        let veins = 0.5 + 0.5 * (across * 9.0 * TAU).cos();
        let midrib = (-((across - 0.5) * 14.0).powi(2)).exp();
        // Slow streaks along the blade.
        let streaks = tiling_fbm(across * 0.25 + variant as f32 * 0.25, v, 8, 3, seed);

        let mut color = base * (0.9 + 0.12 * veins + 0.25 * midrib) * (1.0 + 0.35 * streaks);
        // The pale sheath at the base, and a withered tip on living blades.
        color = color.lerp(SHEATH, (1.0 - smoothstep(0.0, 0.14, v)) * 0.7);
        if variant != DRY {
            let wither =
                smoothstep(0.82, 1.0, v) * (0.5 + 0.5 * tiling_fbm(across, v, 4, 2, seed + 7));
            color = color.lerp(WITHERED, wither.clamp(0.0, 1.0) * 0.8);
        }
        Texel {
            color,
            height: veins * 0.4 + midrib,
            // Fresh blades have a waxy sheen. Dry ones are matte.
            roughness: if variant == DRY { 0.85 } else { 0.6 },
            // Lower blades sit in the shade of the whole tuft.
            cavity: 0.35 + 0.65 * smoothstep(0.0, 0.55, v),
        }
    })
}

/// The angle between two directions, for tests.
#[cfg(test)]
fn angle_between(a: Vec3, b: Vec3) -> f32 {
    a.normalize()
        .dot(b.normalize())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_meadow_has_the_blades_it_asks_for_inside_its_patch() {
        let meadow = Meadow::default();
        let mesh = meadow.mesh();
        let blades = mesh.vertices.len() / (ROWS.len() * 3 + 1);
        let per_square_metre = blades as f32 / (meadow.size * meadow.size);
        // About 50 tufts of 8 to 20 blades, and 450 short ones between.
        assert!(
            (1000.0..1400.0).contains(&per_square_metre),
            "{per_square_metre} blades per m²"
        );
        for vertex in &mesh.vertices {
            let p = vertex.position;
            assert!(p.x > -0.5 && p.x < meadow.size + 0.5 && p.z > -0.5 && p.z < meadow.size + 0.5);
            assert!(p.y >= -ROOT_DEPTH - 1.0e-4 && p.y < meadow.height.1 * 1.2);
            assert!((vertex.normal.length() - 1.0).abs() < 1.0e-3);
        }
    }

    #[test]
    fn blades_face_the_way_their_triangles_wind() {
        let mesh = Meadow::default().mesh();
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|i| mesh.vertices[triangle[i] as usize]);
            let face = (b.position - a.position).cross(c.position - a.position);
            if face.length() < 1.0e-9 {
                continue;
            }
            let normal = a.normal + b.normal + c.normal;
            assert!(
                angle_between(face, normal) < 80.0,
                "a triangle faces {} degrees from its normals",
                angle_between(face, normal)
            );
        }
    }

    #[test]
    fn the_same_seed_grows_the_same_meadow() {
        let a = Meadow::default().mesh();
        let b = Meadow::default().mesh();
        let c = Meadow {
            seed: 1,
            ..Default::default()
        }
        .mesh();
        assert_eq!(a.vertices, b.vertices);
        assert_ne!(a.vertices, c.vertices);
    }

    #[test]
    fn living_grass_is_as_dark_as_real_grass() {
        let layer = blade_atlas(0).bake(64);
        let top = layer.color_mips.last().unwrap();
        // The whole atlas averaged, in sRGB. Real grass is about 0.1 linear albedo in green.
        let green = (top[1] as f32 / 255.0).powf(2.2);
        assert!((0.06..0.2).contains(&green), "green albedo {green}");
    }
}
