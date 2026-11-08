//! What the town is built of: the stonework, and plain massing for what later steps replace.

use crate::geometry::Geometry;
use crate::ground::{self, QUAY_DECK};
use glam::{Vec2, Vec3};

/// How far down the stonework goes, into the seabed.
const FOOTING: f32 = -13.0;

/// One flight of water stairs: 16 cm risers on 30 cm treads, from the deck to below the sea.
const STAIR_RISE: f32 = 0.16;
const STAIR_GOING: f32 = 0.3;
const STAIR_STEPS: usize = 13;
const STAIRS_Z: f32 = -10.0;
const STAIRS_WIDTH: f32 = 2.0;
/// The pier's deck, two of the same steps below the quay.
const PIER_DECK: f32 = QUAY_DECK - 2.0 * STAIR_RISE;

/// The quay: its wall leans back a little as it rises, as harbour walls do.
fn quay_length(geometry: &mut Geometry, from: f32, to: f32) {
    let ring = |z: f32| {
        vec![
            Vec3::new(-0.6, FOOTING, z),
            Vec3::new(0.0, QUAY_DECK, z),
            Vec3::new(10.0, QUAY_DECK, z),
            Vec3::new(10.0, FOOTING, z),
        ]
    };
    geometry.loft(&[ring(from), ring(to)]);
}

/// The quay from end to end, cut by its water stairs.
pub fn quay() -> Geometry {
    let mut g = Geometry::default();
    let (north, south) = (STAIRS_Z - STAIRS_WIDTH * 0.5, STAIRS_Z + STAIRS_WIDTH * 0.5);
    quay_length(&mut g, -52.0, north);
    quay_length(&mut g, south, 50.0);
    let run = STAIR_GOING * STAIR_STEPS as f32;
    g.cuboid(
        Vec3::new(run, FOOTING, north),
        Vec3::new(10.0, QUAY_DECK, south),
    );
    for step in 0..STAIR_STEPS {
        let inner = run - STAIR_GOING * step as f32;
        g.cuboid(
            Vec3::new(inner - STAIR_GOING, FOOTING, north),
            Vec3::new(inner, QUAY_DECK - STAIR_RISE * (step + 1) as f32, south),
        );
    }
    g
}

/// The mole's centre line, from the quay's north end out to the lighthouse.
fn mole_point(t: f32) -> Vec2 {
    let [a, b, c, d] = [
        // Starting well inside the quay, so that their edges meet with no gap.
        Vec2::new(6.0, -48.0),
        Vec2::new(-35.0, -50.0),
        Vec2::new(-65.0, -62.0),
        Vec2::new(-92.0, -65.0),
    ];
    let s = 1.0 - t;
    a * s * s * s + b * 3.0 * s * s * t + c * 3.0 * s * t * t + d * t * t * t
}

/// Deck height of the mole, which climbs from the quay by a ramp.
const MOLE_DECK: f32 = 3.2;
const RAMP: f32 = 16.0;
/// How far the mole runs level with the quay, inside it, before its ramp begins.
const RAMP_START: f32 = 6.0;
const PARAPET: f32 = 1.2;
const MOLE_STATIONS: usize = 48;

/// A breakwater, with a parapet on its seaward (north) side.
pub fn mole() -> Geometry {
    let mut g = Geometry::default();
    let points: Vec<Vec2> = (0..=MOLE_STATIONS)
        .map(|i| mole_point(i as f32 / MOLE_STATIONS as f32))
        .collect();
    let mut along = 0.0;
    let rings: Vec<Vec<Vec3>> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            if i > 0 {
                along += (p - points[i - 1]).length();
            }
            let before = points[i.saturating_sub(1)];
            let after = points[(i + 1).min(MOLE_STATIONS)];
            let tangent = (after - before).normalize();
            let seaward = Vec2::new(-tangent.y, tangent.x);
            let deck =
                QUAY_DECK + (MOLE_DECK - QUAY_DECK) * ((along - RAMP_START) / RAMP).clamp(0.0, 1.0);
            [
                (4.8, FOOTING),
                (3.5, deck + PARAPET),
                (2.9, deck + PARAPET),
                (2.9, deck),
                (-3.5, deck),
                (-4.6, FOOTING),
            ]
            .map(|(offset, y)| {
                let q = p + seaward * offset;
                Vec3::new(q.x, y, q.y)
            })
            .to_vec()
        })
        .collect();
    g.loft(&rings);
    g.frustum(
        Vec3::new(-92.0, FOOTING, -65.0),
        10.0,
        7.5,
        MOLE_DECK - FOOTING,
        48,
    );
    g
}

/// The materials massing is drawn in, in order.
#[derive(Default)]
pub struct Massing {
    pub walls: Geometry,
    pub roofs: Geometry,
    pub stone: Geometry,
    pub timber: Geometry,
}

/// How far a building's walls go below the ground at its lowest corner.
const SINK: f32 = 0.5;

/// A plain building with a pitched roof, standing on the ground. Its ridge runs along z, or along x.
fn building(m: &mut Massing, min: Vec2, max: Vec2, eaves: f32, ridge: f32, ridge_along_z: bool) {
    let foot = [min, max, Vec2::new(min.x, max.y), Vec2::new(max.x, min.y)]
        .map(|c| ground::height(c.x, c.y));
    let base = foot.iter().copied().fold(f32::MAX, f32::min);
    let top = foot.iter().copied().fold(f32::MIN, f32::max);
    let (low, wall, peak) = (base - SINK, top + eaves, top + ridge);
    let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let mid = (min + max) * 0.5;
    let (walls, roofs) = (&mut m.walls, &mut m.roofs);
    walls.polygon(
        &[
            p(min.x, low, min.y),
            p(max.x, low, min.y),
            p(max.x, low, max.y),
            p(min.x, low, max.y),
        ],
        -Vec3::Y,
    );
    let mut wall_face = |corners: &[Vec3], outward: Vec3| walls.polygon(corners, outward);
    if ridge_along_z {
        wall_face(
            &[
                p(min.x, low, min.y),
                p(min.x, low, max.y),
                p(min.x, wall, max.y),
                p(min.x, wall, min.y),
            ],
            -Vec3::X,
        );
        wall_face(
            &[
                p(max.x, low, min.y),
                p(max.x, low, max.y),
                p(max.x, wall, max.y),
                p(max.x, wall, min.y),
            ],
            Vec3::X,
        );
        for (z, out) in [(min.y, -Vec3::Z), (max.y, Vec3::Z)] {
            wall_face(
                &[
                    p(min.x, low, z),
                    p(max.x, low, z),
                    p(max.x, wall, z),
                    p(mid.x, peak, z),
                    p(min.x, wall, z),
                ],
                out,
            );
        }
        roofs.polygon(
            &[
                p(min.x, wall, min.y),
                p(min.x, wall, max.y),
                p(mid.x, peak, max.y),
                p(mid.x, peak, min.y),
            ],
            Vec3::new(-1.0, 1.0, 0.0),
        );
        roofs.polygon(
            &[
                p(max.x, wall, min.y),
                p(max.x, wall, max.y),
                p(mid.x, peak, max.y),
                p(mid.x, peak, min.y),
            ],
            Vec3::new(1.0, 1.0, 0.0),
        );
    } else {
        wall_face(
            &[
                p(min.x, low, min.y),
                p(max.x, low, min.y),
                p(max.x, wall, min.y),
                p(min.x, wall, min.y),
            ],
            -Vec3::Z,
        );
        wall_face(
            &[
                p(min.x, low, max.y),
                p(max.x, low, max.y),
                p(max.x, wall, max.y),
                p(min.x, wall, max.y),
            ],
            Vec3::Z,
        );
        for (x, out) in [(min.x, -Vec3::X), (max.x, Vec3::X)] {
            wall_face(
                &[
                    p(x, low, min.y),
                    p(x, low, max.y),
                    p(x, wall, max.y),
                    p(x, peak, mid.y),
                    p(x, wall, min.y),
                ],
                out,
            );
        }
        roofs.polygon(
            &[
                p(min.x, wall, min.y),
                p(max.x, wall, min.y),
                p(max.x, peak, mid.y),
                p(min.x, peak, mid.y),
            ],
            Vec3::new(0.0, 1.0, -1.0),
        );
        roofs.polygon(
            &[
                p(min.x, wall, max.y),
                p(max.x, wall, max.y),
                p(max.x, peak, mid.y),
                p(min.x, peak, mid.y),
            ],
            Vec3::new(0.0, 1.0, 1.0),
        );
    }
}

/// A row of `houses` buildings filling a block, their ridges along the row.
fn row(m: &mut Massing, min: Vec2, max: Vec2, houses: u32, along_z: bool) {
    let length = if along_z {
        max.y - min.y
    } else {
        max.x - min.x
    };
    let each = length / houses as f32;
    for i in 0..houses {
        // Rooflines step, as a street built house by house does.
        let shift = [0.0, 0.5, -0.3, 0.8, 0.2, -0.5, 0.6][i as usize % 7];
        let (start, end) = (i as f32 * each, (i + 1) as f32 * each);
        let (a, b) = if along_z {
            (
                Vec2::new(min.x, min.y + start),
                Vec2::new(max.x, min.y + end),
            )
        } else {
            (
                Vec2::new(min.x + start, min.y),
                Vec2::new(min.x + end, max.y),
            )
        };
        building(m, a, b, 9.5 + shift, 13.0 + shift, along_z);
    }
}

/// The blocks of the town, the warehouse, chapel, wall, lighthouse, pier and ships, in plain shapes.
/// Each is replaced by the real thing as its step lands.
pub fn massing() -> Massing {
    let mut m = Massing::default();
    let block = |x0: f32, z0: f32, x1: f32, z1: f32| (Vec2::new(x0, z0), Vec2::new(x1, z1));

    // The quay-front row, with gaps for Net Lane, the passage up from the water stairs, and
    // Chapel Steps.
    for (min, max, houses) in [
        (block(11.0, -36.0, 23.0, -31.6), 1),
        (block(11.0, -28.4, 23.0, -11.6), 3),
        (block(11.0, -8.4, 23.0, 23.7), 5),
        (block(11.0, 26.3, 23.0, 50.0), 3),
    ]
    .map(|(b, n)| (b.0, b.1, n))
    {
        row(&mut m, min, max, houses, true);
    }
    // Round the square, north of Net Lane, south of Chapel Steps, and along Upper Street.
    for (min, max, houses, along_z) in [
        (block(26.0, -28.0, 56.0, -16.0), 4, false),
        (block(26.0, 17.0, 56.0, 23.6), 4, false),
        (block(57.0, -14.0, 68.0, 16.0), 4, true),
        (block(38.0, -46.0, 68.0, -32.0), 4, false),
        (block(26.0, 26.5, 68.0, 44.0), 5, false),
        (block(72.0, -46.0, 104.0, -32.0), 4, false),
        (block(72.0, -28.0, 84.0, 22.0), 6, true),
        (block(92.0, -28.0, 104.0, 18.0), 6, true),
        (block(108.0, -46.0, 120.0, -32.0), 2, true),
        (block(108.0, -28.0, 120.0, 22.0), 6, true),
        (block(108.0, 26.0, 120.0, 46.0), 3, true),
    ]
    .map(|(b, n, a)| (b.0, b.1, n, a))
    {
        row(&mut m, min, max, houses, along_z);
    }

    let (min, max) = block(14.0, -46.0, 38.0, -36.0);
    building(&mut m, min, max, 5.5, 8.0, false);
    // The chapel: a nave with its door on Chapel Steps, and a square tower on its seaward side.
    let (min, max) = block(90.5, 27.0, 99.5, 45.0);
    building(&mut m, min, max, 7.0, 11.0, true);
    let foot = ground::height(85.5, 28.0).min(ground::height(90.5, 33.0));
    m.stone.cuboid(
        Vec3::new(85.5, foot - SINK, 28.0),
        Vec3::new(90.5, foot + 22.0, 33.0),
    );

    // The town wall, stepping with the ground in lengths, with the gate's opening at z = 0.
    for (from, to) in [(-70.0_f32, -1.7_f32), (1.7, 70.0)] {
        let lengths = ((to - from) / 8.0).round();
        let step = (to - from) / lengths;
        for i in 0..lengths as u32 {
            let (a, b) = (from + step * i as f32, from + step * (i + 1) as f32);
            let (ground_a, ground_b) = (ground::height(128.0, a), ground::height(128.0, b));
            m.stone.cuboid(
                Vec3::new(127.0, ground_a.min(ground_b) - SINK * 4.0, a),
                Vec3::new(129.0, ground_a.max(ground_b) + 6.5, b),
            );
        }
    }

    // The lighthouse: a tapering tower, then its lantern.
    let base = Vec3::new(-92.0, MOLE_DECK, -64.0);
    m.stone.frustum(base, 3.5, 2.4, 18.0, 32);
    m.timber.frustum(base + Vec3::Y * 18.0, 1.4, 1.4, 4.0, 16);

    // The timber pier and its T-head, two steps down from the quay.
    m.timber
        .cuboid(Vec3::new(-44.0, 0.9, 38.5), Vec3::new(0.0, PIER_DECK, 41.5));
    m.timber.cuboid(
        Vec3::new(-0.3, 0.9, 38.5),
        Vec3::new(0.0, QUAY_DECK - STAIR_RISE, 41.5),
    );
    m.timber.cuboid(
        Vec3::new(-49.0, 0.9, 30.0),
        Vec3::new(-44.0, PIER_DECK, 50.0),
    );

    // A merchant ship alongside the quay, and small boats on their moorings.
    m.timber
        .cuboid(Vec3::new(-9.0, -2.5, -22.0), Vec3::new(-2.5, 3.0, 18.0));
    for z in [-12.0, 8.0] {
        m.timber
            .frustum(Vec3::new(-5.8, 3.0, z), 0.3, 0.15, 24.0, 12);
    }
    for (x, z, yaw) in [
        (-58.0, -15.0, 0.3),
        (-62.0, 10.0, -0.2),
        (-55.0, 24.0, 0.5),
        (-30.0, 30.0, 1.5),
        (-20.0, 33.0, 1.4),
        (-35.0, 47.0, 1.5),
    ] {
        let (sin, cos) = f32::sin_cos(yaw);
        let corner = |dx: f32, dz: f32| Vec2::new(x + dx * cos - dz * sin, z + dx * sin + dz * cos);
        let ring = |y: f32, scale: f32| {
            [(-1.0, -3.2), (1.0, -3.2), (1.0, 3.2), (-1.0, 3.2)]
                .map(|(dx, dz)| {
                    let c = corner(dx * scale, dz);
                    Vec3::new(c.x, y, c.y)
                })
                .to_vec()
        };
        m.timber.loft(&[ring(-0.5, 0.7), ring(0.75, 1.0)]);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_water_stairs_reach_below_the_sea_within_the_collision_limit() {
        assert!(QUAY_DECK - STAIR_RISE * (STAIR_STEPS as f32) < 0.0);
        const { assert!(STAIR_RISE <= 0.17) };
    }

    #[test]
    fn the_ground_stays_below_every_tread_of_the_water_stairs() {
        let run = STAIR_GOING * STAIR_STEPS as f32;
        for step in 0..STAIR_STEPS {
            let top = QUAY_DECK - STAIR_RISE * (step + 1) as f32;
            let inner = run - STAIR_GOING * step as f32;
            for x in [inner - STAIR_GOING, inner] {
                // The mesh is straight between its grid corners, one metre apart.
                let (low, t) = (x.floor(), x.fract());
                let surface = ground::height(low, STAIRS_Z) * (1.0 - t)
                    + ground::height(low + 1.0, STAIRS_Z) * t;
                assert!(surface < top, "step {step} is buried at {x}: {surface}");
            }
        }
    }

    #[test]
    fn the_mole_is_about_a_hundred_metres_and_climbs_gently() {
        let length: f32 = (0..MOLE_STATIONS)
            .map(|i| {
                let t = |k: usize| k as f32 / MOLE_STATIONS as f32;
                (mole_point(t(i + 1)) - mole_point(t(i))).length()
            })
            .sum();
        assert!((90.0..110.0).contains(&length), "{length}");
        const { assert!((MOLE_DECK - QUAY_DECK) / RAMP < 1.0 / 10.0) };
    }
}
