//! The records the particle shaders read and write, and a CPU reference for what they do.
//!
//! The reference runs the same spawn and step as `particles/common.wgsl`, for one particle at a
//! time, from the same emitter record. Tests compare the GPU's particles against it, as the
//! waves' tests compare against `spectrum.rs`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use std::f32::consts::TAU;

/// Bits of [`EmitterRecord::flags`]. Twins of the `FLAG_` constants in `particles/common.wgsl`.
pub const FLAG_LIVE: u32 = 1;
pub const FLAG_ADD: u32 = 2;
pub const FLAG_SURFACE: u32 = 4;

/// Values of [`EmitterRecord::shape_kind`].
pub const SHAPE_POINT: u32 = 0;
pub const SHAPE_SPHERE: u32 = 1;
pub const SHAPE_BOX: u32 = 2;

/// Below this drag times time step, the step uses series for its exponentials, which are
/// otherwise lost to rounding.
const SERIES_BELOW: f32 = 1.0e-2;

/// One emitter, as the GPU sees it this frame, with every value already worked out from its
/// effect's parameters. Twin of `Emitter` in `particles/common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct EmitterRecord {
    /// The effect's space to the world.
    pub transform: [[f32; 4]; 4],
    /// The shape's size: a box's full sizes, or a sphere's radius in x.
    pub shape_size: [f32; 4],
    /// Where the shape is centred in the effect's space, and in w the cone's half-angle in
    /// radians.
    pub offset_cone: [f32; 4],
    /// Initial velocity in the effect's space, and in w the spread of speed.
    pub velocity_spread: [f32; 4],
    /// Acceleration in the world, and in w the linear drag per second.
    pub acceleration_drag: [f32; 4],
    /// Lifetime's range, then size's.
    pub lifetime_size: [f32; 4],
    /// Linear colour times brightness, and opacity.
    pub color: [f32; 4],
    /// The emitter's slice of the pool.
    pub base: u32,
    pub budget: u32,
    /// Particles to spawn this frame, before the budget limits them.
    pub spawn_request: u32,
    pub seed: u32,
    pub shape_kind: u32,
    pub flags: u32,
    /// Distance over which particles fade into the scene behind them.
    pub soft: f32,
    /// The effect's scale, which particle sizes are multiplied by.
    pub scale: f32,
}

/// One particle. Twin of `Particle` in `particles/common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub age: f32,
    pub velocity: [f32; 3],
    pub lifetime: f32,
    /// Where it was at the start of its last step, for motion vectors.
    pub previous: [f32; 3],
    /// Radius at birth, in metres.
    pub size: f32,
    /// Which of its emitter's particles it is, counting every one spawned: its random values
    /// come from this, not from the slot it happens to occupy.
    pub index: u32,
    pub rotation: f32,
    pub _pad: [u32; 2],
}

/// The GPU's running state for one emitter. Twin of `EmitterState` in `particles/common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct EmitterState {
    /// How many slots the free list holds.
    pub free_count: u32,
    /// How many particles each of the two alive lists holds.
    pub alive: [u32; 2],
    /// Every particle ever spawned, and every one that could not be for lack of budget.
    pub spawned: u32,
    pub dropped: u32,
    /// This frame: how many are spawned, where they come from on the free list, and the index
    /// of the first.
    pub spawn_count: u32,
    pub spawn_first: u32,
    pub spawn_index: u32,
    /// This frame: how many particles the update ages.
    pub update_count: u32,
    /// This frame: the first workgroup of this emitter in the spawn and update dispatches.
    pub spawn_start: u32,
    pub update_start: u32,
    pub _pad: [u32; 5],
}

/// PCG hash: twin of `pcg` in `common/math.wgsl`.
pub fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Twin of `hash_to_unit` in `common/math.wgsl`.
pub fn hash_to_unit(v: u32) -> f32 {
    (pcg(v) >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// A particle's random number for `stream`, in `0..1`. Twin of `particle_random`.
pub fn random(seed: u32, index: u32, stream: u32) -> f32 {
    hash_to_unit(seed ^ pcg(index.wrapping_mul(16).wrapping_add(stream)))
}

/// How long the `i`th of `n` particles spawned over a step of `seconds` has lived by its end:
/// they are born evenly across it. Twin of `spawn_elapsed`.
pub fn spawn_elapsed(i: u32, n: u32, seconds: f32) -> f32 {
    seconds * (1.0 - (i as f32 + 0.5) / n.max(1) as f32)
}

/// A basis around `n`, as tangent, bitangent and `n`: twin of `orthonormal_basis`.
fn orthonormal_basis(n: Vec3) -> Mat3 {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    Mat3::from_cols(
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
        n,
    )
}

/// The particle numbered `index` of an emitter, as it is born. Twin of `spawn_particle`.
pub fn spawn(e: &EmitterRecord, index: u32) -> Particle {
    let r = |stream| random(e.seed, index, stream);
    let local = match e.shape_kind {
        SHAPE_SPHERE => {
            let z = 2.0 * r(0) - 1.0;
            let phi = TAU * r(1);
            let ring = (1.0 - z * z).max(0.0).sqrt();
            let radius = if e.flags & FLAG_SURFACE != 0 {
                e.shape_size[0]
            } else {
                e.shape_size[0] * r(2).powf(1.0 / 3.0)
            };
            Vec3::new(ring * phi.cos(), ring * phi.sin(), z) * radius
        }
        SHAPE_BOX => {
            Vec3::new(r(0) - 0.5, r(1) - 0.5, r(2) - 0.5) * Vec3::from_slice(&e.shape_size[..3])
        }
        _ => Vec3::ZERO,
    } + Vec3::from_slice(&e.offset_cone[..3]);
    let transform = Mat4::from_cols_array_2d(&e.transform);
    let position = transform.transform_point3(local);

    let base = Vec3::from_slice(&e.velocity_spread[..3]);
    let speed = base.length() * (1.0 + e.velocity_spread[3] * (2.0 * r(5) - 1.0));
    let axis = if speed > 0.0 {
        base.normalize()
    } else {
        Vec3::Y
    };
    let cos_theta = 1.0 - r(3) * (1.0 - e.offset_cone[3].cos());
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TAU * r(4);
    let direction = orthonormal_basis(axis)
        * Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    let velocity = transform.transform_vector3(direction * speed);

    let [life_low, life_high, size_low, size_high] = e.lifetime_size;
    Particle {
        position: position.into(),
        age: 0.0,
        velocity: velocity.into(),
        lifetime: life_low + (life_high - life_low) * r(7),
        previous: position.into(),
        size: (size_low + (size_high - size_low) * r(6)) * e.scale,
        index,
        rotation: TAU * r(8),
        _pad: [0; 2],
    }
}

/// Moves a particle on by `seconds` under its emitter's forces. Twin of `step_particle`.
///
/// Gravity and linear drag are integrated exactly, so the result does not depend on how time is
/// split into steps: a slow frame lands a particle where two fast ones would have.
pub fn step(p: &mut Particle, e: &EmitterRecord, seconds: f32) {
    let a = Vec3::from_slice(&e.acceleration_drag[..3]);
    let k = e.acceleration_drag[3];
    let h = seconds;
    let kh = k * h;
    // With e = exp(-k h): velocity becomes v e + a F and position moves by v F + a G, where
    // F = (1 - e) / k and G = (h - F) / k, the integrals of e over the step.
    let (decay, f, g) = if kh < SERIES_BELOW {
        (
            1.0 - kh + 0.5 * kh * kh,
            h * (1.0 - 0.5 * kh + kh * kh / 6.0),
            h * h * (0.5 - kh / 6.0 + kh * kh / 24.0),
        )
    } else {
        let decay = (-kh).exp();
        let f = (1.0 - decay) / k;
        (decay, f, (h - f) / k)
    };
    let v = Vec3::from(p.velocity);
    let x = Vec3::from(p.position);
    p.previous = p.position;
    p.position = (x + v * f + a * g).into();
    p.velocity = (v * decay + a * f).into();
    p.age += h;
}

/// Whether a particle has lived its life. Twin of `is_dead`.
pub fn is_dead(p: &Particle) -> bool {
    p.age >= p.lifetime
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> EmitterRecord {
        EmitterRecord {
            transform: Mat4::IDENTITY.to_cols_array_2d(),
            velocity_spread: [1.0, 4.0, 0.0, 0.1],
            acceleration_drag: [0.0, -9.81, 0.0, 0.4],
            lifetime_size: [5.0, 5.0, 0.1, 0.1],
            offset_cone: [0.0, 0.0, 0.0, 0.3],
            scale: 1.0,
            seed: 7,
            flags: FLAG_LIVE,
            ..Default::default()
        }
    }

    /// The step matches the closed form for gravity and linear drag, and does not depend on how
    /// time is split into steps.
    #[test]
    fn steps_follow_the_exact_motion_however_time_is_split() {
        let e = record();
        let born = spawn(&e, 3);
        let mut once = born;
        step(&mut once, &e, 1.2);
        let mut many = born;
        for h in [0.1, 0.05, 0.3, 0.25, 0.2, 0.3] {
            step(&mut many, &e, h);
        }
        let k = e.acceleration_drag[3];
        let a = Vec3::new(0.0, -9.81, 0.0);
        let v0 = Vec3::from(born.velocity);
        let t: f32 = 1.2;
        let terminal = a / k;
        let exact =
            Vec3::from(born.position) + terminal * t + (v0 - terminal) * (1.0 - (-k * t).exp()) / k;
        for p in [once, many] {
            assert!((Vec3::from(p.position) - exact).length() < 1.0e-4, "{p:?}");
            assert!((p.age - t).abs() < 1.0e-6);
        }
    }

    /// Without drag the series branch is plain projectile motion.
    #[test]
    fn without_drag_a_particle_flies_as_a_projectile() {
        let mut e = record();
        e.acceleration_drag[3] = 0.0;
        let mut p = spawn(&e, 0);
        let v0 = Vec3::from(p.velocity);
        step(&mut p, &e, 0.5);
        let expected = v0 * 0.5 + Vec3::new(0.0, -9.81, 0.0) * 0.125;
        assert!((Vec3::from(p.position) - expected).length() < 1.0e-5);
    }

    /// Initial velocities stay inside the cone and speeds inside their spread.
    #[test]
    fn spawns_stay_inside_their_cone_and_spread() {
        let e = record();
        let axis = Vec3::new(1.0, 4.0, 0.0).normalize();
        let speed = Vec3::new(1.0, 4.0, 0.0).length();
        let mut widest: f32 = 0.0;
        for index in 0..2000 {
            let p = spawn(&e, index);
            let v = Vec3::from(p.velocity);
            let angle = v.normalize().dot(axis).clamp(-1.0, 1.0).acos();
            widest = widest.max(angle);
            assert!(angle <= 0.3 + 1.0e-4, "{angle}");
            assert!((v.length() / speed - 1.0).abs() <= 0.1 + 1.0e-5);
        }
        // And they fill it, rather than hugging the axis.
        assert!(widest > 0.28, "{widest}");
    }

    #[test]
    fn shapes_hold_their_particles() {
        let mut e = record();
        e.shape_kind = SHAPE_SPHERE;
        e.shape_size = [2.0, 0.0, 0.0, 0.0];
        e.offset_cone = [0.0, 10.0, 0.0, 0.0];
        for index in 0..500 {
            let p = Vec3::from(spawn(&e, index).position);
            assert!((p - Vec3::new(0.0, 10.0, 0.0)).length() <= 2.0 + 1.0e-4);
        }
        e.flags |= FLAG_SURFACE;
        for index in 0..500 {
            let p = Vec3::from(spawn(&e, index).position);
            assert!(((p - Vec3::new(0.0, 10.0, 0.0)).length() - 2.0).abs() < 1.0e-4);
        }
        e.shape_kind = SHAPE_BOX;
        e.shape_size = [1.0, 2.0, 4.0, 0.0];
        e.offset_cone = [0.0; 4];
        e.transform = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)).to_cols_array_2d();
        for index in 0..500 {
            let p = Vec3::from(spawn(&e, index).position) - Vec3::new(5.0, 0.0, 0.0);
            assert!(p.abs().cmple(Vec3::new(0.5, 1.0, 2.0)).all(), "{p}");
        }
    }

    #[test]
    fn record_layouts_match_the_shaders() {
        // Sizes and alignment the WGSL structs have.
        assert_eq!(std::mem::size_of::<EmitterRecord>(), 192);
        assert_eq!(std::mem::size_of::<Particle>(), 64);
        assert_eq!(std::mem::size_of::<EmitterState>(), 64);
    }

    #[test]
    fn spawns_are_spread_evenly_over_the_step() {
        let ages: Vec<f32> = (0..4).map(|i| spawn_elapsed(i, 4, 1.0)).collect();
        assert_eq!(ages, vec![0.875, 0.625, 0.375, 0.125]);
    }
}
