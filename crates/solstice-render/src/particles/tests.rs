//! The simulation on a real GPU, against the CPU reference and the pool's invariants. They pass
//! trivially, with a note, if no GPU is available.

use super::*;
use crate::frame::FrameUniform;
use reference::{is_dead, spawn, spawn_elapsed, step};
use std::collections::{HashMap, HashSet};

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

/// A pool on the GPU, failing the test on any GPU error.
struct Rig {
    gpu: Gpu,
    particles: Particles,
    errors: Arc<Mutex<Vec<String>>>,
}

impl Rig {
    fn new(capacity: u32) -> Option<Self> {
        let gpu = request_gpu()?;
        let errors: Arc<Mutex<Vec<String>>> = Default::default();
        let sink = errors.clone();
        gpu.device
            .on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                sink.lock().unwrap().push(error.to_string());
            }));
        let frame = gpu.uniform_buffer("frame", std::mem::size_of::<FrameUniform>() as u64);
        let particles = Particles::new(&gpu, &frame, capacity);
        Some(Self {
            gpu,
            particles,
            errors,
        })
    }

    /// Simulates one frame of `seconds`.
    fn frame(&mut self, seconds: f32) {
        self.frame_at(seconds, Vec3::ZERO);
    }

    /// Simulates one frame of `seconds` with the camera at `camera`.
    fn frame_at(&mut self, seconds: f32, camera: Vec3) {
        self.particles.prepare(&self.gpu, camera, seconds, 1.0);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.particles.record_simulation(&mut encoder);
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let errors = self.errors.lock().unwrap();
        assert!(errors.is_empty(), "GPU errors: {}", errors.join("\n"));
    }

    fn read<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer) -> Vec<T> {
        let size = buffer.size();
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle test readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.gpu.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let data = staging.slice(..).get_mapped_range().unwrap();
        bytemuck::cast_slice(&data).to_vec()
    }

    /// Each live emitter's slot, slice, state, free list and alive particles, as the next frame
    /// will read them.
    fn emitters(&self) -> Vec<Snapshot> {
        let states: Vec<EmitterState> = self.read(&self.particles.pool.state);
        let records: Vec<EmitterRecord> = self.read(&self.particles.pool.emitters);
        let free: Vec<u32> = self.read(&self.particles.pool.free);
        let alive: Vec<u32> = self.read(&self.particles.pool.alive);
        let pool: Vec<Particle> = self.read(&self.particles.pool.particles);
        let capacity = self.particles.pool.capacity;
        let parity = self.particles.parity;
        (0..MAX_EMITTERS)
            .filter(|&slot| self.particles.slots[slot])
            .map(|slot| {
                let state = states[slot];
                let record = records[slot];
                let (base, budget) = (record.base, record.budget);
                let free = free[base as usize..(base + state.free_count) as usize].to_vec();
                let start = (parity * capacity + base) as usize;
                let alive: Vec<u32> =
                    alive[start..start + state.alive[parity as usize] as usize].to_vec();
                let particles = alive.iter().map(|&s| pool[s as usize]).collect();
                Snapshot {
                    slot,
                    base,
                    budget,
                    state,
                    record,
                    free,
                    alive,
                    particles,
                }
            })
            .collect()
    }
}

struct Snapshot {
    #[allow(dead_code)]
    slot: usize,
    base: u32,
    budget: u32,
    state: EmitterState,
    record: EmitterRecord,
    free: Vec<u32>,
    alive: Vec<u32>,
    particles: Vec<Particle>,
}

const BALLISTIC: &str = "
    emitter shot
      budget 2000
      spawn burst=1500
      shape sphere 0.5
      offset 1 2 3
      lifetime 30
      velocity 2 8 1 cone=25 spread=0.2
      size 0.1 0.2
      gravity
      drag 0.35
";

/// Frame times that vary as a real frame rate does.
const UNEVEN: [f32; 12] = [
    1.0 / 60.0,
    1.0 / 30.0,
    1.0 / 144.0,
    0.05,
    1.0 / 75.0,
    0.0021,
    1.0 / 20.0,
    1.0 / 60.0,
    0.004,
    1.0 / 90.0,
    0.031,
    1.0 / 60.0,
];

/// Every particle the GPU simulated is where the CPU reference puts it.
#[test]
fn gpu_trajectories_match_the_reference() {
    let Some(mut rig) = Rig::new(4096) else {
        return;
    };
    let effect = Effect::parse(BALLISTIC).unwrap();
    let transform = Mat4::from_rotation_y(0.7) * Mat4::from_scale(Vec3::splat(1.5));
    rig.particles.spawn(&effect, transform).unwrap();
    for seconds in UNEVEN {
        rig.frame(seconds);
    }
    let [emitter] = rig.emitters().try_into().ok().unwrap();
    assert_eq!(emitter.particles.len(), 1500);
    let mut worst: f32 = 0.0;
    for p in &emitter.particles {
        let mut expected = spawn(&emitter.record, p.index);
        step(
            &mut expected,
            &emitter.record,
            spawn_elapsed(p.index, 1500, UNEVEN[0]),
        );
        for &seconds in &UNEVEN[1..] {
            step(&mut expected, &emitter.record, seconds);
        }
        let error = (Vec3::from(p.position) - Vec3::from(expected.position)).length();
        let speed_error = (Vec3::from(p.velocity) - Vec3::from(expected.velocity)).length();
        worst = worst.max(error);
        assert!(error < 1.0e-3, "particle {} is {error} m off", p.index);
        assert!(
            speed_error < 1.0e-3,
            "particle {} is {speed_error} m/s off",
            p.index
        );
        assert!((p.age - expected.age).abs() < 1.0e-5);
        assert!((p.size - expected.size).abs() < 1.0e-6);
        assert!((p.lifetime - expected.lifetime).abs() < 1.0e-5);
    }
    // They really moved: a few metres up and out over a third of a second.
    let mean_height = emitter.particles.iter().map(|p| p.position[1]).sum::<f32>() / 1500.0;
    assert!(mean_height > 4.0, "{mean_height}");
    eprintln!("worst position error {worst} m");
}

/// Splitting the same span of time differently gives the same particles.
#[test]
fn uneven_frame_times_give_the_same_result_as_even_ones() {
    let total: f32 = UNEVEN[1..].iter().sum();
    let run = |times: &[f32]| {
        let mut rig = Rig::new(4096)?;
        let effect = Effect::parse(BALLISTIC).unwrap();
        rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
        rig.frame(UNEVEN[0]);
        for &seconds in times {
            rig.frame(seconds);
        }
        let [emitter] = rig.emitters().try_into().ok().unwrap();
        Some(
            emitter
                .particles
                .iter()
                .map(|p| (p.index, Vec3::from(p.position)))
                .collect::<HashMap<u32, Vec3>>(),
        )
    };
    let Some(uneven) = run(&UNEVEN[1..]) else {
        return;
    };
    let even = run(&[total / 11.0; 11]).unwrap();
    assert_eq!(uneven.len(), even.len());
    for (index, position) in &uneven {
        let other = even[index];
        assert!(
            (*position - other).length() < 1.0e-3,
            "particle {index}: {position} against {other}"
        );
    }
}

/// Over thousands of frames of particles being born and dying, every slot of every slice is
/// always either free or holding exactly one live particle, never both, never neither.
#[test]
fn particles_are_conserved_as_they_live_and_die() {
    let Some(mut rig) = Rig::new(8192) else {
        return;
    };
    let effect = Effect::parse(
        "
        emitter quick
          budget 700
          spawn rate=2400 burst=300 every=0.7
          lifetime 0.05 0.6
          velocity 0 1 0 cone=90
        emitter slow
          budget 1500
          spawn rate=500
          lifetime 1 4
        ",
    )
    .unwrap();
    rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
    rig.particles
        .spawn(&effect, Mat4::from_translation(Vec3::X))
        .unwrap();
    let mut checked = 0;
    for frame in 0..2400 {
        rig.frame(UNEVEN[frame % UNEVEN.len()]);
        if frame % 97 != 0 {
            continue;
        }
        let emitters = rig.emitters();
        assert_eq!(emitters.len(), 4);
        for e in &emitters {
            let total = e.state.free_count + e.state.alive[rig.particles.parity as usize];
            assert_eq!(
                total, e.budget,
                "frame {frame}: free and alive must fill the budget"
            );
            let mut seen = HashSet::new();
            for &slot in e.free.iter().chain(&e.alive) {
                assert!(
                    (e.base..e.base + e.budget).contains(&slot),
                    "slot {slot} is outside its slice"
                );
                assert!(
                    seen.insert(slot),
                    "frame {frame}: slot {slot} is listed twice"
                );
            }
            for p in &e.particles {
                assert!(!is_dead(p), "a dead particle is still alive");
            }
            checked += 1;
        }
    }
    assert!(checked > 40);
}

/// Spawns beyond an emitter's budget are dropped and counted, and its particles never spill out
/// of its slice into its neighbour's.
#[test]
fn spawns_beyond_the_budget_are_dropped_and_counted() {
    let Some(mut rig) = Rig::new(1024) else {
        return;
    };
    let greedy = Effect::parse("emitter e\n budget 100\n spawn burst=1000\n lifetime 10").unwrap();
    let quiet = Effect::parse("emitter e\n budget 200\n lifetime 10").unwrap();
    rig.particles.spawn(&greedy, Mat4::IDENTITY).unwrap();
    rig.particles.spawn(&quiet, Mat4::IDENTITY).unwrap();
    for _ in 0..3 {
        rig.frame(1.0 / 60.0);
    }
    let emitters = rig.emitters();
    let (greedy, quiet) = (&emitters[0], &emitters[1]);
    // The burst is clamped to the free list on the GPU, and the rest counted.
    assert_eq!(greedy.alive.len(), 100);
    assert_eq!(greedy.state.free_count, 0);
    assert_eq!(greedy.state.spawned, 100);
    assert_eq!(greedy.state.dropped, 900);
    assert!(greedy
        .alive
        .iter()
        .all(|s| (greedy.base..greedy.base + 100).contains(s)));
    assert_eq!(quiet.alive.len(), 0);
    assert_eq!(quiet.state.free_count, 200);
    let mut free = quiet.free.clone();
    free.sort_unstable();
    assert_eq!(free, (quiet.base..quiet.base + 200).collect::<Vec<_>>());

    // A rate the budget cannot hold: what does not fit is dropped and counted on the GPU.
    let mut rig = Rig::new(1024).unwrap();
    let flood = Effect::parse("emitter e\n budget 64\n spawn rate=6000\n lifetime 10").unwrap();
    rig.particles.spawn(&flood, Mat4::IDENTITY).unwrap();
    for _ in 0..10 {
        rig.frame(1.0 / 60.0);
    }
    let [flood] = rig.emitters().try_into().ok().unwrap();
    assert_eq!(flood.alive.len(), 64);
    assert_eq!(flood.state.spawned, 64);
    assert_eq!(flood.state.spawned + flood.state.dropped, 1000);
}

/// A rate spawns its count over time however the time is split into frames.
#[test]
fn rates_spawn_their_count_whatever_the_frame_rate() {
    let Some(mut rig) = Rig::new(4096) else {
        return;
    };
    let effect = Effect::parse("emitter e\n budget 2000\n spawn rate=240\n lifetime 100").unwrap();
    rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
    let mut elapsed = 0.0;
    let mut frame = 0;
    while elapsed < 4.0 - 1.0e-4 {
        let seconds = UNEVEN[frame % UNEVEN.len()].min(4.0 - elapsed);
        rig.frame(seconds);
        elapsed += seconds;
        frame += 1;
    }
    let [e] = rig.emitters().try_into().ok().unwrap();
    assert!(
        (959..=960).contains(&e.state.spawned),
        "{}",
        e.state.spawned
    );
    assert_eq!(e.alive.len() as u32, e.state.spawned);
}

/// A stopped effect lives out its particles and then gives its slice back; a killed one gives
/// it back at once.
#[test]
fn stopped_effects_end_and_their_slices_are_reused() {
    let Some(mut rig) = Rig::new(1000) else {
        return;
    };
    let effect = Effect::parse("emitter e\n budget 1000\n spawn rate=600\n lifetime 0.5").unwrap();
    let first = rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
    // The pool is full.
    assert!(rig.particles.spawn(&effect, Mat4::IDENTITY).is_none());
    for _ in 0..30 {
        rig.frame(1.0 / 30.0);
    }
    rig.particles.stop(first);
    for _ in 0..6 {
        rig.frame(1.0 / 30.0);
    }
    assert!(
        rig.particles.is_running(first),
        "it still has particles alive"
    );
    for _ in 0..30 {
        rig.frame(1.0 / 30.0);
    }
    assert!(!rig.particles.is_running(first));
    assert!(!rig.particles.set(first, "anything", 1.0));

    let second = rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
    assert_ne!(first, second);
    for _ in 0..10 {
        rig.frame(1.0 / 30.0);
    }
    let [e] = rig.emitters().try_into().ok().unwrap();
    // A fresh slice: it counts from zero, and holds only the new effect's particles.
    assert_eq!(e.state.spawned, 200);
    rig.particles.kill(second);
    assert!(!rig.particles.is_running(second));
    rig.frame(1.0 / 30.0);
    assert!(!rig.particles.is_active());
    assert!(rig.particles.spawn(&effect, Mat4::IDENTITY).is_some());
}

/// Changing the pool's size, as changing quality does, starts running effects again in the new
/// pool, with budgets that fit it, and nothing is lost or listed twice.
#[test]
fn a_running_effect_moves_to_a_resized_pool() {
    let Some(mut rig) = Rig::new(8192) else {
        return;
    };
    let effect = Effect::parse(
        "
        emitter a
          budget 3000
          spawn rate=3000
          lifetime 0.2 1.5
        emitter b
          budget 3000
          spawn rate=900
          lifetime 2
        ",
    )
    .unwrap();
    let id = rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
    for frame in 0..40 {
        rig.frame(UNEVEN[frame % UNEVEN.len()]);
    }
    rig.particles.resize(&rig.gpu, 4000);
    assert_eq!(rig.particles.capacity(), 4000);
    assert!(rig.particles.is_running(id));
    for frame in 0..200 {
        rig.frame(UNEVEN[frame % UNEVEN.len()]);
    }
    let emitters = rig.emitters();
    assert_eq!(emitters.len(), 2);
    // The first takes all it asked for; the second only what was left.
    assert_eq!((emitters[0].budget, emitters[1].budget), (3000, 1000));
    for e in &emitters {
        assert_eq!(
            e.state.free_count + e.state.alive[rig.particles.parity as usize],
            e.budget
        );
        let mut seen = HashSet::new();
        for &slot in e.free.iter().chain(&e.alive) {
            assert!((e.base..e.base + e.budget).contains(&slot));
            assert!(seen.insert(slot));
        }
        assert!(!e.alive.is_empty());
    }
}

/// An emitter that follows the camera spawns around it, wherever it goes.
#[test]
fn camera_following_emitters_spawn_around_the_camera() {
    let Some(mut rig) = Rig::new(4096) else {
        return;
    };
    let effect = Effect::parse(
        "emitter rain\n budget 2000\n follow camera\n spawn rate=6000\n shape box 20 4 20\n offset 0 10 0\n lifetime 5",
    )
    .unwrap();
    rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
    let camera = Vec3::new(140.0, 3.0, -60.0);
    rig.frame_at(1.0 / 60.0, camera);
    let [e] = rig.emitters().try_into().ok().unwrap();
    assert_eq!(e.particles.len(), 100);
    for p in &e.particles {
        let local = Vec3::from(p.position) - camera - Vec3::new(0.0, 10.0, 0.0);
        assert!(
            local.abs().cmple(Vec3::new(10.0, 2.0, 10.0)).all(),
            "{local}"
        );
    }
}

/// A paused frame, of no time, moves nothing and spawns nothing.
#[test]
fn a_paused_frame_moves_nothing() {
    let Some(mut rig) = Rig::new(4096) else {
        return;
    };
    let effect = Effect::parse(
        BALLISTIC
            .replace("burst=1500", "burst=500 rate=300")
            .as_str(),
    )
    .unwrap();
    rig.particles.spawn(&effect, Mat4::IDENTITY).unwrap();
    for seconds in &UNEVEN[..4] {
        rig.frame(*seconds);
    }
    let before = rig.emitters().remove(0);
    for _ in 0..5 {
        rig.frame(0.0);
    }
    let after = rig.emitters().remove(0);
    assert_eq!(after.state.spawned, before.state.spawned);
    let positions = |e: &Snapshot| -> HashMap<u32, [f32; 3]> {
        e.particles.iter().map(|p| (p.index, p.position)).collect()
    };
    assert_eq!(positions(&after), positions(&before));
}

/// A burst at low detail still gives at least one particle.
#[test]
fn bursts_survive_low_detail() {
    let effect = Effect::parse("emitter e\n spawn burst=1 every=0.5").unwrap();
    let mut emitter = LiveEmitter {
        slot: 0,
        base: 0,
        budget: 10,
        carry: 0.0,
        burst_started: false,
        until_burst: 0.0,
        longest_life: 0.0,
    };
    let desc = &effect.emitters[0];
    assert_eq!(spawns_due(&mut emitter, desc, &[], 0.1, 0.35), 1);
    assert_eq!(spawns_due(&mut emitter, desc, &[], 0.3, 0.35), 0);
    assert_eq!(spawns_due(&mut emitter, desc, &[], 0.2, 0.35), 1);
    // And a burst of nothing stays nothing.
    let none = Effect::parse("emitter e\n spawn rate=0").unwrap();
    emitter.burst_started = false;
    assert_eq!(
        spawns_due(&mut emitter, &none.emitters[0], &[], 0.1, 0.35),
        0
    );
}

/// Every record the shaders share with the CPU has the same size and field offsets in both, as
/// naga lays out the WGSL.
#[test]
fn shader_records_match_their_rust_twins() {
    let source = crate::shaders::expand("particles/simulate.wgsl");
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let layout = |name: &str| -> (u32, HashMap<String, u32>) {
        let ty = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no struct {name}"))
            .1;
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let offsets = members
            .iter()
            .map(|m| (m.name.clone().unwrap(), m.offset))
            .collect();
        (*span, offsets)
    };
    macro_rules! twin {
        ($wgsl:literal, $rust:ty, [$($field:ident),* $(,)?]) => {{
            let (span, offsets) = layout($wgsl);
            assert_eq!(span as usize, std::mem::size_of::<$rust>(), "size of {}", $wgsl);
            $(
                assert_eq!(
                    offsets[stringify!($field)] as usize,
                    std::mem::offset_of!($rust, $field),
                    "{}.{}", $wgsl, stringify!($field)
                );
            )*
        }};
    }
    twin!(
        "Emitter",
        EmitterRecord,
        [
            transform,
            shape_size,
            offset_cone,
            velocity_spread,
            acceleration_drag,
            lifetime_size,
            color,
            base,
            budget,
            spawn_request,
            seed,
            shape_kind,
            flags,
            soft,
            scale,
        ]
    );
    twin!(
        "Particle",
        Particle,
        [position, age, velocity, lifetime, previous, size, index, rotation,]
    );
    twin!(
        "EmitterState",
        EmitterState,
        [
            free_count,
            alive,
            spawned,
            dropped,
            spawn_count,
            spawn_first,
            spawn_index,
            update_count,
            spawn_start,
            update_start,
        ]
    );
    twin!("SimParams", SimUniform, [seconds, parity, capacity]);
}

#[test]
fn ranges_split_and_join() {
    let mut ranges = vec![(0, 100)];
    let a = allocate(&mut ranges, 30).unwrap();
    let b = allocate(&mut ranges, 30).unwrap();
    let c = allocate(&mut ranges, 60).unwrap();
    assert_eq!((a, b, c), ((0, 30), (30, 30), (60, 40)));
    assert!(allocate(&mut ranges, 1).is_none());
    release(&mut ranges, b.0, b.1);
    release(&mut ranges, a.0, a.1);
    assert_eq!(ranges, vec![(0, 60)]);
    release(&mut ranges, c.0, c.1);
    assert_eq!(ranges, vec![(0, 100)]);
}
