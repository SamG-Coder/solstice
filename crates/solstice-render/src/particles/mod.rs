//! Particles: effects made of emitters, simulated on the GPU and drawn over the lit scene
//! (docs/particles.md).
//!
//! * [`effect`]: the text form of an effect and its modules.
//! * [`reference`]: the records the shaders share with the CPU, and a CPU reference for what
//!   they do.
//!
//! One pool of particles serves every emitter. Each live emitter has a fixed slice of it, its
//! budget, and keeps a free list and two alive lists there, which take turns being read and
//! written. Each frame is four small dispatches however many emitters there are (plan, spawn,
//! update, finish), then one indirect draw per emitter. The CPU never reads a particle: it
//! works out each emitter's values from its effect's parameters, how many particles to spawn,
//! and where in the pool each emitter lives, and statistics come back asynchronously.
//!
//! With no effect running, nothing is dispatched or drawn.

pub mod effect;
pub mod reference;

pub use effect::{Blend, Curve, Effect, Emitter, Follow, Shading, Shape, Value};

use crate::gpu::{Gpu, LayoutBuilder};
use crate::shaders;
use effect::CURVE_ENTRIES;
use glam::{Mat4, Vec3};
use reference::{
    EmitterRecord, EmitterState, Particle, FLAG_ADD, FLAG_GLOW, FLAG_LIT, FLAG_LIVE, FLAG_SURFACE,
    SHAPE_BOX, SHAPE_POINT, SHAPE_SPHERE,
};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Emitters alive at once, across every effect. The plan pass gives each a thread of one
/// workgroup. Twin of `MAX_EMITTERS` in `particles/common.wgsl`.
pub const MAX_EMITTERS: usize = 64;
/// Particles in the pool at each quality.
pub const POOL_HIGH: u32 = 262_144;
pub const POOL_MEDIUM: u32 = 131_072;
pub const POOL_LOW: u32 = 65_536;
/// Each emitter's curves on the GPU: its colour's entries, then its size's four to a vector.
const CURVE_STRIDE: usize = CURVE_ENTRIES + CURVE_ENTRIES / 4;
/// Draws read their emitter's record at this stride, the uniform offset alignment WebGPU allows.
const DRAW_STRIDE: u64 = 256;
/// Where the dispatches' indirect arguments are: spawn, update and light, then the draws'.
const SPAWN_ARGS: u64 = 0;
const UPDATE_ARGS: u64 = 12;
const LIGHT_ARGS: u64 = 24;
const DRAWS_OFFSET: u64 = 48;
/// Bytes of light each particle holds: twin of `ParticleLight` in `particles/light_common.wgsl`.
const LIGHT_BYTES: u64 = 32;
const INDIRECT_BYTES: u64 = DRAWS_OFFSET + MAX_EMITTERS as u64 * 16;
const STATE_BYTES: u64 = (MAX_EMITTERS * std::mem::size_of::<EmitterState>()) as u64;
/// Seconds a stopped emitter's slice is held after its longest-lived particle should have died.
const RELEASE_MARGIN: f32 = 0.5;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Twin of `SimParams` in `particles/common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimUniform {
    seconds: f32,
    parity: u32,
    capacity: u32,
    light_interval: u32,
    ray_steps: u32,
    _pad: [u32; 3],
}

/// A running effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectId {
    index: u32,
    generation: u32,
}

/// How full the pool is, from a recent frame. Counts lag a few frames, as readback is
/// asynchronous.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParticleStats {
    /// Particles alive.
    pub alive: u32,
    /// Particles spawned and spawns dropped for lack of budget, by the emitters running now
    /// since each started.
    pub spawned: u64,
    pub dropped: u64,
    /// Emitters running, and the slots of the pool their budgets hold.
    pub emitters: u32,
    pub reserved: u32,
    pub capacity: u32,
}

struct LiveEmitter {
    slot: usize,
    base: u32,
    budget: u32,
    /// Spawns owed but not yet whole.
    carry: f32,
    /// Whether the first burst has gone, and seconds until the next.
    burst_started: bool,
    until_burst: f32,
    /// The longest a particle of this emitter may have been given to live.
    longest_life: f32,
}

struct LiveEffect {
    effect: Effect,
    params: Vec<f32>,
    transform: Mat4,
    emitters: Vec<LiveEmitter>,
    /// Seconds since the effect was stopped, if it has been.
    stopped_for: Option<f32>,
    /// Which effect this is, counting every one spawned, for its seeds.
    serial: u32,
}

/// What the light pass reads of the rest of the renderer, besides the environment group: the
/// scene's local lights and the sun's shadow maps.
#[derive(Clone)]
pub struct LightInputs {
    pub lights: wgpu::Buffer,
    pub shadow_uniform: wgpu::Buffer,
    pub shadow_maps: wgpu::TextureView,
    pub shadow_sampler: wgpu::Sampler,
}

/// Size-dependent targets: the depth attachment particles test against, made from the scene's
/// depth each frame.
pub struct ParticleTargets {
    depth: wgpu::TextureView,
    copy_group: wgpu::BindGroup,
    scene_group: wgpu::BindGroup,
}

/// The buffers of a pool of one size, and their bind groups.
struct Pool {
    capacity: u32,
    // Held by the bind groups too; tests read them back.
    #[cfg_attr(not(test), allow(dead_code))]
    particles: wgpu::Buffer,
    free: wgpu::Buffer,
    #[cfg_attr(not(test), allow(dead_code))]
    alive: wgpu::Buffer,
    state: wgpu::Buffer,
    emitters: wgpu::Buffer,
    curves: wgpu::Buffer,
    indirect: wgpu::Buffer,
    sim: wgpu::Buffer,
    draws: wgpu::Buffer,
    /// The light each particle receives, by slot.
    // Held by the bind groups.
    #[allow(dead_code)]
    light: wgpu::Buffer,
    plan_group: wgpu::BindGroup,
    simulate_group: wgpu::BindGroup,
    light_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
}

/// The layouts a pool's bind groups are made with.
struct PoolLayouts {
    plan: wgpu::BindGroupLayout,
    simulate: wgpu::BindGroupLayout,
    light: wgpu::BindGroupLayout,
    draw: wgpu::BindGroupLayout,
}

/// Every effect running, the pool their particles live in, and the passes that move and draw
/// them.
pub struct Particles {
    pool: Pool,
    layouts: PoolLayouts,
    inputs: LightInputs,
    light: wgpu::ComputePipeline,
    scene_layout: wgpu::BindGroupLayout,
    copy_layout: wgpu::BindGroupLayout,
    plan: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
    depth_copy: wgpu::RenderPipeline,
    frame_group: wgpu::BindGroup,
    effects: Vec<Option<LiveEffect>>,
    generations: Vec<u32>,
    slots: [bool; MAX_EMITTERS],
    /// Free ranges of the pool, as (first slot, count), in order and never touching.
    ranges: Vec<(u32, u32)>,
    /// Emitter slots whose GPU state must be reset before the next frame.
    pending: Vec<usize>,
    /// The alive lists read by the next frame's simulation.
    parity: u32,
    serial: u32,
    /// Whether this frame simulates and draws anything.
    active: bool,
    /// Each particle's light is worked out afresh once in this many frames, with ambient rays
    /// of this many steps.
    light_interval: u32,
    ray_steps: u32,
    stats: StatsReadback,
}

impl Particles {
    /// Creates the passes and a pool of `capacity` particles, or as many as the device can
    /// bind. `frame` is the renderer's frame uniform, which the draw reads; the light pass reads
    /// the environment group, made with `environment`, and `inputs`.
    pub fn new(
        gpu: &Gpu,
        frame: &wgpu::Buffer,
        capacity: u32,
        environment: &wgpu::BindGroupLayout,
        inputs: LightInputs,
    ) -> Self {
        let plan_layout = LayoutBuilder::new()
            .uniform()
            .storage_read()
            .storage_write()
            .storage_write()
            .build(gpu, "particle plan");
        let simulate_layout = LayoutBuilder::new()
            .uniform()
            .storage_read()
            .storage_write()
            .storage_write()
            .storage_write()
            .storage_write()
            .build(gpu, "particle simulate");
        let light_layout = crate::shadow::ShadowMaps::add_to_layout(
            LayoutBuilder::new()
                .uniform()
                .storage_read()
                .storage_write()
                .storage_read()
                .storage_read()
                .storage_write()
                .storage_read(),
        )
        .build(gpu, "particle light");
        let frame_layout = LayoutBuilder::new()
            .visibility(wgpu::ShaderStages::VERTEX_FRAGMENT)
            .uniform()
            .build(gpu, "particle frame");
        let pool_draw_layout = LayoutBuilder::new()
            .visibility(wgpu::ShaderStages::VERTEX)
            .storage_read()
            .storage_read()
            .visibility(wgpu::ShaderStages::VERTEX_FRAGMENT)
            .storage_read()
            .visibility(wgpu::ShaderStages::VERTEX)
            .storage_read()
            .visibility(wgpu::ShaderStages::VERTEX_FRAGMENT)
            .uniform_dynamic()
            .visibility(wgpu::ShaderStages::VERTEX)
            .storage_read()
            .build(gpu, "particle draw");
        let scene_layout = LayoutBuilder::new()
            .visibility(wgpu::ShaderStages::FRAGMENT)
            .texture_unfilterable()
            .build(gpu, "particle scene depth");
        let copy_layout = LayoutBuilder::new()
            .visibility(wgpu::ShaderStages::FRAGMENT)
            .texture_unfilterable()
            .build(gpu, "particle depth copy");

        let plan_module = shaders::module(&gpu.device, "particles/plan.wgsl");
        let simulate_module = shaders::module(&gpu.device, "particles/simulate.wgsl");
        let plan = gpu.compute_pipeline("particle plan", &plan_module, "plan", &[&plan_layout]);
        let finish =
            gpu.compute_pipeline("particle finish", &plan_module, "finish", &[&plan_layout]);
        let spawn = gpu.compute_pipeline(
            "particle spawn",
            &simulate_module,
            "spawn",
            &[&simulate_layout],
        );
        let update = gpu.compute_pipeline(
            "particle update",
            &simulate_module,
            "update",
            &[&simulate_layout],
        );
        let light = gpu.compute_pipeline(
            "particle light",
            &shaders::module(&gpu.device, "particles/light.wgsl"),
            "main",
            &[&light_layout, environment],
        );

        let draw_module = shaders::module(&gpu.device, "particles/draw.wgsl");
        let draw_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle draw"),
                bind_group_layouts: &[
                    Some(&frame_layout),
                    Some(&pool_draw_layout),
                    Some(&scene_layout),
                ],
                immediate_size: 0,
            });
        // Premultiplied: what is behind is covered by the particle's opacity, and its light
        // added. Additive emitters write no opacity, so they only add.
        let premultiplied = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let draw = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particle draw"),
                layout: Some(&draw_layout),
                vertex: wgpu::VertexState {
                    module: &draw_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                // Tested against the scene, which is nearer where its reversed depth is
                // greater, and never written: particles do not hide one another.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &draw_module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: premultiplied,
                            alpha: premultiplied,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let copy_module = shaders::module(&gpu.device, "particles/depth_copy.wgsl");
        let copy_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("particle depth copy"),
                    bind_group_layouts: &[Some(&copy_layout)],
                    immediate_size: 0,
                });
        let depth_copy = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particle depth copy"),
                layout: Some(&copy_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &copy_module,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                // Shares the particles' pass, so it has their colour target, but writes none
                // of it.
                fragment: Some(wgpu::FragmentState {
                    module: &copy_module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let frame_group = gpu.bind_group(
            "particle frame",
            &frame_layout,
            &[frame.as_entire_binding()],
        );
        let capacity = fitted_capacity(gpu, capacity);
        let layouts = PoolLayouts {
            plan: plan_layout,
            simulate: simulate_layout,
            light: light_layout,
            draw: pool_draw_layout,
        };
        let pool = Pool::new(gpu, capacity, &layouts, &inputs);
        Self {
            pool,
            layouts,
            inputs,
            light,
            scene_layout,
            copy_layout,
            plan,
            finish,
            spawn,
            update,
            draw,
            depth_copy,
            frame_group,
            effects: Vec::new(),
            generations: Vec::new(),
            slots: [false; MAX_EMITTERS],
            ranges: vec![(0, capacity)],
            pending: Vec::new(),
            parity: 0,
            serial: 0,
            active: false,
            light_interval: 1,
            ray_steps: 16,
            stats: StatsReadback::new(gpu),
        }
    }

    /// Works out each particle's light afresh once in `frames` frames, staggered across them,
    /// keeping what it had in between: the light pass's cost is divided by about as much, for
    /// light that lags a moving particle by as many frames. New particles are always lit at once.
    pub fn set_light_interval(&mut self, frames: u32) {
        self.light_interval = frames.max(1);
    }

    /// Steps each ambient ray takes through the GI volume, over its 30 m reach: fewer are
    /// cheaper, and miss more of what is near.
    pub fn set_ray_steps(&mut self, steps: u32) {
        self.ray_steps = steps.max(1);
    }

    /// Particles the pool holds.
    pub fn capacity(&self) -> u32 {
        self.pool.capacity
    }

    /// Makes the pool hold `capacity` particles. Running effects start again in the new pool,
    /// with budgets that fit it.
    pub fn resize(&mut self, gpu: &Gpu, capacity: u32) {
        let capacity = fitted_capacity(gpu, capacity);
        if capacity == self.pool.capacity {
            return;
        }
        self.pool = Pool::new(gpu, capacity, &self.layouts, &self.inputs);
        self.ranges = vec![(0, capacity)];
        self.slots = [false; MAX_EMITTERS];
        self.pending.clear();
        self.parity = 0;
        for index in 0..self.effects.len() {
            let Some(mut live) = self.effects[index].take() else {
                continue;
            };
            let budgets: Vec<u32> = live.effect.emitters.iter().map(|e| e.budget).collect();
            match self.place(&budgets) {
                Some(placed) => {
                    live.emitters = placed;
                    self.effects[index] = Some(live);
                }
                None => {
                    log::warn!("a particle effect no longer fits the pool, and has stopped");
                    self.generations[index] = self.generations[index].wrapping_add(1);
                }
            }
        }
    }

    /// Starts an effect at `transform`, or returns `None`, with a warning, if the pool or the
    /// emitter slots are full. An effect started while the pool is nearly full gets smaller
    /// budgets than it asks for.
    pub fn spawn(&mut self, effect: &Effect, transform: Mat4) -> Option<EffectId> {
        let budgets: Vec<u32> = effect.emitters.iter().map(|e| e.budget).collect();
        let Some(emitters) = self.place(&budgets) else {
            log::warn!("no room for another particle effect");
            return None;
        };
        let live = LiveEffect {
            effect: effect.clone(),
            params: effect.params.iter().map(|(_, v)| *v).collect(),
            transform,
            emitters,
            stopped_for: None,
            serial: self.serial,
        };
        self.serial = self.serial.wrapping_add(1);
        let index = match self.effects.iter().position(Option::is_none) {
            Some(index) => index,
            None => {
                self.effects.push(None);
                self.generations.push(0);
                self.effects.len() - 1
            }
        };
        self.effects[index] = Some(live);
        Some(EffectId {
            index: index as u32,
            generation: self.generations[index],
        })
    }

    /// Gives each budget a slice of the pool and an emitter slot, or none at all.
    fn place(&mut self, budgets: &[u32]) -> Option<Vec<LiveEmitter>> {
        let mut placed: Vec<LiveEmitter> = Vec::new();
        for &wanted in budgets {
            let slot = (0..MAX_EMITTERS).find(|&s| !self.slots[s]);
            let range = allocate(&mut self.ranges, wanted);
            match (slot, range) {
                (Some(slot), Some((base, budget))) => {
                    self.slots[slot] = true;
                    self.pending.push(slot);
                    placed.push(LiveEmitter {
                        slot,
                        base,
                        budget,
                        carry: 0.0,
                        burst_started: false,
                        until_burst: 0.0,
                        longest_life: 0.0,
                    });
                }
                (_, range) => {
                    if let Some((base, budget)) = range {
                        release(&mut self.ranges, base, budget);
                    }
                    for emitter in placed {
                        self.free_emitter(&emitter);
                    }
                    return None;
                }
            }
        }
        Some(placed)
    }

    fn free_emitter(&mut self, emitter: &LiveEmitter) {
        self.slots[emitter.slot] = false;
        self.pending.retain(|&s| s != emitter.slot);
        release(&mut self.ranges, emitter.base, emitter.budget);
    }

    fn live_mut(&mut self, id: EffectId) -> Option<&mut LiveEffect> {
        let index = id.index as usize;
        if self.generations.get(index) != Some(&id.generation) {
            return None;
        }
        self.effects.get_mut(index)?.as_mut()
    }

    /// Whether the effect is still running, or still has particles alive after being stopped.
    pub fn is_running(&self, id: EffectId) -> bool {
        let index = id.index as usize;
        self.generations.get(index) == Some(&id.generation)
            && self.effects.get(index).is_some_and(Option::is_some)
    }

    /// Sets one of the effect's parameters, from the next frame. Returns false if the effect
    /// has no parameter of that name, or has ended.
    pub fn set(&mut self, id: EffectId, name: &str, value: f32) -> bool {
        let Some(live) = self.live_mut(id) else {
            return false;
        };
        match live.effect.param(name) {
            Some(p) => {
                live.params[p] = value;
                true
            }
            None => false,
        }
    }

    /// Moves the effect. Its particles already alive stay where they are in the world.
    pub fn set_transform(&mut self, id: EffectId, transform: Mat4) {
        if let Some(live) = self.live_mut(id) {
            live.transform = transform;
        }
    }

    /// Stops the effect spawning. What is alive lives out its life, and then the effect ends.
    pub fn stop(&mut self, id: EffectId) {
        if let Some(live) = self.live_mut(id) {
            live.stopped_for.get_or_insert(0.0);
        }
    }

    /// Ends the effect at once, with all its particles.
    pub fn kill(&mut self, id: EffectId) {
        if self.live_mut(id).is_some() {
            self.end(id.index as usize);
        }
    }

    /// Ends every effect at once.
    pub fn clear(&mut self) {
        for index in 0..self.effects.len() {
            if self.effects[index].is_some() {
                self.end(index);
            }
        }
    }

    fn end(&mut self, index: usize) {
        if let Some(live) = self.effects[index].take() {
            for emitter in &live.emitters {
                self.free_emitter(emitter);
            }
            self.generations[index] = self.generations[index].wrapping_add(1);
        }
    }

    /// The pool's use, from a recent frame.
    pub fn stats(&self) -> ParticleStats {
        let mut stats = self.stats.latest();
        stats.capacity = self.pool.capacity;
        stats.reserved = self
            .effects
            .iter()
            .flatten()
            .flat_map(|e| &e.emitters)
            .map(|e| e.budget)
            .sum();
        stats
    }

    /// Whether this frame has anything to simulate and draw.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Works out this frame's emitters, `seconds` after the last, with spawn rates scaled by
    /// `detail`, and writes them for the GPU. Ends effects that have died out.
    pub fn prepare(&mut self, gpu: &Gpu, camera: Vec3, seconds: f32, detail: f32) {
        // Effects stopped long enough ago that nothing of theirs can be alive.
        for index in 0..self.effects.len() {
            let done = self.effects[index].as_mut().is_some_and(|live| {
                let Some(stopped) = live.stopped_for.as_mut() else {
                    return false;
                };
                *stopped += seconds;
                let longest = live
                    .emitters
                    .iter()
                    .map(|e| e.longest_life)
                    .fold(0.0, f32::max);
                *stopped > longest + RELEASE_MARGIN
            });
            if done {
                self.end(index);
            }
        }

        self.active = self.slots.iter().any(|&s| s);
        if !self.active {
            self.pending.clear();
            return;
        }

        let capacity = self.pool.capacity;
        let mut records = [EmitterRecord::default(); MAX_EMITTERS];
        let mut draws = vec![0u8; MAX_EMITTERS * DRAW_STRIDE as usize];
        let written = 1 - self.parity;
        for live in self.effects.iter_mut().flatten() {
            let stopped = live.stopped_for.is_some();
            for (number, emitter) in live.emitters.iter_mut().enumerate() {
                let desc = &live.effect.emitters[number];
                let params = &live.params;
                let transform = match desc.follow {
                    Follow::Effect => live.transform,
                    Follow::Camera => Mat4::from_translation(camera) * live.transform,
                };
                let request = if stopped {
                    0
                } else {
                    spawns_due(emitter, desc, params, seconds, detail)
                };
                let record = emitter_record(
                    desc,
                    params,
                    transform,
                    emitter,
                    request,
                    live.serial,
                    number,
                );
                emitter.longest_life = emitter
                    .longest_life
                    .max(record.lifetime_size[0])
                    .max(record.lifetime_size[1]);
                records[emitter.slot] = record;
                let at = emitter.slot * DRAW_STRIDE as usize;
                let list_offset = written * capacity + emitter.base;
                draws[at..at + 4].copy_from_slice(&list_offset.to_le_bytes());
                draws[at + 4..at + 8].copy_from_slice(&(emitter.slot as u32).to_le_bytes());
            }
        }

        // New emitters start with an empty slice: every slot free, nothing alive.
        let queue = &gpu.queue;
        for slot in std::mem::take(&mut self.pending) {
            let Some((desc, emitter)) = self.effects.iter().flatten().find_map(|live| {
                live.emitters
                    .iter()
                    .position(|e| e.slot == slot)
                    .map(|n| (&live.effect.emitters[n], &live.emitters[n]))
            }) else {
                continue;
            };
            let state = EmitterState {
                free_count: emitter.budget,
                ..Default::default()
            };
            queue.write_buffer(
                &self.pool.state,
                (slot * std::mem::size_of::<EmitterState>()) as u64,
                bytemuck::bytes_of(&state),
            );
            let free: Vec<u32> = (emitter.base..emitter.base + emitter.budget).collect();
            queue.write_buffer(
                &self.pool.free,
                u64::from(emitter.base) * 4,
                bytemuck::cast_slice(&free),
            );
            queue.write_buffer(
                &self.pool.curves,
                (slot * CURVE_STRIDE * 16) as u64,
                bytemuck::cast_slice(&baked_curves(desc)),
            );
        }
        queue.write_buffer(&self.pool.emitters, 0, bytemuck::cast_slice(&records));
        queue.write_buffer(&self.pool.draws, 0, &draws);
        let sim = SimUniform {
            seconds,
            parity: self.parity,
            capacity,
            light_interval: self.light_interval.max(1),
            ray_steps: self.ray_steps.max(1),
            _pad: [0; 3],
        };
        queue.write_buffer(&self.pool.sim, 0, bytemuck::bytes_of(&sim));
        self.stats.live = self.slots;
        self.stats.parity = written;
        self.parity = written;
    }

    /// Records this frame's simulation, if [`Self::prepare`] found anything to do.
    pub fn record_simulation(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.active {
            return;
        }
        let pool = &self.pool;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particles"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.plan);
        pass.set_bind_group(0, &pool.plan_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_bind_group(0, &pool.simulate_group, &[]);
        pass.set_pipeline(&self.spawn);
        pass.dispatch_workgroups_indirect(&pool.indirect, SPAWN_ARGS);
        pass.set_pipeline(&self.update);
        pass.dispatch_workgroups_indirect(&pool.indirect, UPDATE_ARGS);
        pass.set_pipeline(&self.finish);
        pass.set_bind_group(0, &pool.plan_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Records the light each live particle receives this frame, after the simulation, with
    /// the renderer's environment group: the sun, its shadows, the sky, the GI volume and the
    /// scene's lights.
    pub fn record_light(&self, encoder: &mut wgpu::CommandEncoder, environment: &wgpu::BindGroup) {
        if !self.active {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle light"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.light);
        pass.set_bind_group(0, &self.pool.light_group, &[]);
        pass.set_bind_group(1, environment, &[]);
        pass.dispatch_workgroups_indirect(&self.pool.indirect, LIGHT_ARGS);
    }

    /// The depth target and bind groups for drawing at one size, over a scene whose depth is
    /// `scene_depth`.
    pub fn targets(
        &self,
        gpu: &Gpu,
        size: (u32, u32),
        scene_depth: &wgpu::TextureView,
    ) -> ParticleTargets {
        let depth = gpu
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("particle depth"),
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let view = wgpu::BindingResource::TextureView(scene_depth);
        ParticleTargets {
            depth,
            copy_group: gpu.bind_group(
                "particle depth copy",
                &self.copy_layout,
                std::slice::from_ref(&view),
            ),
            scene_group: gpu.bind_group("particle scene depth", &self.scene_layout, &[view]),
        }
    }

    /// Draws every emitter's particles over `color`. With `overdraw`, `color` is cleared first
    /// and each layer of particle adds a step of the overdraw view's heat, so with nothing
    /// running the view is black.
    pub fn record_draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        targets: &ParticleTargets,
        overdraw: bool,
    ) {
        if !self.active && !overdraw {
            return;
        }
        let load = if overdraw {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particle draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !self.active {
            return;
        }
        pass.set_pipeline(&self.depth_copy);
        pass.set_bind_group(0, &targets.copy_group, &[]);
        pass.draw(0..3, 0..1);

        pass.set_pipeline(&self.draw);
        pass.set_bind_group(0, &self.frame_group, &[]);
        pass.set_bind_group(2, &targets.scene_group, &[]);
        for slot in (0..MAX_EMITTERS).filter(|&s| self.slots[s]) {
            pass.set_bind_group(
                1,
                &self.pool.draw_group,
                &[(slot as u64 * DRAW_STRIDE) as u32],
            );
            pass.draw_indirect(&self.pool.indirect, DRAWS_OFFSET + slot as u64 * 16);
        }
    }

    /// Copies this frame's counters for the statistics, when the last copy has been read.
    pub fn capture_stats(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.active {
            self.stats.capture(encoder, &self.pool.state);
        } else {
            self.stats.idle();
        }
    }

    /// Starts reading back the counters captured this frame. Call after submitting.
    pub fn map_stats(&self) {
        self.stats.request_map();
    }
}

/// Twin of `GROUP_SIZE` in `particles/common.wgsl`.
const GROUP_SIZE_U32: u32 = 64;

/// `wanted` particles, or as many as the device can bind in one storage buffer, and at least a
/// workgroup's worth.
fn fitted_capacity(gpu: &Gpu, wanted: u32) -> u32 {
    let particle = std::mem::size_of::<Particle>() as u64;
    let largest = gpu
        .limits
        .max_storage_buffer_binding_size
        .min(gpu.limits.max_buffer_size)
        / particle;
    (u64::from(wanted).min(largest) as u32).max(GROUP_SIZE_U32)
}

impl Pool {
    fn new(gpu: &Gpu, capacity: u32, layouts: &PoolLayouts, inputs: &LightInputs) -> Self {
        let copy = wgpu::BufferUsages::COPY_DST;
        let cap = u64::from(capacity);
        let particles = gpu.storage_buffer(
            "particles",
            cap * std::mem::size_of::<Particle>() as u64,
            wgpu::BufferUsages::COPY_SRC,
        );
        let free = gpu.storage_buffer(
            "particle free lists",
            cap * 4,
            copy | wgpu::BufferUsages::COPY_SRC,
        );
        let alive = gpu.storage_buffer(
            "particle alive lists",
            2 * cap * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let state = gpu.storage_buffer(
            "particle emitter state",
            STATE_BYTES,
            copy | wgpu::BufferUsages::COPY_SRC,
        );
        let emitters = gpu.storage_buffer(
            "particle emitters",
            (MAX_EMITTERS * std::mem::size_of::<EmitterRecord>()) as u64,
            copy | wgpu::BufferUsages::COPY_SRC,
        );
        let curves = gpu.storage_buffer(
            "particle curves",
            (MAX_EMITTERS * CURVE_STRIDE * 16) as u64,
            copy,
        );
        let indirect = gpu.storage_buffer(
            "particle indirect",
            INDIRECT_BYTES,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        );
        let light = gpu.storage_buffer(
            "particle light",
            cap * LIGHT_BYTES,
            wgpu::BufferUsages::COPY_SRC,
        );
        let sim = gpu.uniform_buffer("particle sim", std::mem::size_of::<SimUniform>() as u64);
        let draws = gpu.uniform_buffer("particle draws", MAX_EMITTERS as u64 * DRAW_STRIDE);
        fn whole(b: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
            b.as_entire_binding()
        }
        let plan_group = gpu.bind_group(
            "particle plan",
            &layouts.plan,
            &[
                whole(&sim),
                whole(&emitters),
                whole(&state),
                whole(&indirect),
            ],
        );
        let light_group = gpu.bind_group(
            "particle light",
            &layouts.light,
            &[
                whole(&sim),
                whole(&emitters),
                whole(&state),
                whole(&particles),
                whole(&alive),
                whole(&light),
                whole(&inputs.lights),
                whole(&inputs.shadow_uniform),
                wgpu::BindingResource::TextureView(&inputs.shadow_maps),
                wgpu::BindingResource::Sampler(&inputs.shadow_sampler),
            ],
        );
        let simulate_group = gpu.bind_group(
            "particle simulate",
            &layouts.simulate,
            &[
                whole(&sim),
                whole(&emitters),
                whole(&state),
                whole(&particles),
                whole(&free),
                whole(&alive),
            ],
        );
        let draw_group = gpu.bind_group(
            "particle draw",
            &layouts.draw,
            &[
                whole(&particles),
                whole(&alive),
                whole(&emitters),
                whole(&curves),
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &draws,
                    offset: 0,
                    size: NonZeroU64::new(16),
                }),
                whole(&light),
            ],
        );
        Self {
            capacity,
            particles,
            free,
            alive,
            state,
            emitters,
            curves,
            indirect,
            sim,
            draws,
            light,
            plan_group,
            simulate_group,
            light_group,
            draw_group,
        }
    }
}

/// Takes `wanted` slots from the free ranges, or as many as the largest range holds if none
/// holds that many. Returns the first slot and the count, or `None` if the pool is full.
fn allocate(ranges: &mut Vec<(u32, u32)>, wanted: u32) -> Option<(u32, u32)> {
    let index = ranges
        .iter()
        .position(|&(_, count)| count >= wanted)
        .or_else(|| (0..ranges.len()).max_by_key(|&i| ranges[i].1))?;
    let (first, count) = ranges[index];
    let taken = wanted.min(count);
    if taken == 0 {
        return None;
    }
    if taken == count {
        ranges.remove(index);
    } else {
        ranges[index] = (first + taken, count - taken);
    }
    Some((first, taken))
}

/// Returns slots to the free ranges, joining them to their neighbours.
fn release(ranges: &mut Vec<(u32, u32)>, first: u32, count: u32) {
    let index = ranges.partition_point(|&(f, _)| f < first);
    ranges.insert(index, (first, count));
    if index + 1 < ranges.len() && ranges[index].0 + ranges[index].1 == ranges[index + 1].0 {
        ranges[index].1 += ranges[index + 1].1;
        ranges.remove(index + 1);
    }
    if index > 0 && ranges[index - 1].0 + ranges[index - 1].1 == ranges[index].0 {
        ranges[index - 1].1 += ranges[index].1;
        ranges.remove(index);
    }
}

/// How many particles an emitter should spawn over the next `seconds`: its rate's share, with
/// what was owed and not yet whole carried over, and its bursts as they fall due.
fn spawns_due(
    emitter: &mut LiveEmitter,
    desc: &Emitter,
    params: &[f32],
    seconds: f32,
    detail: f32,
) -> u32 {
    let rate = desc.rate.eval(params).max(0.0) * detail;
    emitter.carry += rate * seconds;
    let mut count = emitter.carry.floor();
    emitter.carry -= count;
    // A burst asked for always gives at least one particle, however low the detail: a flash
    // or a hit marker must not vanish on small GPUs.
    let asked = desc.burst.eval(params).max(0.0);
    let burst = if asked > 0.0 {
        (asked * detail).round().max(1.0)
    } else {
        0.0
    };
    let every = desc.every.eval(params);
    if !emitter.burst_started {
        emitter.burst_started = true;
        emitter.until_burst = every;
        count += burst;
    } else if every > 0.0 {
        emitter.until_burst -= seconds;
        // However long the frame, at most a few bursts are owed.
        let mut owed = 0;
        while emitter.until_burst <= 0.0 && owed < 4 {
            count += burst;
            emitter.until_burst += every;
            owed += 1;
        }
        emitter.until_burst = emitter.until_burst.max(0.0);
    }
    // The GPU spawns what its free list holds and counts the rest as dropped.
    count as u32
}

/// The record the GPU reads for one emitter this frame.
fn emitter_record(
    desc: &Emitter,
    params: &[f32],
    transform: Mat4,
    emitter: &LiveEmitter,
    spawn_request: u32,
    serial: u32,
    number: usize,
) -> EmitterRecord {
    let v = |value: &Value| value.eval(params);
    let (shape_kind, shape_size, surface) = match desc.shape {
        Shape::Point => (SHAPE_POINT, [0.0; 4], false),
        Shape::Sphere { radius, surface } => {
            (SHAPE_SPHERE, [v(&radius).max(0.0), 0.0, 0.0, 0.0], surface)
        }
        Shape::Box { size } => (
            SHAPE_BOX,
            [
                v(&size[0]).max(0.0),
                v(&size[1]).max(0.0),
                v(&size[2]).max(0.0),
                0.0,
            ],
            false,
        ),
    };
    let [life_low, life_high] = desc.lifetime.map(|l| v(&l).max(1.0e-3));
    let [size_low, size_high] = desc.size.map(|s| v(&s).max(0.0));
    let brightness = v(&desc.brightness).max(0.0);
    let [hot_low, hot_high] = desc.temperature.map(|t| v(&t).max(0.0));
    let mut flags = FLAG_LIVE;
    if desc.blend == Blend::Add {
        flags |= FLAG_ADD;
    }
    if desc.shading == Shading::Lit {
        flags |= FLAG_LIT;
    }
    if hot_high.max(hot_low) > 0.0 {
        flags |= FLAG_GLOW;
    }
    if surface {
        flags |= FLAG_SURFACE;
    }
    EmitterRecord {
        transform: transform.to_cols_array_2d(),
        shape_size,
        offset_cone: [
            v(&desc.offset[0]),
            v(&desc.offset[1]),
            v(&desc.offset[2]),
            v(&desc.cone).clamp(0.0, 180.0).to_radians(),
        ],
        velocity_spread: [
            v(&desc.velocity[0]),
            v(&desc.velocity[1]),
            v(&desc.velocity[2]),
            v(&desc.spread).clamp(0.0, 1.0),
        ],
        acceleration_drag: [0.0, -v(&desc.gravity), 0.0, v(&desc.drag).max(0.0)],
        lifetime_size: [
            life_low,
            life_high.max(life_low),
            size_low,
            size_high.max(size_low),
        ],
        color: [
            v(&desc.color[0]).max(0.0) * brightness,
            v(&desc.color[1]).max(0.0) * brightness,
            v(&desc.color[2]).max(0.0) * brightness,
            (v(&desc.color[3]) * v(&desc.opacity)).clamp(0.0, 1.0),
        ],
        base: emitter.base,
        budget: emitter.budget,
        spawn_request,
        seed: reference::pcg(serial.wrapping_mul(8).wrapping_add(number as u32) ^ 0x2545_f491),
        shape_kind,
        flags,
        soft: v(&desc.soft).max(0.0),
        scale: transform.x_axis.truncate().length(),
        temperature_glow: [
            hot_low,
            hot_high.max(hot_low),
            v(&desc.cooling).max(0.0),
            v(&desc.glow).clamp(0.0, 1.0),
        ],
        phase: [v(&desc.phase).clamp(-0.95, 0.95), 0.0, 0.0, 0.0],
    }
}

/// An emitter's curves as the GPU holds them: its colour's entries, then its size's.
fn baked_curves(desc: &Emitter) -> Vec<[f32; 4]> {
    let mut out: Vec<[f32; 4]> = desc.color_life.bake().to_vec();
    let size = desc.size_life.bake();
    for chunk in size.chunks_exact(4) {
        out.push([chunk[0][0], chunk[1][0], chunk[2][0], chunk[3][0]]);
    }
    debug_assert_eq!(out.len(), CURVE_STRIDE);
    out
}

/// Asynchronous readback of the emitters' counters, as culling statistics are read: copy, map,
/// wait for the callback, read, unmap, never blocking the frame.
struct StatsReadback {
    staging: wgpu::Buffer,
    shared: Arc<StatsShared>,
    /// This frame's live slots, and the alive lists it writes.
    live: [bool; MAX_EMITTERS],
    parity: u32,
}

struct StatsShared {
    /// What the copy in flight was taken from.
    captured: Mutex<([bool; MAX_EMITTERS], u32)>,
    latest: Mutex<ParticleStats>,
    copied: AtomicBool,
    busy: AtomicBool,
    mapped: AtomicBool,
}

impl StatsReadback {
    fn new(gpu: &Gpu) -> Self {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle stats"),
            size: STATE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            staging,
            shared: Arc::new(StatsShared {
                captured: Mutex::new(([false; MAX_EMITTERS], 0)),
                latest: Mutex::default(),
                copied: AtomicBool::new(false),
                busy: AtomicBool::new(false),
                mapped: AtomicBool::new(false),
            }),
            live: [false; MAX_EMITTERS],
            parity: 0,
        }
    }

    fn consume_if_mapped(&self) {
        if self.shared.mapped.swap(false, Ordering::AcqRel) {
            {
                let data = self
                    .staging
                    .slice(..)
                    .get_mapped_range()
                    .expect("the map has completed");
                let states: Vec<EmitterState> = bytemuck::pod_collect_to_vec(&data);
                let (live, parity) = *self.shared.captured.lock().expect("not poisoned");
                let mut stats = ParticleStats::default();
                for (state, _) in states.iter().zip(live).filter(|(_, live)| *live) {
                    stats.alive += state.alive[parity as usize];
                    stats.spawned += u64::from(state.spawned);
                    stats.dropped += u64::from(state.dropped);
                    stats.emitters += 1;
                }
                *self.shared.latest.lock().expect("not poisoned") = stats;
            }
            self.staging.unmap();
            self.shared.busy.store(false, Ordering::Release);
        }
    }

    fn capture(&self, encoder: &mut wgpu::CommandEncoder, state: &wgpu::Buffer) {
        self.consume_if_mapped();
        if !self.shared.busy.load(Ordering::Acquire) {
            encoder.copy_buffer_to_buffer(state, 0, &self.staging, 0, STATE_BYTES);
            *self.shared.captured.lock().expect("not poisoned") = (self.live, self.parity);
            self.shared.copied.store(true, Ordering::Release);
        }
    }

    /// With nothing running there is nothing to read: the counts are zero.
    fn idle(&self) {
        self.consume_if_mapped();
        if !self.shared.busy.load(Ordering::Acquire) {
            *self.shared.latest.lock().expect("not poisoned") = ParticleStats::default();
        }
    }

    fn request_map(&self) {
        if !self.shared.copied.swap(false, Ordering::AcqRel) {
            return;
        }
        self.shared.busy.store(true, Ordering::Release);
        let shared = self.shared.clone();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => shared.mapped.store(true, Ordering::Release),
                Err(_) => shared.busy.store(false, Ordering::Release),
            });
    }

    fn latest(&self) -> ParticleStats {
        *self.shared.latest.lock().expect("not poisoned")
    }
}

#[cfg(test)]
mod tests;
