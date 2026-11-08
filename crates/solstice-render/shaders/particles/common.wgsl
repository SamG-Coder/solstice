// The particle system's records, and how particles are born and move (docs/particles.md).
// Twins of the structs and functions in particles/reference.rs, which tests compare against.

#include "common/math.wgsl"

const FLAG_LIVE: u32 = 1u;
const FLAG_ADD: u32 = 2u;
const FLAG_SURFACE: u32 = 4u;
const FLAG_LIT: u32 = 8u;
const FLAG_GLOW: u32 = 16u;
const FLAG_SOLID: u32 = 32u;
const FLAG_SORT: u32 = 64u;
const AIR_TEMPERATURE: f32 = 300.0;

const SHAPE_SPHERE: u32 = 1u;
const SHAPE_BOX: u32 = 2u;

const MAX_EMITTERS: u32 = 64u;
const GROUP_SIZE: u32 = 64u;
const SERIES_BELOW: f32 = 1.0e-2;

// One emitter this frame, its values worked out from its effect's parameters.
struct Emitter {
    transform: mat4x4f,
    shape_size: vec4f,
    // xyz where the shape is centred; w the cone's half-angle in radians.
    offset_cone: vec4f,
    // xyz initial velocity in the effect's space; w the spread of speed.
    velocity_spread: vec4f,
    // xyz acceleration in the world; w linear drag per second.
    acceleration_drag: vec4f,
    // Lifetime's range, then size's.
    lifetime_size: vec4f,
    color: vec4f,
    base: u32,
    budget: u32,
    spawn_request: u32,
    seed: u32,
    shape_kind: u32,
    flags: u32,
    soft: f32,
    scale: f32,
    // Temperature's range at birth in kelvin, then cooling per second and emissivity.
    temperature_glow: vec4f,
    // Henyey-Greenstein asymmetry in x.
    phase: vec4f,
}

struct Particle {
    position: vec3f,
    age: f32,
    velocity: vec3f,
    lifetime: f32,
    previous: vec3f,
    size: f32,
    index: u32,
    rotation: f32,
    // Temperature at birth, in kelvin; 0 for particles that do not glow.
    temperature: f32,
    _pad: u32,
}

// The GPU's running state for one emitter.
struct EmitterState {
    free_count: atomic<u32>,
    alive: array<atomic<u32>, 2>,
    spawned: u32,
    dropped: u32,
    spawn_count: u32,
    spawn_first: u32,
    spawn_index: u32,
    update_count: u32,
    spawn_start: u32,
    update_start: u32,
    light_start: u32,
    _pad: array<u32, 4>,
}

struct SimParams {
    // Seconds since the last simulated frame.
    seconds: f32,
    // The alive lists read this frame; the other is written.
    parity: u32,
    // Slots in the pool, and so in each alive list.
    capacity: u32,
    // Each particle's light is worked out afresh once in this many frames, staggered.
    light_interval: u32,
    // Steps each ambient ray takes through the GI volume.
    ray_steps: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // Where the camera is and which way it looks, for sorting by distance along the view.
    camera_position: vec4f,
    camera_forward: vec4f,
}

fn particle_random(seed: u32, index: u32, stream: u32) -> f32 {
    return hash_to_unit(seed ^ pcg(index * 16u + stream));
}

fn spawn_elapsed(i: u32, n: u32, seconds: f32) -> f32 {
    return seconds * (1.0 - (f32(i) + 0.5) / f32(max(n, 1u)));
}

fn spawn_particle(e: Emitter, index: u32) -> Particle {
    var local = vec3f(0.0);
    if e.shape_kind == SHAPE_SPHERE {
        let z = 2.0 * particle_random(e.seed, index, 0u) - 1.0;
        let phi = TAU * particle_random(e.seed, index, 1u);
        let ring = sqrt(max(1.0 - z * z, 0.0));
        var radius = e.shape_size.x;
        if (e.flags & FLAG_SURFACE) == 0u {
            radius *= pow(particle_random(e.seed, index, 2u), 1.0 / 3.0);
        }
        local = vec3f(ring * cos(phi), ring * sin(phi), z) * radius;
    } else if e.shape_kind == SHAPE_BOX {
        let u = vec3f(
            particle_random(e.seed, index, 0u),
            particle_random(e.seed, index, 1u),
            particle_random(e.seed, index, 2u),
        );
        local = (u - vec3f(0.5)) * e.shape_size.xyz;
    }
    local += e.offset_cone.xyz;
    let position = (e.transform * vec4f(local, 1.0)).xyz;

    let base = e.velocity_spread.xyz;
    let speed = length(base) * (1.0 + e.velocity_spread.w * (2.0 * particle_random(e.seed, index, 5u) - 1.0));
    var axis = vec3f(0.0, 1.0, 0.0);
    if speed > 0.0 {
        axis = normalize(base);
    }
    let cos_theta = 1.0 - particle_random(e.seed, index, 3u) * (1.0 - cos(e.offset_cone.w));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = TAU * particle_random(e.seed, index, 4u);
    let direction = orthonormal_basis(axis) * vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    let velocity = (e.transform * vec4f(direction * speed, 0.0)).xyz;

    let ls = e.lifetime_size;
    var p: Particle;
    p.position = position;
    p.age = 0.0;
    p.velocity = velocity;
    p.lifetime = ls.x + (ls.y - ls.x) * particle_random(e.seed, index, 7u);
    p.previous = position;
    p.size = (ls.z + (ls.w - ls.z) * particle_random(e.seed, index, 6u)) * e.scale;
    p.index = index;
    p.rotation = TAU * particle_random(e.seed, index, 8u);
    let hot = e.temperature_glow;
    p.temperature = hot.x + (hot.y - hot.x) * particle_random(e.seed, index, 9u);
    return p;
}

fn particle_temperature(p: Particle, e: Emitter) -> f32 {
    if p.temperature <= 0.0 {
        return 0.0;
    }
    return AIR_TEMPERATURE + (p.temperature - AIR_TEMPERATURE) * exp(-e.temperature_glow.z * p.age);
}

// Gravity and linear drag, integrated exactly: with d = exp(-k h), velocity becomes v d + a F
// and position moves by v F + a G, where F = (1 - d) / k and G = (h - F) / k.
fn step_particle(p: ptr<function, Particle>, e: Emitter, seconds: f32) {
    let a = e.acceleration_drag.xyz;
    let k = e.acceleration_drag.w;
    let h = seconds;
    let kh = k * h;
    var decay: f32;
    var f: f32;
    var g: f32;
    if kh < SERIES_BELOW {
        decay = 1.0 - kh + 0.5 * kh * kh;
        f = h * (1.0 - 0.5 * kh + kh * kh / 6.0);
        g = h * h * (0.5 - kh / 6.0 + kh * kh / 24.0);
    } else {
        decay = exp(-kh);
        f = (1.0 - decay) / k;
        g = (h - f) / k;
    }
    let v = (*p).velocity;
    let x = (*p).position;
    (*p).previous = x;
    (*p).position = x + v * f + a * g;
    (*p).velocity = v * decay + a * f;
    (*p).age += h;
}

fn is_dead(p: Particle) -> bool {
    return p.age >= p.lifetime;
}
