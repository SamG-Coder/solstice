// Collisions, after particles have moved this frame: each live particle of an emitter that
// collides is tested against what it asked for, and a hit bounces it, sticks it, or kills it,
// and may send an event for another emitter to spawn from next frame (docs/particles.md, step 4).
//
// - Depth: this frame's opaque depth and normals. Exact where it is drawn, blind off screen and
//   behind what is drawn.
// - Field: the GI volume's distance field, everywhere within the volume, on screen or not, to
//   within a voxel (0.2 m near the camera).
// - Water: the water's height now, from the water-line map near the camera and the calm level
//   beyond it.
//
// A particle counts as touching once its centre is within its radius of the surface, so it rests
// on a surface rather than sinking into it, and collides only from 0.05 s after its birth, so
// that it can leave what it was born on. Only a particle moving into the surface bounces or
// sends an event; one resting there is held at its surface.

#include "common/environment.wgsl"
#include "common/wetness.wgsl"
#include "particles/common.wgsl"

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<storage, read> emitters: array<Emitter, MAX_EMITTERS>;
@group(0) @binding(2) var<storage, read_write> state: array<EmitterState, MAX_EMITTERS>;
@group(0) @binding(3) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(4) var<storage, read> alive: array<u32>;
@group(0) @binding(5) var<storage, read_write> events: array<Event>;
// The opaque scene's depth (reversed Z, zero for sky) and normals, octahedral in RG.
@group(2) @binding(0) var scene_depth: texture_2d<f32>;
@group(2) @binding(1) var scene_normal: texture_2d<f32>;
@group(3) @binding(0) var<uniform> waterline: WaterlineParams;
@group(3) @binding(1) var waterline_map: texture_2d<f32>;

// Seconds after birth before a particle collides.
const COLLIDE_GRACE: f32 = 0.05;

var<workgroup> group_emitter: u32;

// A surface the particle touches: where, which way it faces, and whether it was found.
struct Contact {
    hit: bool,
    point: vec3f,
    normal: vec3f,
}

fn no_contact() -> Contact {
    return Contact(false, vec3f(0.0), vec3f(0.0, 1.0, 0.0));
}

fn against_water(x: vec3f, radius: f32) -> Contact {
    // With no sea, its level lies far below everything.
    if frame.sea_level < -1.0e20 {
        return no_contact();
    }
    let height = water_heights(x.xz).y;
    if x.y - radius >= height {
        return no_contact();
    }
    return Contact(true, vec3f(x.x, height, x.z), vec3f(0.0, 1.0, 0.0));
}

// The field holds the distance to the nearest surface, which is no help in telling which side
// of a thin surface a point is on. So the particle's path this frame is marched from where it
// started, in steps no longer than that distance, which cannot pass through a surface, and it
// touches where the distance first falls within its reach.
fn against_field(start: vec3f, end: vec3f, radius: f32) -> Contact {
    let path = end - start;
    let length_of_path = length(path);
    let direction = select(vec3f(0.0, -1.0, 0.0), path / length_of_path, length_of_path > 1.0e-6);
    var t = 0.0;
    for (var step = 0u; step < 12u; step++) {
        let x = start + direction * t;
        let level = volume_level_at(x, 0u);
        if level >= VOLUME_LEVELS {
            return no_contact();
        }
        let voxel = volume_params.levels[level].w;
        let distance = sample_volume(x, level).a;
        let reach = max(radius, voxel * 0.5);
        if distance < reach {
            // The field's gradient points away from the nearest surface.
            let d = voxel;
            let gradient = vec3f(
                sample_volume(x + vec3f(d, 0.0, 0.0), level).a - sample_volume(x - vec3f(d, 0.0, 0.0), level).a,
                sample_volume(x + vec3f(0.0, d, 0.0), level).a - sample_volume(x - vec3f(0.0, d, 0.0), level).a,
                sample_volume(x + vec3f(0.0, 0.0, d), level).a - sample_volume(x - vec3f(0.0, 0.0, d), level).a,
            );
            if length(gradient) < 1.0e-6 {
                return no_contact();
            }
            let normal = normalize(gradient);
            return Contact(true, x - normal * distance, normal);
        }
        if t >= length_of_path {
            return no_contact();
        }
        t = min(t + max(distance - reach * 0.5, voxel * 0.25), length_of_path);
    }
    return no_contact();
}

fn against_depth(x: vec3f, radius: f32) -> Contact {
    let clip = frame.unjittered_view_proj * vec4f(x, 1.0);
    if clip.w <= frame.near {
        return no_contact();
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0)) {
        return no_contact();
    }
    let pixel = vec2i(uv * frame.viewport);
    let depth = textureLoad(scene_depth, pixel, 0).r;
    if depth <= 0.0 {
        return no_contact();
    }
    // Distances along the view: the surface drawn there, and the particle.
    let surface = frame.near / depth;
    let along = clip.w;
    // Behind the surface by more than this, it is taken to be hidden behind it rather than in it.
    let thickness = max(0.3, radius * 4.0);
    if along < surface - radius || along > surface + thickness {
        return no_contact();
    }
    let normal = oct_decode(textureLoad(scene_normal, pixel, 0).xy);
    let eye = frame.camera_position;
    let point = eye + (x - eye) * (surface / along);
    return Contact(true, point, normal);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    let written = 1u - sim.parity;
    // The same workgroups as the light pass: the last emitter whose start this group reached.
    if lane == 0u {
        var found = MAX_EMITTERS;
        for (var k = 0u; k < MAX_EMITTERS; k++) {
            let start = state[k].light_start;
            if start > group.x {
                break;
            }
            if group.x < start + (atomicLoad(&state[k].alive[written]) + GROUP_SIZE - 1u) / GROUP_SIZE {
                found = k;
            }
        }
        group_emitter = found;
    }
    let e = workgroupUniformLoad(&group_emitter);
    if e >= MAX_EMITTERS {
        return;
    }
    let emitter = emitters[e];
    if (emitter.flags & FLAG_LIVE) == 0u || (emitter.flags & FLAG_COLLIDE) == 0u {
        return;
    }
    let i = (group.x - state[e].light_start) * GROUP_SIZE + lane;
    if i >= atomicLoad(&state[e].alive[written]) {
        return;
    }
    let slot = alive[written * sim.capacity + emitter.base + i];
    var p = particles[slot];
    // A particle just born may be touching what it was born on, such as a fountain's drops the
    // ground, where which side it is on is unclear: it has a moment to leave first.
    if is_dead(p) || (p.flags & PARTICLE_STUCK) != 0u || p.age < COLLIDE_GRACE {
        return;
    }
    let radius = p.size * 0.5;
    var contact = no_contact();
    if (emitter.collide_against & COLLIDE_WATER) != 0u {
        contact = against_water(p.position, radius);
    }
    if !contact.hit && (emitter.collide_against & COLLIDE_DEPTH) != 0u {
        contact = against_depth(p.position, radius);
    }
    if !contact.hit && (emitter.collide_against & COLLIDE_FIELD) != 0u {
        contact = against_field(p.previous, p.position, radius);
    }
    if !contact.hit {
        return;
    }
    let n = contact.normal;
    let into = dot(p.velocity, n);
    let resting = contact.point + n * radius;
    if into >= 0.0 {
        // Already leaving, or resting: kept out of the surface, nothing more.
        if dot(p.position - contact.point, n) < radius {
            p.position = resting;
        }
        particles[slot] = p;
        return;
    }
    if emitter.response == RESPONSE_KILL {
        p.age = p.lifetime;
    } else if emitter.response == RESPONSE_STICK {
        p.position = resting;
        p.velocity = vec3f(0.0);
        p.flags |= PARTICLE_STUCK;
    } else {
        let normal_part = into * n;
        let along = p.velocity - normal_part;
        let rf = emitter.restitution_friction;
        p.velocity = along * (1.0 - rf.y) - normal_part * rf.x;
        p.position = resting;
    }
    particles[slot] = p;
    let to = emitter.event_target;
    if to != NO_TARGET && to < MAX_EMITTERS {
        let k = atomicAdd(&state[to].event_count, 1u);
        if k < EVENT_CAPACITY {
            events[to * EVENT_CAPACITY + k] = Event(contact.point, 0.0, n, 0.0);
        }
    }
}
