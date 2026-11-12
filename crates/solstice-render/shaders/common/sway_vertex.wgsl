// Bends what sways in the wind, where its vertices are placed: in both rasterizers and in
// resolve, which must agree to the bit. A shader that includes this declares `wind_field`, the
// texture wind/field.wgsl fills. See docs/weather-effects.md.
//
// Grass bends as a blade does: along an arc, so it keeps its length, by an angle growing with
// the drag of the wind on it, the square of the wind's speed, until it lies over. It shimmers
// around that lean at its own rate, and swings a little across the wind. Neighbouring blades
// read the same wind, so a gust is seen running over a meadow as a wave.

#include "common/math.wgsl"
#include "common/sway.wgsl"

// Twin of `FIELD_TEXELS` in wind.rs.
const WIND_FIELD_TEXELS: u32 = 128u;
// The bend per metre of height in a wind of `s` m/s 10 m up:
// GRASS_BEND s² / (1 + (s / GRASS_SATURATION)²). A breeze of 5 m/s turns a half-metre blade's tip
// about a quarter of a radian; a gale lays it over.
const GRASS_BEND: f32 = 0.022;
const GRASS_SATURATION: f32 = 15.0;
// The most a point on a blade turns, in radians. GRASS_REACH bounds the arc it makes.
const GRASS_MAX_ANGLE: f32 = 1.35;
// The share of the bend that comes and goes as the blade shimmers, how often it does, and how
// far across the wind it swings, as a share of the wind's direction.
const GRASS_SHIMMER: f32 = 0.22;
const GRASS_SHIMMER_HZ: f32 = 1.3;
const GRASS_SWING: f32 = 0.12;

// What a vertex needs besides the field's texels: written by wind/field.wgsl below the field.
struct WindParams {
    origin: vec2f,
    texel: f32,
    sway: f32,
    time: f32,
    previous_time: f32,
    fade: vec2f,
}

fn wind_params() -> WindParams {
    let a = textureLoad(wind_field, vec2u(0u, WIND_FIELD_TEXELS), 0);
    let b = textureLoad(wind_field, vec2u(1u, WIND_FIELD_TEXELS), 0);
    return WindParams(a.xy, a.z, a.w, b.x, b.y, b.zw);
}

// The wind 10 m up at (x, z): this frame's in xy and last frame's in zw, between the texels
// around it, and held at the field's edge.
fn wind_sample(w: WindParams, xz: vec2f) -> vec4f {
    let last = f32(WIND_FIELD_TEXELS - 1u);
    let t = clamp((xz - w.origin) / w.texel, vec2f(0.0), vec2f(last));
    let i = min(vec2u(t), vec2u(WIND_FIELD_TEXELS - 2u));
    let f = t - vec2f(i);
    let a = textureLoad(wind_field, i, 0);
    let b = textureLoad(wind_field, i + vec2u(1u, 0u), 0);
    let c = textureLoad(wind_field, i + vec2u(0u, 1u), 0);
    let d = textureLoad(wind_field, i + vec2u(1u, 1u), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// A blade's bend at one point of it: the way it leans, across the ground, and the angle the
// blade has turned through by there.
struct Bend {
    toward: vec2f,
    angle: f32,
}

// How far up its blade a point of a swaying instance is, in metres: its height in the mesh,
// whose ground is y = 0, scaled as the instance is.
fn sway_height(i: Instance, local: vec3f) -> f32 {
    return max(local.y, 0.0) * length(vec3f(i.world_0.y, i.world_1.y, i.world_2.y));
}

// The bend of the point `height` metres up a blade at `at`, at rest, in the wind `wind` at
// `time`, seen from `distance` metres: swaying fades out with distance, where it can't be seen.
fn grass_bend(w: WindParams, wind: vec2f, time: f32, at: vec3f, height: f32, distance: f32) -> Bend {
    let speed = length(wind);
    if speed < 1.0e-3 || height <= 0.0 {
        return Bend(vec2f(1.0, 0.0), 0.0);
    }
    let along = wind / speed;
    let across = vec2f(-along.y, along.x);
    // Neighbouring blades swing together, but not in step: the phase turns over tens of
    // centimetres, so all of one blade's points share it.
    let phase = 2.3 * sin(at.x * 1.7 + 1.9 * sin(at.z * 1.3)) + 2.1 * sin(at.z * 2.1 + 1.4 * sin(at.x * 0.9));
    let swing = TAU * GRASS_SHIMMER_HZ * time + phase;
    let saturation = speed / GRASS_SATURATION;
    let bend = GRASS_BEND * speed * speed / (1.0 + saturation * saturation);
    let curvature = bend * (1.0 + GRASS_SHIMMER * sin(swing));
    let toward = normalize(along + across * (GRASS_SWING * sin(swing * 0.73 + 1.7)));
    let fade = 1.0 - smoothstep(w.fade.x, w.fade.y, distance);
    return Bend(toward, min(curvature * height, GRASS_MAX_ANGLE) * fade * w.sway);
}

// Where a bend takes the point `height` up its blade, as an offset in the world. The blade is
// an arc of the same length, so it leans out and comes down, and never stretches.
fn bend_offset(b: Bend, height: f32) -> vec3f {
    let a = b.angle;
    var outward: f32;
    var up: f32;
    if a < 0.02 {
        // The series, where the closed forms would divide by almost nothing.
        outward = a * 0.5 - a * a * a / 24.0;
        up = 1.0 - a * a / 6.0;
    } else {
        outward = (1.0 - cos(a)) / a;
        up = sin(a) / a;
    }
    return vec3f(b.toward.x * outward * height, (up - 1.0) * height, b.toward.y * outward * height);
}

// A direction turned as the blade has turned at that point: up tips towards the way it leans.
fn bend_direction(b: Bend, n: vec3f) -> vec3f {
    let axis = vec3f(b.toward.y, 0.0, -b.toward.x);
    let c = cos(b.angle);
    let s = sin(b.angle);
    return n * c + cross(axis, n) * s + axis * (dot(axis, n) * (1.0 - c));
}

// What the wind does to one vertex of a swaying instance: the bend now and a frame ago.
struct Sway {
    now: Bend,
    before: Bend,
    height: f32,
}

// The bend now of the vertex `local` in its mesh and `world` at rest, seen from `eye`, with
// the wind there in `wind`. The rasterizers need only this; resolve also needs last frame's,
// and both come here for this one, so they place the vertex alike.
fn vertex_bend(w: WindParams, i: Instance, local: vec3f, world: vec3f, eye: vec3f, wind: vec4f) -> Bend {
    return grass_bend(w, wind.xy, w.time, world, sway_height(i, local), distance(eye, world));
}

// The sway of the vertex `local` in its mesh and `world` at rest, seen from `eye`.
fn vertex_sway(w: WindParams, i: Instance, local: vec3f, world: vec3f, eye: vec3f) -> Sway {
    let wind = wind_sample(w, world.xz);
    let height = sway_height(i, local);
    return Sway(
        vertex_bend(w, i, local, world, eye, wind),
        grass_bend(w, wind.zw, w.previous_time, world, height, distance(eye, world)),
        height,
    );
}

// Where a vertex goes now, from where it stands at rest, `world`, seen from `eye`. Instances
// that don't sway stay where they are.
fn swayed_position(i: Instance, local: vec3f, world: vec3f, eye: vec3f) -> vec3f {
    if !instance_sways(i) {
        return world;
    }
    let w = wind_params();
    if w.sway == 0.0 {
        return world;
    }
    let bend = vertex_bend(w, i, local, world, eye, wind_sample(w, world.xz));
    return world + bend_offset(bend, sway_height(i, local));
}
