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
// Trees, from their own trunk's base at y = 0 in the mesh. Their trunk leans with the wind's
// drag, saturating, and sways at its own slow rate, its offset growing with the square of the
// height; their branches bob and swing by their reach from the trunk, each around the crown in
// its own phase, faster; their leaves flutter, fastest. All in metres, for 10 m of height or a
// metre of reach, per (m/s)² or m/s of wind 10 m up.
const TREE_LEAN: f32 = 0.0008;
const TREE_LEAN_SATURATION: f32 = 20.0;
const TREE_SWAY: f32 = 0.012;
const TREE_BRANCH: f32 = 0.004;
const TREE_FLUTTER: f32 = 0.012;
// A tree's sway can be seen further off than grass's: it fades out this much further. Its
// branches' bob and its leaves' flutter are small, and are worked out only nearer, within
// these shares of that.
const TREE_FADE_REACH: f32 = 1.5;
const TREE_BRANCH_REACH: f32 = 0.7;
const TREE_FLUTTER_REACH: f32 = 0.35;
// How far from a tree's axis its trunk reaches, flare included, in metres: branches bob only
// beyond it.
const TRUNK_RADIUS: f32 = 0.5;

// A small hash of an instance's place in the world, so each tree keeps its own rhythm.
fn instance_seed(i: Instance) -> f32 {
    let origin = vec2f(i.world_0.w, i.world_2.w);
    return fract(sin(dot(origin, vec2f(12.9898, 78.233))) * 43758.5453);
}

// Where the wind carries the vertex `local` of a tree, at rest at `world`, in the wind `wind` at
// `time`, seen from `seen_from` metres, as an offset in the world.
fn tree_offset(w: WindParams, i: Instance, local: vec3f, world: vec3f, wind: vec2f, time: f32, seen_from: f32) -> vec3f {
    let speed = length(wind);
    let fade = 1.0 - smoothstep(w.fade.x * TREE_FADE_REACH, w.fade.y * TREE_FADE_REACH, seen_from);
    if speed < 1.0e-3 || fade <= 0.0 {
        return vec3f(0.0);
    }
    let along = wind / speed;
    let across = vec2f(-along.y, along.x);
    let scale = i.max_scale;
    let height = max(local.y, 0.0) * scale;
    let reach = length(local.xz) * scale;
    let seed = instance_seed(i);

    // The trunk: a steady lean, and a slow sway with and across the wind.
    let drag = speed * speed;
    let lean = TREE_LEAN * drag / (1.0 + drag / (TREE_LEAN_SATURATION * TREE_LEAN_SATURATION));
    let rate = 0.3 + 0.25 * seed;
    let sway = TREE_SWAY * speed * sin(TAU * rate * time + seed * TAU);
    let sideways = 0.5 * TREE_SWAY * speed * sin(TAU * rate * 0.77 * time + seed * 17.0);
    let up = sq(height / 10.0);
    var offset = vec3f(along.x, 0.0, along.y) * (lean + sway) * up
        + vec3f(across.x, 0.0, across.y) * sideways * up;
    // Bending, not shearing: the top comes down a little as it leans.
    offset.y -= 0.5 * dot(offset.xz, offset.xz) / max(height, 0.5);

    // Branches, each around the crown in its own phase: which way a branch points from the
    // trunk, and how high it is, set its phase, smoothly, so all of one branch moves together.
    let far = w.fade.y * TREE_FADE_REACH;
    if seen_from < far * TREE_BRANCH_REACH && reach > TRUNK_RADIUS {
        let way = local.xz / max(length(local.xz), 1.0e-5);
        let phase = 2.5 * (3.0 * way.y - 4.0 * way.y * way.y * way.y) * cos(seed * 20.0)
            + 2.5 * way.x * sin(seed * 20.0) + 1.3 * sin(height * 0.8 + way.x * 2.0);
        // Branches bob slowly, about once in one to two seconds: they are long and springy, and
        // only twigs move faster.
        let branch_rate = 0.55 + 0.35 * seed;
        // Only beyond the trunk: its own bark, a short way from its axis, moves with it whole.
        let off_trunk = smoothstep(TRUNK_RADIUS, TRUNK_RADIUS * 3.0, reach);
        // A breeze barely stirs them; the bob grows with the drag, the speed squared, until a
        // fresh wind, then with the speed.
        let stir = speed * min(speed / 10.0, 1.0);
        let out = reach * sqrt(reach) * TREE_BRANCH * stir * off_trunk
            * (1.0 - smoothstep(far * TREE_BRANCH_REACH * 0.7, far * TREE_BRANCH_REACH, seen_from));
        offset += vec3f(along.x, 0.0, along.y) * out * (0.6 + 0.4 * sin(TAU * branch_rate * time + phase));
        offset.y += out * 0.5 * sin(TAU * branch_rate * 1.3 * time + phase * 1.3);
    }

    // Leaves flutter, each its own way: the phase turns over a few centimetres.
    if instance_flutters(i) && seen_from < far * TREE_FLUTTER_REACH {
        let q = world * 7.0;
        // Leaves only flutter hard in a stiff wind: by the drag, from almost nothing in a breeze.
        let strength = TREE_FLUTTER * min(sq(speed / 10.0), 1.5)
            * (1.0 - smoothstep(far * TREE_FLUTTER_REACH * 0.6, far * TREE_FLUTTER_REACH, seen_from));
        let f = TAU * (3.0 + 1.5 * seed) * time;
        offset += strength * vec3f(
            sin(f + q.x * 1.3 + q.y * 0.7),
            sin(f * 1.13 + q.z * 1.1 + q.x * 0.5),
            sin(f * 0.91 + q.y * 1.2 + q.z * 0.8),
        );
    }

    // Never beyond what culling allows for (common/sway.wgsl).
    let most = (TREE_REACH * max(local.y, 0.0) + TREE_REACH_BASE) * scale;
    let moved = length(offset);
    if moved > most {
        offset *= most / moved;
    }
    return offset * fade;
}

// What the wind does to one vertex of a swaying instance: where it carries it now and a frame
// ago, from where it stands at rest, and the bend now, which turns its normal.
struct Sway {
    now: vec3f,
    before: vec3f,
    bend: Bend,
}

// Where the wind carries the vertex `local` in its mesh, `world` at rest, seen from `eye`, in
// the wind `wind` (10 m up) at `time`, with the bend that turns its normal. Grass bends as a
// blade; a tree sways as a tree. The rasterizers and resolve both come here, so they place the
// vertex alike.
fn sway_now(w: WindParams, i: Instance, local: vec3f, world: vec3f, eye: vec3f, wind: vec2f, time: f32) -> Sway {
    let seen_from = distance(eye, world);
    if instance_tree(i) {
        return Sway(tree_offset(w, i, local, world, wind, time, seen_from), vec3f(0.0), Bend(vec2f(1.0, 0.0), 0.0));
    }
    let height = sway_height(i, local);
    let bend = grass_bend(w, wind, time, world, height, seen_from);
    return Sway(bend_offset(bend, height), vec3f(0.0), bend);
}

// Where an instance's wind is read: a tree's at its trunk, so all of it moves together.
fn wind_at(w: WindParams, i: Instance, world: vec3f) -> vec4f {
    if instance_tree(i) {
        return wind_sample(w, vec2f(i.world_0.w, i.world_2.w));
    }
    return wind_sample(w, world.xz);
}

// The sway of the vertex `local` in its mesh and `world` at rest, seen from `eye`, now and a
// frame ago.
fn vertex_sway(w: WindParams, i: Instance, local: vec3f, world: vec3f, eye: vec3f) -> Sway {
    let wind = wind_at(w, i, world);
    var now = sway_now(w, i, local, world, eye, wind.xy, w.time);
    now.before = sway_now(w, i, local, world, eye, wind.zw, w.previous_time).now;
    return now;
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
    return world + sway_now(w, i, local, world, eye, wind_at(w, i, world).xy, w.time).now;
}
