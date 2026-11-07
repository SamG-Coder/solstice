// Bind group 1 of every lighting pass: frame constants, sky LUTs and the GI volume.
// Includes the functions that sample them.

#include "common/math.wgsl"
#include "common/frame.wgsl"
#include "common/atmosphere.wgsl"

const VOLUME_LEVELS: u32 = 5u;

struct VolumeParams {
    // Per clipmap level: world position of the grid's corner, and the voxel size.
    levels: array<vec4f, 5>,
    resolution: f32,
    inv_resolution: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(1) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(1) var<uniform> volume_params: VolumeParams;
@group(1) @binding(2) var volume_atlas: texture_3d<f32>;
@group(1) @binding(3) var clamp_sampler: sampler;
@group(1) @binding(4) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(5) var sky_view_lut: texture_2d<f32>;
@group(1) @binding(6) var repeat_sampler: sampler;

// Height of the viewer above the planet centre, in kilometres.
fn viewer_radius() -> f32 {
    return GROUND_RADIUS + max(frame.camera_position.y, 1.0) * 0.001;
}

// Radiance of the sky towards `direction`, excluding the sun disc.
fn sky_radiance(direction: vec3f) -> vec3f {
    let color = textureSampleLevel(sky_view_lut, repeat_sampler, sky_view_uv(direction), 0.0).rgb;
    return color * frame.sun_illuminance * frame.sky_intensity;
}

// Sunlight reaching the ground, after the atmosphere has filtered it.
fn sun_irradiance() -> vec3f {
    let uv = transmittance_uv(viewer_radius(), frame.sun_direction.y);
    let transmittance = textureSampleLevel(transmittance_lut, clamp_sampler, uv, 0.0).rgb;
    return transmittance * frame.sun_color * frame.sun_illuminance;
}

// The finest clipmap level containing `p` with some margin, or VOLUME_LEVELS if none.
fn volume_level_at(p: vec3f, first_level: u32) -> u32 {
    for (var level = first_level; level < VOLUME_LEVELS; level++) {
        let grid = volume_params.levels[level];
        let local = (p - grid.xyz) / grid.w;
        let margin = 1.5;
        let upper = volume_params.resolution - margin;
        if all(local > vec3f(margin)) && all(local < vec3f(upper)) {
            return level;
        }
    }
    return VOLUME_LEVELS;
}

// Radiance in RGB, distance to the nearest surface in A (world units).
fn sample_volume(p: vec3f, level: u32) -> vec4f {
    let grid = volume_params.levels[level];
    let local = (p - grid.xyz) / grid.w * volume_params.inv_resolution;
    // Each level occupies one slab of the atlas. Stay half a texel inside, so that
    // filtering cannot read a neighbouring level.
    let half_texel = 0.5 * volume_params.inv_resolution;
    let z = clamp(local.z, half_texel, 1.0 - half_texel);
    let atlas_z = (z + f32(level)) / f32(VOLUME_LEVELS);
    return textureSampleLevel(volume_atlas, clamp_sampler, vec3f(local.xy, atlas_z), 0.0);
}

struct VolumeHit {
    hit: bool,
    t: f32,
    radiance: vec3f,
}

// Sphere traces the GI volume. `start` skips self-intersection near the ray's origin.
fn trace_volume(origin: vec3f, direction: vec3f, start: f32, max_t: f32, max_steps: u32) -> VolumeHit {
    return trace_volume_with(origin, direction, start, max_t, max_steps, 0.85);
}

// `hit_voxels` is how close counts as a hit, in voxels. Lighting uses a generous value, so that
// thin walls stop light. Reflections can use less, so that thin objects look thin.
fn trace_volume_with(origin: vec3f, direction: vec3f, start: f32, max_t: f32, max_steps: u32, hit_voxels: f32) -> VolumeHit {
    var t = start;
    var level = 0u;
    var measured_level = VOLUME_LEVELS;
    var previous_distance = 0.0;
    for (var step = 0u; step < max_steps && t < max_t; step++) {
        let p = origin + direction * t;
        level = volume_level_at(p, level);
        if level >= VOLUME_LEVELS {
            break;
        }
        let voxel = volume_params.levels[level].w;
        let s = sample_volume(p, level);
        // A hit needs the ray to be closing in on a surface, measured within a single level:
        // distances from different levels are not comparable. Without this, a ray skimming away
        // from flat ground falsely hits it on entering a coarser level, where the threshold
        // doubles, which draws the boundaries of the levels onto glossy surfaces as lines.
        let approaching = level == measured_level && s.a <= previous_distance;
        measured_level = level;
        previous_distance = s.a;
        if s.a < voxel * hit_voxels && approaching {
            return VolumeHit(true, t, s.rgb);
        }
        t += max(s.a - voxel * 0.5, voxel * 0.6);
    }
    return VolumeHit(false, t, vec3f(0.0));
}

// How far inside the outermost level `p` is: 0 at its edge, reaching 1 well inside. Rays from
// beyond the edge see only sky, so anything that blends by this avoids a visible boundary there.
fn volume_edge_fade(p: vec3f) -> f32 {
    let grid = volume_params.levels[VOLUME_LEVELS - 1u];
    let local = (p - grid.xyz) / grid.w * volume_params.inv_resolution;
    let edge = min(min(local.x, 1.0 - local.x), min(local.z, 1.0 - local.z));
    return smoothstep(0.02, 0.2, edge);
}

// True if nothing in the volume blocks the path towards `direction`.
fn volume_visibility(origin: vec3f, direction: vec3f, start: f32, max_t: f32, max_steps: u32) -> f32 {
    return select(1.0, 0.0, trace_volume(origin, direction, start, max_t, max_steps).hit);
}
