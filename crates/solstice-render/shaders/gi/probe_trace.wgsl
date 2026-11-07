// Traces all probe rays. Each ray tries the screen first, where geometry is exact and last
// frame's lighting is sharp, and on leaving it continues through the GI volume. Screen hits
// recover the contact detail that the volume's voxels are too coarse to represent.

#include "common/environment.wgsl"
#include "gi/probe_common.wgsl"

@group(0) @binding(0) var normal_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

const SCREEN_STEPS: u32 = 10u;
const SCREEN_DISTANCE: f32 = 2.5;

struct ScreenHit {
    hit: bool,
    t: f32,
    radiance: vec3f,
}

fn project(world: vec3f) -> vec3f {
    let clip = frame.unjittered_view_proj * vec4f(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

fn trace_screen(origin: vec3f, direction: vec3f, jitter: f32) -> ScreenHit {
    let size = vec2i(frame.viewport);
    var previous_t = 0.0;
    for (var i = 0u; i < SCREEN_STEPS; i++) {
        let f = (f32(i) + jitter) / f32(SCREEN_STEPS);
        let t = f * f * SCREEN_DISTANCE + 0.02;
        let p = project(origin + direction * t);
        if any(p.xy < vec2f(0.0)) || any(p.xy >= vec2f(1.0)) || p.z <= 0.0 {
            break;
        }
        let pixel = vec2i(p.xy * frame.viewport);
        let scene_depth = textureLoad(depth_tex, min(pixel, size - vec2i(1)), 0).r;
        if scene_depth > p.z {
            // Behind something. Only count it if the surface is plausibly this thin:
            // compare linear distances, since reversed Z is not linear.
            let scene_distance = frame.near / scene_depth;
            let ray_distance = frame.near / p.z;
            let thickness = max(t - previous_t, 0.05) + 0.15;
            if ray_distance - scene_distance < thickness {
                // History may be at output resolution, which can differ from the viewport.
                let history_pixel = vec2i(p.xy * vec2f(textureDimensions(history_tex)));
                let radiance = textureLoad(history_tex, history_pixel, 0).rgb;
                return ScreenHit(true, t, radiance);
            }
        }
        previous_t = t;
    }
    return ScreenHit(false, 0.0, vec3f(0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if any(id.xy >= dims) {
        return;
    }
    let tile = id.xy / PROBE_RAYS;
    let ray = id.xy % PROBE_RAYS;

    let pixel = probe_pixel(tile, frame.viewport, frame.frame_index);
    let depth = textureLoad(depth_tex, vec2i(pixel), 0).r;
    if depth <= 0.0 {
        textureStore(output, id.xy, vec4f(0.0, 0.0, 0.0, -1.0));
        return;
    }
    let packed_normal = textureLoad(normal_tex, vec2i(pixel), 0);
    let normal = oct_decode(packed_normal.xy);
    let world = reconstruct_world(frame, vec2f(pixel) + vec2f(0.5), depth);
    let direction = probe_ray_direction(tile, ray, normal, frame.frame_index);

    let jitter = hash_to_unit(id.x * 3u + id.y * 1009u + frame.frame_index * 71u);
    let screen = trace_screen(world + normal * 0.03, direction, jitter);
    if screen.hit {
        textureStore(output, id.xy, vec4f(screen.radiance, screen.t));
        return;
    }

    // The volume is coarse near its surfaces, so start clear of the local voxel.
    let level = min(volume_level_at(world, 0u), VOLUME_LEVELS - 1u);
    let voxel = volume_params.levels[level].w;
    let origin = world + normal * voxel * 1.5;
    let hit = trace_volume(origin, direction, voxel * 0.5, 320.0, 40u);
    if hit.hit {
        textureStore(output, id.xy, vec4f(hit.radiance, hit.t));
    } else {
        textureStore(output, id.xy, vec4f(sky_radiance(direction), 1000.0));
    }
}
