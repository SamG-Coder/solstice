// Traces one glossy reflection ray per half-resolution pixel: first in screen space for
// sharp detail, then through the GI volume for what the screen cannot see, then the sky.
// Very rough surfaces skip tracing, since diffuse GI already approximates them well.

#include "common/environment.wgsl"

@group(0) @binding(0) var normal_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var motion_tex: texture_2d<f32>;
@group(0) @binding(3) var scene_history: texture_2d<f32>;
@group(0) @binding(4) var reflection_history: texture_2d<f32>;
@group(0) @binding(5) var output: texture_storage_2d<rgba16float, write>;

const MAX_TRACED_ROUGHNESS: f32 = 0.65;
const SCREEN_STEPS: u32 = 24u;
const SCREEN_DISTANCE: f32 = 40.0;
const SCREEN_EDGE_FADE: f32 = 0.08;
const REFINE_ATTEMPTS: u32 = 6u;

fn project(world: vec3f) -> vec3f {
    let clip = frame.unjittered_view_proj * vec4f(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

// Samples the GGX distribution of visible normals (Heitz 2018).
fn sample_vndf(view_local: vec3f, a: f32, u: vec2f) -> vec3f {
    let vh = normalize(vec3f(a * view_local.x, a * view_local.y, view_local.z));
    let len2 = vh.x * vh.x + vh.y * vh.y;
    let t1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-vh.y, vh.x, 0.0) / sqrt(len2), len2 > 1.0e-7);
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = TAU * u.y;
    let p1 = r * cos(phi);
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * r * sin(phi);
    let nh = t1 * p1 + t2 * p2 + vh * sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
    return normalize(vec3f(a * nh.x, a * nh.y, max(nh.z, 0.0)));
}

struct Trace {
    // How much of the result to use: 0 for a miss, and falling off towards the screen edges,
    // where this data is about to run out.
    weight: f32,
    radiance: vec3f,
}

fn trace_screen(origin: vec3f, direction: vec3f, jitter: f32) -> Trace {
    let dims = vec2i(frame.viewport);
    var previous_t = 0.0;
    for (var i = 0u; i < SCREEN_STEPS; i++) {
        let f = (f32(i) + jitter) / f32(SCREEN_STEPS);
        let t = f * f * SCREEN_DISTANCE + 0.03;
        let p = project(origin + direction * t);
        if any(p.xy < vec2f(0.0)) || any(p.xy >= vec2f(1.0)) || p.z <= 0.0 {
            break;
        }
        let pixel = min(vec2i(p.xy * frame.viewport), dims - vec2i(1));
        let scene_depth = textureLoad(depth_tex, pixel, 0).r;
        if scene_depth > p.z {
            let scene_distance = frame.near / scene_depth;
            let ray_distance = frame.near / p.z;
            if ray_distance - scene_distance < max(t - previous_t, 0.05) + 0.2 {
                // A ray can only strike the front of a surface. Without this, a ray leaving a
                // surface seen at a grazing angle "hits" that same surface: one pixel there spans
                // metres of depth, so the neighbouring pixel can read as nearer than the ray.
                let surface = oct_decode(textureLoad(normal_tex, pixel, 0).xy);
                if dot(surface, direction) < 0.0 {
                    let edge = min(min(p.x, 1.0 - p.x), min(p.y, 1.0 - p.y));
                    let history_pixel = vec2i(p.xy * vec2f(textureDimensions(scene_history)));
                    return Trace(smoothstep(0.0, SCREEN_EDGE_FADE, edge), textureLoad(scene_history, history_pixel, 0).rgb);
                }
            }
        }
        previous_t = t;
    }
    return Trace(0.0, vec3f(0.0));
}

// Traces the volume, then checks each hit against what the camera can see. Voxels are coarse, so
// thin geometry such as rigging swells into solid blobs. Where the camera sees past a hit point,
// that point is really empty, and the ray carries on. Where the camera sees a surface there, its
// colour replaces the voxel's, since it is sharper and fully shaded.
fn trace_refined(origin: vec3f, direction: vec3f, start: f32) -> VolumeHit {
    var from_t = start;
    for (var attempt = 0u; attempt < REFINE_ATTEMPTS; attempt++) {
        let hit = trace_volume_with(origin, direction, from_t, 320.0, 48u, 0.55);
        if !hit.hit {
            return hit;
        }
        let point = origin + direction * hit.t;
        let p = project(point);
        if any(p.xy < vec2f(0.0)) || any(p.xy >= vec2f(1.0)) || p.z <= 0.0 {
            return hit;
        }
        let level = min(volume_level_at(point, 0u), VOLUME_LEVELS - 1u);
        let margin = volume_params.levels[level].w * 1.5;
        let pixel = min(vec2i(p.xy * frame.viewport), vec2i(frame.viewport) - vec2i(1));
        let scene_depth = textureLoad(depth_tex, pixel, 0).r;
        let ray_distance = frame.near / p.z;
        if scene_depth <= 0.0 || frame.near / scene_depth > ray_distance + margin {
            from_t = hit.t + margin * 0.33;
            continue;
        }
        if frame.near / scene_depth < ray_distance - margin {
            return hit; // Something nearer hides this point, so the voxel is all there is.
        }
        let surface = oct_decode(textureLoad(normal_tex, pixel, 0).xy);
        if dot(surface, direction) >= 0.0 {
            return hit;
        }
        let edge = min(min(p.x, 1.0 - p.x), min(p.y, 1.0 - p.y));
        let history_pixel = vec2i(p.xy * vec2f(textureDimensions(scene_history)));
        let seen = textureLoad(scene_history, history_pixel, 0).rgb;
        return VolumeHit(true, hit.t, mix(hit.radiance, seen, smoothstep(0.0, SCREEN_EDGE_FADE, edge)));
    }
    return VolumeHit(false, from_t, vec3f(0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if any(id.xy >= dims) {
        return;
    }
    // Each output pixel covers a 2x2 block. Sample a different corner per frame.
    let corner = vec2u(frame.frame_index & 1u, (frame.frame_index >> 1u) & 1u);
    let pixel = min(id.xy * 2u + corner, vec2u(frame.viewport) - vec2u(1u));
    let depth = textureLoad(depth_tex, vec2i(pixel), 0).r;
    if depth <= 0.0 {
        textureStore(output, id.xy, vec4f(0.0));
        return;
    }
    let packed = textureLoad(normal_tex, vec2i(pixel), 0);
    let roughness = max(packed.z, 0.045);
    if roughness > MAX_TRACED_ROUGHNESS {
        textureStore(output, id.xy, vec4f(0.0));
        return;
    }
    let normal = oct_decode(packed.xy);
    let center = vec2f(pixel) + vec2f(0.5);
    let world = reconstruct_world(frame, center, depth);
    let view = normalize(frame.camera_position - world);

    // Build the reflection direction in the normal's tangent frame.
    let basis = orthonormal_basis(normal);
    let view_local = transpose(basis) * view;
    let noise = vec2f(
        interleaved_gradient_noise(vec2f(id.xy), frame.frame_index),
        interleaved_gradient_noise(vec2f(id.yx) + vec2f(47.0, 17.0), frame.frame_index),
    );
    let half_vector = basis * sample_vndf(view_local, roughness * roughness, noise);
    var direction = reflect(-view, half_vector);
    if dot(direction, normal) <= 0.0 {
        direction = reflect(-view, normal);
    }

    let screen = trace_screen(world + normal * 0.02, direction, noise.x);
    var radiance = screen.radiance;
    if screen.weight < 1.0 {
        let level = min(volume_level_at(world, 0u), VOLUME_LEVELS - 1u);
        let voxel = volume_params.levels[level].w;
        let hit = trace_refined(world + normal * voxel * 1.5, direction, voxel * 0.5);
        let sky = sky_radiance(direction);
        // Fade by where the ray ended up. A ray leaving the volume before reaching something sees
        // only sky, so hits near the edge must already be blending towards that, or the edge shows
        // as a hard line across glossy surfaces.
        let hit_point = world + direction * hit.t;
        let far = mix(sky, select(sky, hit.radiance, hit.hit), volume_edge_fade(hit_point) * volume_edge_fade(world));
        radiance = mix(far, screen.radiance, screen.weight);
    }

    // Fireflies would otherwise persist through the temporal filter.
    let peak = max(radiance.x, max(radiance.y, radiance.z));
    let limit = frame.sun_illuminance * 4.0;
    if peak > limit {
        radiance *= limit / peak;
    }

    var result = vec4f(radiance, 1.0);
    let motion = textureLoad(motion_tex, vec2i(pixel), 0).xy;
    let previous_uv = center * frame.inv_viewport + motion;
    if all(previous_uv > vec2f(0.0)) && all(previous_uv < vec2f(1.0)) && frame.frame_index > 0u {
        let history = textureSampleLevel(reflection_history, clamp_sampler, previous_uv, 0.0);
        // Rough reflections are noisier and depend less on the view, so they keep more history.
        let blend = mix(0.25, 0.07, clamp(roughness / MAX_TRACED_ROUGHNESS, 0.0, 1.0));
        let moved = length(motion * frame.viewport);
        result = mix(history, result, clamp(blend + moved * 0.04, blend, 0.6));
    }
    textureStore(output, id.xy, result);
}
