// Traces one glossy reflection ray per half-resolution pixel: first through the screen's depth
// pyramid for sharp detail, then through the GI volume for what the screen cannot see, then the sky.
// Very rough surfaces skip tracing, since diffuse GI already approximates them well.

#include "common/environment.wgsl"
#include "gi/screen_trace.wgsl"

@group(0) @binding(0) var normal_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var motion_tex: texture_2d<f32>;
@group(0) @binding(3) var scene_history: texture_2d<f32>;
@group(0) @binding(4) var reflection_history: texture_2d<f32>;
@group(0) @binding(5) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var depth_pyramid: texture_2d<f32>;

const MAX_TRACED_ROUGHNESS: f32 = 0.65;
const SCREEN_DISTANCE: f32 = 200.0;
const SCREEN_STEPS: u32 = 64u;
const SCREEN_THICKNESS: f32 = 0.25;
const REFINE_ATTEMPTS: u32 = 6u;

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
        let p = project_to_screen(point);
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
        let seen = screen_hit_radiance(p.xy);
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

    let screen = trace_screen(world + normal * 0.02, direction, SCREEN_DISTANCE, SCREEN_STEPS, SCREEN_THICKNESS);
    var radiance = vec3f(0.0);
    if screen.hit {
        radiance = screen_hit_radiance(screen.uv);
    }
    if screen.weight < 1.0 {
        let level = min(volume_level_at(world, 0u), VOLUME_LEVELS - 1u);
        let voxel = volume_params.levels[level].w;
        let hit = trace_refined(world + normal * voxel * 1.5, direction, voxel * 0.5);
        let sky = sky_radiance(toward_horizon(direction));
        // Fade by where the ray ended up. A ray leaving the volume before reaching something sees
        // only sky, so hits near the edge must already be blending towards that, or the edge shows
        // as a hard line across glossy surfaces.
        let hit_point = world + direction * hit.t;
        let far = mix(sky, select(sky, hit.radiance, hit.hit), volume_edge_fade(hit_point) * volume_edge_fade(world));
        radiance = mix(far, radiance, screen.weight);
    }

    // Fireflies would otherwise persist through the temporal filter. At night the limit stays
    // where exposure stops brightening, or a dim or absent moon would black out the sky and
    // lamps seen in reflections.
    let peak = max(radiance.x, max(radiance.y, radiance.z));
    let limit = max(frame.sun_illuminance, frame.daylight_illuminance / 128.0) * 4.0;
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
