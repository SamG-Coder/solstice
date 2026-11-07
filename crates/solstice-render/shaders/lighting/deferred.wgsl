// Combines everything into the HDR scene colour: direct sun with shadow maps and contact
// shadows, local lights, indirect diffuse and specular from the GI system, and sky.

#include "common/environment.wgsl"
#include "common/brdf.wgsl"
#include "common/scene.wgsl"

const CASCADE_COUNT: u32 = 4u;

struct Shadows {
    view_proj: array<mat4x4f, 4>,
    // View distance at the far end of each cascade.
    splits: vec4f,
    // World size of one shadow texel per cascade.
    texel_size: vec4f,
}

@group(0) @binding(0) var albedo_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var emissive_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var gi_tex: texture_2d<f32>;
@group(0) @binding(5) var reflection_tex: texture_2d<f32>;
@group(0) @binding(6) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> shadows: Shadows;
@group(0) @binding(8) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(9) var shadow_sampler: sampler_comparison;
@group(0) @binding(10) var<storage, read> lights: array<Light>;

// Distance over which distant objects dissolve into the air, as extinction per metre.
const HAZE: f32 = 0.0011;
const SUN_ANGULAR_RADIUS: f32 = 0.0065;

const DISK_SAMPLES: u32 = 12u;

// Fraction of light a leaf or blade passes on to its far side, relative to what it reflects.
// Measured leaf spectra transmit about as much as they reflect across visible light (Jacquemoud
// and Ustin 2019), slightly less towards blue.
const FOLIAGE_TRANSMITTANCE: vec3f = vec3f(0.8, 0.8, 0.6);

fn sample_cascade(cascade: u32, world: vec3f, normal: vec3f, angle: f32) -> f32 {
    let texel = shadows.texel_size[cascade];
    // Offset towards the surface normal, by more as the sun grazes the surface: a texel then
    // spans a long stretch of it, and the surface would otherwise shadow itself. This matters
    // most in the far cascades, whose texels are large.
    let n_dot_l = clamp(dot(normal, frame.sun_direction), 0.05, 1.0);
    let slope = min(sqrt(1.0 - n_dot_l * n_dot_l) / n_dot_l, 8.0);
    let offset_world = world + normal * texel * (1.2 + slope * 0.8);
    let clip = shadows.view_proj[cascade] * vec4f(offset_world, 1.0);
    let uv = vec2f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || clip.z > 1.0 {
        return 1.0;
    }

    // Penumbra sized in world units, so that it stays consistent across cascades. A rotated
    // Vogel disc turns the blocky steps of few samples into noise that TAA can resolve.
    let filter_radius = max(0.018 / texel, 1.0) * 1.5 / f32(textureDimensions(shadow_maps).x);
    var sum = 0.0;
    for (var i = 0u; i < DISK_SAMPLES; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(DISK_SAMPLES));
        let theta = f32(i) * 2.399963 + angle;
        let offset = vec2f(cos(theta), sin(theta)) * r * filter_radius;
        sum += textureSampleCompareLevel(shadow_maps, shadow_sampler, uv + offset, i32(cascade), clip.z);
    }
    return sum / f32(DISK_SAMPLES);
}

fn sun_shadow(world: vec3f, normal: vec3f, distance: f32, noise: f32) -> f32 {
    var cascade = CASCADE_COUNT;
    for (var i = 0u; i < CASCADE_COUNT; i++) {
        if distance < shadows.splits[i] {
            cascade = i;
            break;
        }
    }
    if cascade >= CASCADE_COUNT {
        return 1.0;
    }
    // Dither across the end of a cascade, so that the change in resolution is invisible.
    let split = shadows.splits[cascade];
    let previous_split = select(0.0, shadows.splits[cascade - 1u], cascade > 0u);
    let blend_start = mix(previous_split, split, 0.85);
    if distance > blend_start && cascade + 1u < CASCADE_COUNT {
        if noise < (distance - blend_start) / (split - blend_start) {
            cascade += 1u;
        }
    }
    var shadow = sample_cascade(cascade, world, normal, noise * TAU);
    // Fade out towards the end of the last cascade, so that shadows have no visible edge.
    if cascade + 1u == CASCADE_COUNT {
        shadow = mix(shadow, 1.0, smoothstep(split * 0.8, split, distance));
    }
    return shadow;
}

// A short ray through the depth buffer towards the sun. Catches the small-scale shadowing
// of dense geometry, which is finer than any shadow map texel.
fn contact_shadow(world: vec3f, normal: vec3f, view: vec3f, distance: f32, noise: f32) -> f32 {
    const STEPS: u32 = 10u;
    const LENGTH: f32 = 0.35;
    const MAX_GAP: f32 = 0.12;
    // The depth that one pixel of this surface spans: small when facing the camera, large at
    // grazing angles. A closer depth than that may just be the neighbouring pixel of the same
    // surface, so it proves nothing. Where this exceeds MAX_GAP, the test cannot tell the two
    // apart, and it is right to do nothing.
    let pixel_angle = 1.04 * frame.inv_viewport.y;
    let tolerance = max(0.002, 2.0 * distance * pixel_angle / max(abs(dot(normal, view)), 0.02));
    if tolerance >= MAX_GAP {
        return 1.0;
    }
    let start = world + normal * 0.01;
    let size = vec2i(frame.viewport);
    for (var i = 0u; i < STEPS; i++) {
        let t = (f32(i) + noise) / f32(STEPS) * LENGTH;
        let clip = frame.view_proj * vec4f(start + frame.sun_direction * t, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0)) {
            return 1.0;
        }
        let scene_depth = textureLoad(depth_tex, min(vec2i(uv * frame.viewport), size - vec2i(1)), 0).r;
        if scene_depth > ndc.z {
            let gap = frame.near / ndc.z - frame.near / scene_depth;
            if gap > tolerance && gap < MAX_GAP {
                return 1.0 - clamp(1.0 - t / LENGTH, 0.0, 1.0) * 0.85;
            }
        }
    }
    return 1.0;
}

fn local_light(light: Light, s: Surface, view: vec3f, world: vec3f) -> vec3f {
    let to_light = light.position - world;
    let distance = length(to_light);
    if distance >= light.range {
        return vec3f(0.0);
    }
    let direction = to_light / distance;
    if dot(s.normal, direction) <= 0.0 {
        return vec3f(0.0);
    }
    let window = sq(clamp(1.0 - sq(sq(distance / light.range)), 0.0, 1.0));
    let falloff = window / max(distance * distance, 0.04);
    let level = min(volume_level_at(world, 0u), VOLUME_LEVELS - 1u);
    let voxel = volume_params.levels[level].w;
    let open = volume_visibility(
        world + s.normal * voxel * 1.5,
        direction,
        voxel * 0.5,
        max(distance - voxel * 2.5, 0.0),
        16u,
    );
    return shade_light(s, view, direction) * light.intensity * falloff * open;
}

fn sun_disc(direction: vec3f) -> vec3f {
    let cos_angle = dot(direction, frame.sun_direction);
    let edge = cos(SUN_ANGULAR_RADIUS);
    let disc = smoothstep(edge - 0.00002, edge + 0.00002, cos_angle);
    // The true disc would be vastly brighter. This keeps it in a range that still blooms,
    // without one pixel's value dominating the exposure.
    return sun_irradiance() * disc * 120.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(frame.viewport);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2i(id.xy);
    let center = vec2f(id.xy) + vec2f(0.5);
    let depth = textureLoad(depth_tex, pixel, 0).r;
    let emissive_model = textureLoad(emissive_tex, pixel, 0);
    let emissive = emissive_model.rgb;
    let foliage = emissive_model.a > 0.5;

    if depth <= 0.0 {
        let ndc = pixel_to_ndc(center, frame.inv_viewport);
        let far_point = frame.inv_view_proj * vec4f(ndc, 1.0e-6, 1.0);
        let direction = normalize(far_point.xyz / far_point.w - frame.camera_position);
        var sky = sky_radiance(direction);
        if frame.debug_mode == DEBUG_NONE {
            sky += sun_disc(direction);
        }
        textureStore(output, pixel, vec4f(sky, 1.0));
        return;
    }

    // Views that the resolve pass wrote directly to emissive are shown as they are.
    if frame.debug_mode >= DEBUG_CLUSTERS && frame.debug_mode <= DEBUG_NORMALS {
        textureStore(output, pixel, vec4f(emissive, 1.0));
        return;
    }

    let albedo_cavity = textureLoad(albedo_tex, pixel, 0);
    let packed = textureLoad(normal_tex, pixel, 0);
    let normal = oct_decode(packed.xy);
    let roughness = packed.z;
    let metallic = packed.w;
    let world = reconstruct_world(frame, center, depth);
    let to_camera = frame.camera_position - world;
    let distance = length(to_camera);
    let view = to_camera / distance;
    let surface = make_surface(albedo_cavity.rgb, metallic, roughness, normal);

    let noise = interleaved_gradient_noise(vec2f(id.xy), frame.frame_index);

    // Sun.
    var color = vec3f(0.0);
    let sun_facing = dot(normal, frame.sun_direction);
    if sun_facing > 0.0 {
        var shadow = sun_shadow(world, normal, distance, noise);
        if shadow > 0.0 && distance < 40.0 {
            shadow *= contact_shadow(world, normal, view, distance, noise);
        }
        color += shade_light(surface, view, frame.sun_direction) * sun_irradiance() * shadow;
    } else if foliage {
        // Sunlight through a leaf from behind. The shadow is looked up for the lit side.
        let shadow = sun_shadow(world, -normal, distance, noise);
        let transmitted = surface.diffuse * FOLIAGE_TRANSMITTANCE / PI * -sun_facing;
        color += transmitted * sun_irradiance() * shadow;
    }

    // Local lights.
    for (var i = 0u; i < frame.light_count; i++) {
        color += local_light(lights[i], surface, view, world);
    }

    // Indirect diffuse. Cavity darkens crevices the probes are too coarse to resolve.
    let gi = textureLoad(gi_tex, pixel, 0);
    let occlusion = gi.a * mix(1.0, albedo_cavity.a, 0.7);
    let irradiance = gi.rgb * frame.gi_intensity;
    color += surface.diffuse / PI * irradiance * occlusion;
    if foliage {
        // Skylight through the leaf. Probes only know the light arriving at its front, so the
        // back is estimated from the same, as it is for most foliage in a canopy.
        color += surface.diffuse * FOLIAGE_TRANSMITTANCE / PI * irradiance * occlusion * 0.5;
    }

    // Indirect specular. Rough surfaces did not trace, and their blurry reflection is
    // reasonably well estimated from the same irradiance.
    let reflection_uv = center * frame.inv_viewport;
    let reflection = textureSampleLevel(reflection_tex, clamp_sampler, reflection_uv, 0.0);
    let n_dot_v = max(dot(normal, view), 0.0);
    let specular_scale = environment_brdf(surface.f0, roughness, n_dot_v);
    let rough_estimate = irradiance / PI;
    let traced_weight = clamp((0.65 - roughness) / 0.1, 0.0, 1.0) * clamp(reflection.a * 4.0, 0.0, 1.0);
    let incoming = mix(rough_estimate, reflection.rgb * frame.gi_intensity, traced_weight);
    color += incoming * specular_scale * occlusion;

    color += emissive;

    // Aerial perspective: distant objects fade towards the sky colour behind them.
    let transmittance = exp(-distance * HAZE);
    // Sampled above the horizon and partly desaturated: the air between here and a distant
    // object scatters light from all around, not only the long path to the horizon line.
    let haze_direction = normalize(vec3f(-view.x, max(-view.y, 0.2), -view.z));
    let haze_sky = sky_radiance(haze_direction);
    var haze_color = mix(vec3f(luminance(haze_sky)), haze_sky, 0.55);
    // That direction can land on the glow around the sun even when looking down at nearby ground,
    // whose radiance is so high that this small share of it shows as smoke across shadows. So the
    // haze is limited relative to the open sky on the side away from the sun.
    let away_from_sun = normalize(vec3f(-frame.sun_direction.x, 0.2, -frame.sun_direction.z));
    let haze_limit = luminance(sky_radiance(away_from_sun)) * 2.0;
    haze_color *= min(haze_limit / max(luminance(haze_color), 1.0e-4), 1.0);
    color = mix(haze_color, color, transmittance);

    switch frame.debug_mode {
        case DEBUG_DIFFUSE_GI: {
            color = irradiance * occlusion / PI;
        }
        case DEBUG_REFLECTIONS: {
            color = incoming;
        }
        case DEBUG_VOXELS: {
            let hit = trace_volume(frame.camera_position, -view, 0.0, 320.0, 96u);
            color = select(sky_radiance(-view), hit.radiance, hit.hit);
        }
        default: {}
    }
    textureStore(output, pixel, vec4f(color, 1.0));
}
