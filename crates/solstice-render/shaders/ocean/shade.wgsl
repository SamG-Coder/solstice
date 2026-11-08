// The water pass: where the sea's surface is nearer than the opaque scene, shades it and puts it
// over the lit scene, and gives the water's own depth and motion to everything after. Elsewhere
// the lit scene passes through unchanged.
//
// The shading is the first, simple form (docs/ocean.md, step 2): the sky reflected by Fresnel's
// law, the sun's highlight, and the glow of the water's body from its own absorption and
// scattering, all through the same haze as everything else. Refraction, shadows on the water and
// foam come in later steps.

#include "common/environment.wgsl"
#include "common/brdf.wgsl"
#include "ocean/surface_params.wgsl"
#include "lighting/sun_shadow.wgsl"
#include "gi/screen_trace.wgsl"

// The opaque scene, lit, with its depth and motion.
@group(0) @binding(0) var lit: texture_2d<f32>;
@group(0) @binding(1) var opaque_depth: texture_2d<f32>;
@group(0) @binding(2) var opaque_motion: texture_2d<f32>;
// The water layer: its depth, and where its water lay on the calm sea.
@group(0) @binding(3) var water_depth: texture_depth_2d;
@group(0) @binding(4) var water_calm: texture_2d<f32>;
// The waves' slopes and stretch, and their displacement (its shear in w), per cascade.
@group(0) @binding(5) var derivatives: texture_2d_array<f32>;
@group(0) @binding(6) var displacement: texture_2d_array<f32>;
@group(0) @binding(7) var wave_sampler: sampler;
@group(0) @binding(8) var<uniform> surface: SurfaceParams;
@group(0) @binding(9) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(11) var out_motion: texture_storage_2d<rg32float, write>;
// The sun's shadows, as the lighting pass has them.
@group(0) @binding(12) var<uniform> shadows: Shadows;
@group(0) @binding(13) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(14) var shadow_sampler: sampler_comparison;
// For reflections of the scene: the opaque scene's closest-depth pyramid, and its normals.
@group(0) @binding(15) var depth_pyramid: texture_2d<f32>;
@group(0) @binding(16) var normal_tex: texture_2d<f32>;

const DEBUG_OCEAN: u32 = 10u;
// Water's reflectance looking straight down: from its index of refraction, 1.33.
const WATER_F0: f32 = 0.02;
// Roughness of the water's surface between its drawn waves, for the sun's highlight. The
// spectrum's finer ripples will set this properly (step 3).
const WATER_ROUGHNESS: f32 = 0.08;
// How far reflections are traced across the screen, and how thick a surface they can hit.
const REFLECTION_DISTANCE: f32 = 300.0;
const REFLECTION_STEPS: u32 = 64u;
const REFLECTION_THICKNESS: f32 = 0.5;
// Depth of water over which the sea fades into the shore, in metres.
const SHORE_FADE: f32 = 0.25;

// The surface's normal at `calm`, from every cascade's slopes at the level whose texels match
// `footprint` metres.
fn wave_normal(calm: vec2f, footprint: f32) -> vec3f {
    var d = vec4f(0.0);
    var shear = 0.0;
    for (var cascade = 0u; cascade < 3u; cascade++) {
        let tile = surface.sizes[cascade];
        let level = max(log2(footprint / (tile / surface.grid.x)), 0.0);
        d += textureSampleLevel(derivatives, wave_sampler, calm / tile, cascade, level);
        shear += textureSampleLevel(displacement, wave_sampler, calm / tile, cascade, level).w;
    }
    // The displaced surface's tangents along x and z, with the horizontal displacement's
    // stretch and shear (∂Dz/∂x = ∂Dx/∂z), and their cross product. Where a crest folds over,
    // the surface turns back on itself; it is held upright rather than flipped.
    let choppiness = surface.grid.y;
    let along_x = vec3f(1.0 + choppiness * d.z, d.x, choppiness * shear);
    let along_z = vec3f(choppiness * shear, d.y, 1.0 + choppiness * d.w);
    var normal = cross(along_z, along_x);
    normal.y = max(normal.y, 0.05);
    return normalize(normal);
}

// What the water at `world`, facing `normal`, reflects along `direction`: the scene where the
// screen shows it, then the scene's voxels, then the sky. As the reflections pass traces, but
// a hit's colour is this frame's, as the water is shaded after the scene is lit.
fn reflection(world: vec3f, normal: vec3f, direction: vec3f) -> vec3f {
    let sky = sky_radiance(toward_horizon(direction));
    // A ray reflected down into the water reflects nothing but the sky above the waves.
    if direction.y <= 0.0 {
        return sky;
    }
    let origin = world + normal * 0.05;
    let screen = trace_screen(origin, direction, REFLECTION_DISTANCE, REFLECTION_STEPS, REFLECTION_THICKNESS);
    var seen = vec3f(0.0);
    if screen.hit {
        seen = textureSampleLevel(lit, clamp_sampler, screen.uv, 0.0).rgb;
    }
    if screen.weight >= 1.0 {
        return seen;
    }
    // Off the screen: the volume of voxels around the camera, then the sky beyond it.
    let level = min(volume_level_at(world, 0u), VOLUME_LEVELS - 1u);
    let voxel = volume_params.levels[level].w;
    let hit = trace_volume_with(origin + normal * voxel, direction, voxel * 0.5, 320.0, 48u, 0.55);
    let hit_point = origin + direction * hit.t;
    let far = mix(sky, select(sky, hit.radiance, hit.hit), volume_edge_fade(hit_point) * volume_edge_fade(world));
    return mix(far, seen, screen.weight);
}

// What is seen through the water at `world`: the lit scene behind it, bent by refraction where
// the view ray enters, and dimmed and tinted by the water the light crosses, down to the bed and
// back up to the eye, with the water's own glow, `body`, filling in as the path deepens. Over
// sand a few centimetres down the water is almost clear; a metre turns it turquoise, red going
// first; deeper, and over the open sea, it is the body's colour alone.
fn through_water(
    center: vec2f,
    world: vec3f,
    view: vec3f,
    normal: vec3f,
    water_depth: f32,
    scene_depth: f32,
    body: vec3f,
    extinction: vec3f,
) -> vec3f {
    // Nothing behind this pixel but the sky: open sea, as deep as can be.
    if scene_depth <= 0.0 {
        return body;
    }
    let straight = reconstruct_world(frame, center, scene_depth);
    let thickness = distance(straight, world);
    // Follow the bent ray as far as the water is thick, and find the bed it reaches on screen.
    let bent = refract(-view, normal, 1.0 / 1.33);
    var pixel = center;
    let reached = frame.view_proj * vec4f(world + bent * thickness, 1.0);
    if reached.w > 0.0 {
        let ndc = reached.xy / reached.w;
        pixel = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * frame.viewport;
    }
    let last = vec2i(frame.viewport) - vec2i(1);
    var bed_depth = textureLoad(opaque_depth, clamp(vec2i(pixel), vec2i(0), last), 0).r;
    // What is found there must lie beyond the water's surface. Anything nearer is in front of
    // the water, and the sky is beyond the scene; either way, look straight through instead.
    if bed_depth <= 0.0 || bed_depth >= water_depth {
        pixel = center;
        bed_depth = scene_depth;
    }
    let bed = reconstruct_world(frame, pixel, bed_depth);
    let path = distance(bed, world);
    // The light reaching the bed came down through the water first.
    let down = max(surface.sizes.w - bed.y, 0.0);
    let seen = textureSampleLevel(lit, clamp_sampler, pixel * frame.inv_viewport, 0.0).rgb;
    return seen * exp(-extinction * (path + down)) + body * (vec3f(1.0) - exp(-extinction * path));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(frame.viewport);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2i(id.xy);
    let behind = textureLoad(lit, pixel, 0);
    let scene_depth = textureLoad(opaque_depth, pixel, 0).r;
    let water = textureLoad(water_depth, pixel, 0);
    // Reversed depth: nearer is larger. No water here, or something stands in front of it; or a
    // debug view of the opaque scene, which water would hide.
    let other_view = frame.debug_mode != DEBUG_NONE && frame.debug_mode != DEBUG_OCEAN;
    if water <= scene_depth || other_view {
        textureStore(out_color, pixel, behind);
        textureStore(out_depth, pixel, vec4f(scene_depth, 0.0, 0.0, 0.0));
        textureStore(out_motion, pixel, textureLoad(opaque_motion, pixel, 0));
        return;
    }

    let center = vec2f(id.xy) + vec2f(0.5);
    let world = reconstruct_world(frame, center, water);
    let to_camera = frame.camera_position - world;
    let distance = length(to_camera);
    let view = to_camera / distance;
    // How much sea one pixel covers: wider at grazing angles.
    let next = reconstruct_world(frame, center + vec2f(1.0, 0.0), water);
    let footprint = length(next - world) / sqrt(max(abs(view.y), 0.02));
    let calm = textureLoad(water_calm, pixel, 0).xy;
    var normal = wave_normal(calm, footprint);
    // Waves turned away from the camera show their backs only at grazing angles; lean them back
    // into view, as the waves in front would hide them.
    let facing = dot(normal, view);
    if facing < 0.02 {
        normal = normalize(normal + view * (0.02 - facing));
    }

    var color: vec3f;
    if frame.debug_mode == DEBUG_OCEAN {
        color = normal * 0.5 + vec3f(0.5);
    } else {
        let n_dot_v = max(dot(normal, view), 1.0e-3);
        let fresnel = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - n_dot_v, 5.0);
        let reflected = reflection(world, normal, reflect(-view, normal));

        // Sunlight at the water, after the clouds, and the shadows of what stands in its way:
        // cliffs, the fort, the ship. Shadows show on water by dimming its highlight and the
        // light it scatters from within; the sky it reflects is untouched.
        let noise = interleaved_gradient_noise(vec2f(id.xy), frame.frame_index);
        let shadow = sun_shadow(world, vec3f(0.0, 1.0, 0.0), distance, noise);
        let sun_light = sun_irradiance_at(world) * shadow;

        // The sun's highlight off the waves.
        let sun = frame.sun_direction;
        let n_dot_l = dot(normal, sun);
        var highlight = vec3f(0.0);
        if n_dot_l > 0.0 {
            let a = WATER_ROUGHNESS * WATER_ROUGHNESS;
            let h = normalize(view + sun);
            let specular = d_ggx(max(dot(normal, h), 0.0), a) * v_smith(n_dot_v, n_dot_l, a)
                * f_schlick(vec3f(WATER_F0), max(dot(view, h), 0.0));
            highlight = specular * n_dot_l * sun_light;
        }

        // Light scattered back out of the water's body: its share of what is not absorbed, of
        // the light from the sky and sun that gets in.
        let extinction = surface.absorption.rgb + surface.scattering.rgb;
        let albedo = surface.scattering.rgb / max(extinction, vec3f(1.0e-4));
        let skylight = clear_sky_radiance(vec3f(0.0, 1.0, 0.0)) * PI;
        let sunlight = sun_light * max(sun.y, 0.0);
        let body = albedo * (skylight + sunlight) / PI * 0.5;

        let underwater = through_water(center, world, view, normal, water, scene_depth, body, extinction);
        color = mix(underwater, reflected, fresnel) + highlight;
        color = aerial_perspective(color, world, view, distance);
        // At the very edge, where the water thins to nothing over the sand, it fades into the
        // shore rather than ending in a line: a sheet a few centimetres deep barely shows. The
        // shore behind has its own haze already.
        if scene_depth > 0.0 {
            let bed = reconstruct_world(frame, center, scene_depth);
            let depth_here = max(world.y - bed.y, 0.0);
            color = mix(behind.rgb, color, smoothstep(0.0, SHORE_FADE, depth_here));
        }
    }

    let prev_clip = frame.prev_view_proj * vec4f(world, 1.0);
    let current_clip = frame.unjittered_view_proj * vec4f(world, 1.0);
    let motion = (prev_clip.xy / prev_clip.w - current_clip.xy / current_clip.w) * vec2f(0.5, -0.5);

    textureStore(out_color, pixel, vec4f(color, 1.0));
    textureStore(out_depth, pixel, vec4f(water, 0.0, 0.0, 0.0));
    textureStore(out_motion, pixel, vec4f(motion, 0.0, 0.0));
}
