// The sun's cascaded shadows, sampled as the lighting and water passes both need them.
//
// Needs common/environment.wgsl, and, declared by whatever includes this:
//   var<uniform> shadows: Shadows;
//   var shadow_maps: texture_depth_2d_array;
//   var shadow_sampler: sampler_comparison;

const CASCADE_COUNT: u32 = 4u;

struct Shadows {
    view_proj: array<mat4x4f, 4>,
    // View distance at the far end of each cascade.
    splits: vec4f,
    // World size of one shadow texel per cascade.
    texel_size: vec4f,
}

const DISK_SAMPLES: u32 = 12u;

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
