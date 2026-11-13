// Turns the visibility buffer into a G-buffer.
//
// With a visibility buffer, attributes are not interpolated by the rasterizer. Each
// pixel fetches its triangle and computes barycentrics itself. Evaluating them again
// at the neighbouring pixels gives exact analytic derivatives for texture filtering,
// with none of the seams that quad derivatives would cause on tiny triangles.

#include "common/math.wgsl"
#include "common/scene.wgsl"
#include "common/skin.wgsl"
#include "common/skin_vertex.wgsl"
#include "common/sway_vertex.wgsl"
#include "common/frame.wgsl"
#include "common/wetness.wgsl"
#include "rain/surface.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> vis_depth: array<u32>;
@group(0) @binding(2) var<storage, read> vis_id: array<u32>;
@group(0) @binding(3) var<storage, read> visible: array<VisibleCluster>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(6) var<storage, read> vertices: array<u32>;
@group(0) @binding(7) var<storage, read> triangles: array<u32>;
@group(0) @binding(8) var<storage, read> materials: array<Material>;
@group(0) @binding(9) var bone_texture: texture_2d<f32>;
// The wind, for what sways in it: see common/sway_vertex.wgsl.
@group(0) @binding(10) var wind_field: texture_2d<f32>;

// The page pool, bound twice. See common/pool.wgsl.
fn vertex_word(word: u32) -> u32 {
    return vertices[word];
}

fn triangle_word(word: u32) -> u32 {
    return triangles[word];
}

@group(1) @binding(0) var out_albedo: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(1) var out_normal: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var out_emissive: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var out_motion: texture_storage_2d<rg32float, write>;
// Left out on a GPU allowing four storage textures per stage, where vg/resolve_depth.wgsl
// writes depth instead.
// depth:begin
@group(1) @binding(4) var out_depth: texture_storage_2d<r32float, write>;
// depth:end
// Each texture lives in one of three arrays, per `Material.texture_tier`.
@group(1) @binding(5) var material_sampler: sampler;
@group(1) @binding(6) var base_color: texture_2d_array<f32>;
@group(1) @binding(7) var base_normal: texture_2d_array<f32>;
@group(1) @binding(8) var base_material: texture_2d_array<f32>;
@group(1) @binding(9) var middle_color: texture_2d_array<f32>;
@group(1) @binding(10) var middle_normal: texture_2d_array<f32>;
@group(1) @binding(11) var middle_material: texture_2d_array<f32>;
@group(1) @binding(12) var top_color: texture_2d_array<f32>;
@group(1) @binding(13) var top_normal: texture_2d_array<f32>;
@group(1) @binding(14) var top_material: texture_2d_array<f32>;
// glow:begin
// The emissive maps, for a scene that has any. Compiled out of one that hasn't.
@group(1) @binding(15) var glow_maps: texture_2d_array<f32>;
// `flags` above this bit: one more than the material's layer among the emissive maps, or zero.
const MATERIAL_GLOW_SHIFT: u32 = 16u;
// `Material.emissive` of a material with a map is the map's light taken over its surface, as this
// fraction of it glows: `Material::GLOW_MAP_AVERAGE`.
const MATERIAL_GLOW_AVERAGE: f32 = 0.1;
// glow:end
// Where the sea has been, for how wet each surface is.
@group(2) @binding(0) var<uniform> waterline: WaterlineParams;
@group(2) @binding(1) var waterline_map: texture_2d<f32>;

const TIER_MIDDLE: u32 = 1u;
const TIER_TOP: u32 = 2u;

fn sample_color(tier: u32, uv: vec2f, layer: i32, dx: vec2f, dy: vec2f) -> vec4f {
    switch tier {
        case TIER_TOP: { return textureSampleGrad(top_color, material_sampler, uv, layer, dx, dy); }
        case TIER_MIDDLE: { return textureSampleGrad(middle_color, material_sampler, uv, layer, dx, dy); }
        default: { return textureSampleGrad(base_color, material_sampler, uv, layer, dx, dy); }
    }
}

// The texture's average colour, from its highest mip.
fn average_color(tier: u32, uv: vec2f, layer: i32) -> vec3f {
    switch tier {
        case TIER_TOP: { return textureSampleLevel(top_color, material_sampler, uv, layer, 16.0).rgb; }
        case TIER_MIDDLE: { return textureSampleLevel(middle_color, material_sampler, uv, layer, 16.0).rgb; }
        default: { return textureSampleLevel(base_color, material_sampler, uv, layer, 16.0).rgb; }
    }
}

fn sample_normal(tier: u32, uv: vec2f, layer: i32, dx: vec2f, dy: vec2f) -> vec2f {
    switch tier {
        case TIER_TOP: { return textureSampleGrad(top_normal, material_sampler, uv, layer, dx, dy).rg; }
        case TIER_MIDDLE: { return textureSampleGrad(middle_normal, material_sampler, uv, layer, dx, dy).rg; }
        default: { return textureSampleGrad(base_normal, material_sampler, uv, layer, dx, dy).rg; }
    }
}

fn sample_material(tier: u32, uv: vec2f, layer: i32, dx: vec2f, dy: vec2f) -> vec2f {
    switch tier {
        case TIER_TOP: { return textureSampleGrad(top_material, material_sampler, uv, layer, dx, dy).rg; }
        case TIER_MIDDLE: { return textureSampleGrad(middle_material, material_sampler, uv, layer, dx, dy).rg; }
        default: { return textureSampleGrad(base_material, material_sampler, uv, layer, dx, dy).rg; }
    }
}

// Perspective-correct barycentrics of a pixel inside a triangle.
fn barycentrics(clip: array<vec4f, 3>, ndc: vec2f) -> vec3f {
    // Solve in homogeneous coordinates. Dividing each vertex by w first produces
    // NaNs when a triangle crosses the eye plane with a vertex exactly on it.
    let x = vec3f(clip[0].x, clip[1].x, clip[2].x) - ndc.x * vec3f(clip[0].w, clip[1].w, clip[2].w);
    let y = vec3f(clip[0].y, clip[1].y, clip[2].y) - ndc.y * vec3f(clip[0].w, clip[1].w, clip[2].w);
    let weighted = cross(x, y);
    let total = weighted.x + weighted.y + weighted.z;
    if abs(total) > 1.0e-30 {
        let b = weighted / total;
        if finite3(b) {
            return b;
        }
    }
    // Degenerate projection (including an extrapolated derivative at infinity).
    return vec3f(1.0, 0.0, 0.0);
}

fn interp2(b: vec3f, a0: vec2f, a1: vec2f, a2: vec2f) -> vec2f {
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

fn interp3(b: vec3f, a0: vec3f, a1: vec3f, a2: vec3f) -> vec3f {
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

// Hides the repetition of a tiling texture, using only that texture. A second copy, rotated and
// at a different scale, is mixed in so that its grid never lines up with the first. Then two far
// larger copies vary the brightness, which breaks up the pattern seen from a distance.
fn hide_tiling(tint: vec3f, color: vec4f, uv: vec2f, duv_dx: vec2f, duv_dy: vec2f, tier: u32, layer: i32, strength: f32) -> vec3f {
    const ROTATION: mat2x2f = mat2x2f(0.799, 0.602, -0.602, 0.799);
    const SECOND_SCALE: f32 = 0.73;
    const LARGE_SCALE_A: f32 = 0.083;
    const LARGE_SCALE_B: f32 = 0.0137;

    let second_transform = ROTATION * SECOND_SCALE;
    let second = sample_color(
        tier,
        second_transform * uv,
        layer,
        second_transform * duv_dx,
        second_transform * duv_dy,
    ).rgb;
    let mixed = mix(color.rgb, second, 0.5 * strength);

    // The highest mip is the texture's average, which makes these ratios centre on one.
    let average = max(luminance(average_color(tier, uv, layer)), 0.01);
    let large_a = sample_color(tier, uv * LARGE_SCALE_A, layer, duv_dx * LARGE_SCALE_A, duv_dy * LARGE_SCALE_A).rgb;
    let large_b = sample_color(tier, ROTATION * uv * LARGE_SCALE_B, layer, ROTATION * duv_dx * LARGE_SCALE_B, ROTATION * duv_dy * LARGE_SCALE_B).rgb;
    let variation = clamp(luminance(large_a) * luminance(large_b) / (average * average), 0.6, 1.45);
    return tint * mixed * mix(1.0, variation, strength);
}

// Ground blended of three kinds (land.rs): shares below this are left out, which is only near
// the ends of the blend (land.rs's BLEND_WIDTH), where breakup has faded out; how much each
// texture's relief counts against its share; and over how much of the difference between the
// leading score and the rest the others still show, which sets how crisp the meeting is.
const BLEND_LEAST: f32 = 0.002;
const BLEND_RELIEF: f32 = 0.5;
const BLEND_DEPTH: f32 = 0.25;
// How far the noise that breaks borders up moves each kind's share.
const BLEND_BREAKUP: f32 = 0.6;

// One textured material's surface at `uv`, before it meets the light.
struct Layer {
    albedo: vec3f,
    cavity: f32,
    roughness: f32,
    metallic: f32,
    // The normal map's tilt, in the tangent frame, scaled by its strength.
    tangent: vec2f,
}

fn layer_of(m: Material, uv: vec2f, duv_dx: vec2f, duv_dy: vec2f) -> Layer {
    let tier = m.texture_tier;
    let layer = i32(m.texture_slot);
    let sample_dx = duv_dx * frame.mip_scale;
    let sample_dy = duv_dy * frame.mip_scale;
    let color = sample_color(tier, uv, layer, sample_dx, sample_dy);
    let normal_map = sample_normal(tier, uv, layer, sample_dx, sample_dy);
    let material_map = sample_material(tier, uv, layer, sample_dx, sample_dy);
    var albedo = m.base_color.rgb;
    if m.macro_variation > 0.0 {
        albedo = hide_tiling(albedo, color, uv, sample_dx, sample_dy, tier, layer, m.macro_variation);
    } else {
        albedo *= color.rgb;
    }
    return Layer(
        albedo,
        // A cut-out material's alpha is its coverage.
        select(color.a, 1.0, (m.flags & MATERIAL_MASKED) != 0u),
        clamp(m.roughness * material_map.r * 2.0, 0.04, 1.0),
        m.metallic * material_map.g,
        (normal_map * 2.0 - vec2f(1.0)) * m.normal_strength,
    );
}

fn hash_color(value: u32) -> vec3f {
    let h = pcg(value);
    return vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.2;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(frame.viewport);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let pixel = id.xy;
    let index = pixel.y * size.x + pixel.x;
    let depth = bitcast<f32>(vis_depth[index]);
    // depth:begin
    textureStore(out_depth, pixel, vec4f(depth, 0.0, 0.0, 0.0));
    // depth:end
    if depth <= 0.0 {
        textureStore(out_albedo, pixel, vec4f(0.0));
        textureStore(out_normal, pixel, vec4f(0.0));
        textureStore(out_emissive, pixel, vec4f(0.0));
        // Sky moves only with camera rotation, which reprojection derives from depth.
        textureStore(out_motion, pixel, vec4f(0.0));
        return;
    }

    let packed_id = vis_id[index];
    let visible_index = packed_id >> 7u;
    let triangle_index = packed_id & 0x7fu;
    let entry = visible[visible_index];
    let instance = instances[entry.instance];
    let cluster = clusters[entry.cluster];
    let material = materials[instance_material(instance)];

    let packed = load_triangle(cluster, triangle_index);
    let v0 = load_vertex(cluster, packed & 0xffu);
    let v1 = load_vertex(cluster, (packed >> 8u) & 0xffu);
    let v2 = load_vertex(cluster, (packed >> 16u) & 0xffu);

    var local0 = decode_position(instance, v0);
    var local1 = decode_position(instance, v1);
    var local2 = decode_position(instance, v2);
    var normal0 = decode_normal(v0);
    var normal1 = decode_normal(v1);
    var normal2 = decode_normal(v2);
    // Where this frame's corners were last frame, in the mesh's own space.
    var before0 = local0;
    var before1 = local1;
    var before2 = local2;
    if instance_skinned(instance) {
        let corners = vec3u(packed & 0xffu, (packed >> 8u) & 0xffu, (packed >> 16u) & 0xffu);
        let skin0 = vertex_skin(cluster, corners.x);
        let skin1 = vertex_skin(cluster, corners.y);
        let skin2 = vertex_skin(cluster, corners.z);
        let now0 = skin_rows(instance.bones, skin0, false);
        let now1 = skin_rows(instance.bones, skin1, false);
        let now2 = skin_rows(instance.bones, skin2, false);
        before0 = skin_point(skin_rows(instance.bones, skin0, true), local0);
        before1 = skin_point(skin_rows(instance.bones, skin1, true), local1);
        before2 = skin_point(skin_rows(instance.bones, skin2, true), local2);
        local0 = skin_point(now0, local0);
        local1 = skin_point(now1, local1);
        local2 = skin_point(now2, local2);
        normal0 = normalize(skin_direction(now0, normal0));
        normal1 = normalize(skin_direction(now1, normal1));
        normal2 = normalize(skin_direction(now2, normal2));
    }
    var world0 = transform_point(instance, local0);
    var world1 = transform_point(instance, local1);
    var world2 = transform_point(instance, local2);
    var world_normal0 = transform_normal(instance, normal0);
    var world_normal1 = transform_normal(instance, normal1);
    var world_normal2 = transform_normal(instance, normal2);
    // Where the wind had each corner last frame, from where it stands at rest, in the world.
    var sway_before0 = vec3f(0.0);
    var sway_before1 = vec3f(0.0);
    var sway_before2 = vec3f(0.0);
    if instance_sways(instance) {
        let w = wind_params();
        if w.sway != 0.0 {
            // As the rasterizers placed them (vg/raster_common.wgsl), and turned with them.
            let eye = frame.camera_position;
            let s0 = vertex_sway(w, instance, local0, world0, eye);
            let s1 = vertex_sway(w, instance, local1, world1, eye);
            let s2 = vertex_sway(w, instance, local2, world2, eye);
            sway_before0 = s0.before;
            sway_before1 = s1.before;
            sway_before2 = s2.before;
            world0 += s0.now;
            world1 += s1.now;
            world2 += s2.now;
            world_normal0 = bend_direction(s0.bend, world_normal0);
            world_normal1 = bend_direction(s1.bend, world_normal1);
            world_normal2 = bend_direction(s2.bend, world_normal2);
        }
    }
    let clip = array<vec4f, 3>(
        frame.view_proj * vec4f(world0, 1.0),
        frame.view_proj * vec4f(world1, 1.0),
        frame.view_proj * vec4f(world2, 1.0),
    );

    let center = vec2f(pixel) + vec2f(0.5);
    let ndc = pixel_to_ndc(center, frame.inv_viewport);
    let ndc_right = pixel_to_ndc(center + vec2f(1.0, 0.0), frame.inv_viewport);
    let ndc_below = pixel_to_ndc(center + vec2f(0.0, 1.0), frame.inv_viewport);
    let b = barycentrics(clip, ndc);
    let b_x = barycentrics(clip, ndc_right);
    let b_y = barycentrics(clip, ndc_below);

    let world = interp3(b, world0, world1, world2);
    let dp_dx = interp3(b_x, world0, world1, world2) - world;
    let dp_dy = interp3(b_y, world0, world1, world2) - world;

    let uv0 = decode_uv(instance, v0) * material.uv_scale;
    let uv1 = decode_uv(instance, v1) * material.uv_scale;
    let uv2 = decode_uv(instance, v2) * material.uv_scale;
    let uv = interp2(b, uv0, uv1, uv2);
    let duv_dx = interp2(b_x, uv0, uv1, uv2) - uv;
    let duv_dy = interp2(b_y, uv0, uv1, uv2) - uv;

    var normal = normalize(interp3(b, world_normal0, world_normal1, world_normal2));
    // Seen from behind, a two-sided surface is lit as its other side.
    let face = cross(world1 - world0, world2 - world0);
    if (material.flags & MATERIAL_TWO_SIDED) != 0u && dot(face, frame.camera_position - world0) < 0.0 {
        normal = -normal;
    }
    // The geometry's own normal, before any normal map, for water to pool against.
    let surface_normal = normal;

    var albedo = material.base_color.rgb;
    var roughness = material.roughness;
    var cavity = 1.0;
    var metallic = material.metallic;
    if material.texture_layer >= 0 {
        var duv_x = duv_dx;
        var duv_y = duv_dy;
        var surface = layer_of(material, uv, duv_dx, duv_dy);
        if (material.flags & MATERIAL_BLENDS) != 0u {
            // Ground of three kinds blended (land.rs): this material's, and the two after it.
            // The mesh's texture coordinates hold how much of the second and third there is, and
            // every texture is laid by the world's x and z.
            let index = instance_material(instance);
            let raw = uv / max(material.uv_scale, 1.0e-6);
            let shares = vec3f(1.0 - raw.x - raw.y, raw.x, raw.y);
            // Where kinds meet, each border breaks up into tongues and islands a metre or two
            // across, as sand blows over a path and grass creeps onto it, not one smooth line.
            var breakup = vec3f(0.0);
            if max(shares.x, max(shares.y, shares.z)) < 0.999 {
                let p = vec3f(world.x, 0.0, world.z);
                let sand = gradient_noise(p * 0.45, 811u).x + 0.5 * gradient_noise(p * 1.3, 812u).x;
                let dirt = gradient_noise(p * 0.45 + 17.0, 813u).x + 0.5 * gradient_noise(p * 1.3 + 17.0, 814u).x;
                // Faded out with each share, so a kind leaves off smoothly before it is skipped.
                breakup = vec3f(0.0, sand, dirt) * BLEND_BREAKUP
                    * smoothstep(vec3f(BLEND_LEAST), vec3f(0.1), shares);
            }
            var layers: array<Layer, 3>;
            var scores = vec3f(-1.0);
            for (var k = 0u; k < 3u; k++) {
                if shares[k] <= BLEND_LEAST {
                    continue;
                }
                let m = materials[index + k];
                let scale = m.uv_scale;
                layers[k] = layer_of(m, world.xz * scale, dp_dx.xz * scale, dp_dy.xz * scale);
                // Each texture's relief, from its cavity: what stands proud shows first where
                // two kinds meet, so grass shows on the bumps and sand settles in the hollows.
                scores[k] = shares[k] + breakup[k] + layers[k].cavity * BLEND_RELIEF;
            }
            let top = max(scores.x, max(scores.y, scores.z));
            let weights = max(scores - vec3f(top - BLEND_DEPTH), vec3f(0.0));
            let blend = weights / max(weights.x + weights.y + weights.z, 1.0e-5);
            surface = Layer(vec3f(0.0), 0.0, 0.0, 0.0, vec2f(0.0));
            for (var k = 0u; k < 3u; k++) {
                if blend[k] > 0.0 {
                    surface.albedo += layers[k].albedo * blend[k];
                    surface.cavity += layers[k].cavity * blend[k];
                    surface.roughness += layers[k].roughness * blend[k];
                    surface.metallic += layers[k].metallic * blend[k];
                    surface.tangent += layers[k].tangent * blend[k];
                }
            }
            // The tangent frame follows the world's x and z, as all three are laid.
            duv_x = dp_dx.xz;
            duv_y = dp_dy.xz;
        }
        albedo = surface.albedo;
        cavity = surface.cavity;
        roughness = surface.roughness;
        metallic = surface.metallic;

        // Normal mapping without stored tangents (Mikkelsen 2010): the tangent frame
        // comes from the same analytic derivatives used for texture filtering.
        let tangent_xy = surface.tangent;
        let tangent_z = sqrt(max(1.0 - dot(tangent_xy, tangent_xy), 0.0));
        let perp_y = cross(dp_dy, normal);
        let perp_x = cross(normal, dp_dx);
        // These come out scaled by the signed area of a pixel on the surface. Pixel rows count
        // downwards, which makes that negative on any surface facing the camera, and left as it
        // is that would turn every bump into a dent.
        let pixel_area = select(-1.0, 1.0, dot(cross(dp_dx, dp_dy), normal) >= 0.0);
        let tangent = (perp_y * duv_x.x + perp_x * duv_y.x) * pixel_area;
        let bitangent = (perp_y * duv_x.y + perp_x * duv_y.y) * pixel_area;
        let scale = max(dot(tangent, tangent), dot(bitangent, bitangent));
        if scale > 1.0e-20 {
            let inv = 1.0 / sqrt(scale);
            normal = normalize(tangent * inv * tangent_xy.x + bitangent * inv * tangent_xy.y + normal * tangent_z);
        }
    }

    // Water on the surface, where the sea has been (docs/wetness.md).
    // And where the rain has fallen, with the puddles it has left (rain/surface.wgsl).
    let sea = sea_wetness(roughness, metallic, cavity, world);
    let rained = rain_wetness(world, surface_normal);
    // Rain drains through what is porous, sand and soil, which darken but keep only a sheen;
    // smooth stone and paint take a full film. Puddles are water standing on either.
    let drains = mix(1.0, RAIN_POROUS_WETNESS, porosity(roughness, metallic));
    let rain_wet = clamp(
        max(min(rained.wet, drains), rained.puddle) * (1.0 + (1.0 - cavity) * 0.25),
        0.0,
        1.0,
    );
    var wet = wet_material(albedo, roughness, metallic, normal, surface_normal, max(sea.x, rain_wet), sea.y);
    wet = under_puddle(wet, surface_normal, rained);
    albedo = wet.albedo;
    roughness = wet.roughness;
    normal = wet.normal;

    // Motion, in UV units from this pixel to where it was last frame.
    let local = interp3(b, before0, before1, before2);
    // Where the instance had it last frame, and the wind carried it then.
    let prev_world = transform_point_prev(instance, local) + interp3(b, sway_before0, sway_before1, sway_before2);
    let prev_clip = frame.prev_view_proj * vec4f(prev_world, 1.0);
    let current_clip = frame.unjittered_view_proj * vec4f(world, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let current_ndc = current_clip.xy / current_clip.w;
    let motion = (prev_ndc - current_ndc) * vec2f(0.5, -0.5);

    var emissive = material.emissive * instance.glow;
    // glow:begin
    let glow_layer = material.flags >> MATERIAL_GLOW_SHIFT;
    if glow_layer != 0u {
        emissive *= textureSampleGrad(glow_maps, material_sampler, uv, i32(glow_layer - 1u), duv_dx * frame.mip_scale, duv_dy * frame.mip_scale).rgb / MATERIAL_GLOW_AVERAGE;
    }
    // glow:end

    switch frame.debug_mode {
        case DEBUG_CLUSTERS: {
            emissive = hash_color(entry.cluster * 31u + entry.instance * 977u) * max(dot(normal, frame.sun_direction) * 0.5 + 0.5, 0.1);
        }
        case DEBUG_TRIANGLES: {
            emissive = hash_color(packed_id) * max(dot(normal, frame.sun_direction) * 0.5 + 0.5, 0.1);
        }
        case DEBUG_RASTER_MODE: {
            // The hardware ranges grow down from the ends of their lists. The opaque list has four
            // fifths of the entries if the scene has cut-outs too, and these are never so many
            // as to tell that apart; the masked list has the last fifth.
            let middle = select(arrayLength(&visible) / 2u, arrayLength(&visible) / 10u * 9u, instance_masked(instance));
            let hardware = visible_index >= middle;
            emissive = select(vec3f(0.1, 0.9, 0.2), vec3f(1.0, 0.15, 0.1), hardware) * max(dot(normal, frame.sun_direction) * 0.5 + 0.5, 0.1);
        }
        case DEBUG_INSTANCES: {
            emissive = hash_color(entry.instance) * max(dot(normal, frame.sun_direction) * 0.5 + 0.5, 0.1);
        }
        case DEBUG_ALBEDO: {
            emissive = albedo;
        }
        case DEBUG_NORMALS: {
            emissive = normal * 0.5 + vec3f(0.5);
        }
        case DEBUG_MOTION: {
            // `motion` points from here to last frame's place, so this shows the way it went:
            // a hundredth of the screen per frame is full colour.
            emissive = vec3f(clamp(vec2f(0.5) - vec2f(motion.x, -motion.y) * 50.0, vec2f(0.0), vec2f(1.0)), 0.5);
        }
        case DEBUG_WIND: {
            let w = wind_params();
            var shown = vec3f(0.02);
            if SWAYING && w.texel > 0.0 {
                // Hue the way it blows, brightness how hard: 12 m/s 10 m up is full.
                let v = wind_sample(w, world.xz).xy;
                let hue = atan2(v.y, v.x) / TAU + 0.5;
                let k = fract(vec3f(hue) + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3f(3.0);
                let tint = mix(vec3f(1.0), clamp(abs(k) - vec3f(1.0), vec3f(0.0), vec3f(1.0)), 0.6);
                shown = tint * clamp(length(v) / 12.0, 0.0, 1.0);
            }
            emissive = shown * max(dot(normal, frame.sun_direction) * 0.5 + 0.5, 0.1);
        }
        case DEBUG_WETNESS: {
            let shade = max(dot(normal, frame.sun_direction) * 0.5 + 0.5, 0.1);
            emissive = mix(vec3f(0.12), vec3f(0.1, 0.45, 1.0), wet.wetness) * shade;
        }
        default: {}
    }

    textureStore(out_albedo, pixel, vec4f(albedo, cavity));
    textureStore(out_normal, pixel, vec4f(oct_encode(normal), roughness, metallic));
    // Alpha holds the shading model for the lighting pass.
    textureStore(out_emissive, pixel, vec4f(emissive, f32(material.flags & MATERIAL_FOLIAGE)));
    textureStore(out_motion, pixel, vec4f(motion, 0.0, 0.0));
}
