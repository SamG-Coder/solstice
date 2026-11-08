// What every rasterizer of a cut-out material reads: the materials, for where each texture is
// now, and the colour maps, whose alpha is coverage. See docs/cutouts.md.

@group(2) @binding(0) var<storage, read> materials: array<Material>;
// One array per tier, as in vg/resolve.wgsl.
@group(2) @binding(1) var mask_base: texture_2d_array<f32>;
@group(2) @binding(2) var mask_middle: texture_2d_array<f32>;
@group(2) @binding(3) var mask_top: texture_2d_array<f32>;
@group(2) @binding(4) var mask_sampler: sampler;

const MASK_TIER_MIDDLE: u32 = 1u;
const MASK_TIER_TOP: u32 = 2u;

// The level of a material's tier to test, given how far its texture coordinates move per pixel,
// squared. A whole level, so that each one's kept coverage is what the test sees: blending two
// levels would blend away what each kept.
fn mask_level(tier: u32, uv_per_pixel_squared: f32) -> f32 {
    var edge: f32;
    var levels: u32;
    switch tier {
        case MASK_TIER_TOP: {
            edge = f32(textureDimensions(mask_top).x);
            levels = textureNumLevels(mask_top);
        }
        case MASK_TIER_MIDDLE: {
            edge = f32(textureDimensions(mask_middle).x);
            levels = textureNumLevels(mask_middle);
        }
        default: {
            edge = f32(textureDimensions(mask_base).x);
            levels = textureNumLevels(mask_base);
        }
    }
    let lod = 0.5 * log2(max(uv_per_pixel_squared * edge * edge, 1.0e-12));
    return clamp(floor(lod + 0.5), 0.0, f32(levels - 1u));
}

// Whether a material's surface is solid at `uv`, at a level from `mask_level`.
fn mask_solid(tier: u32, slot: u32, uv: vec2f, level: f32) -> bool {
    var alpha: f32;
    switch tier {
        case MASK_TIER_TOP: {
            alpha = textureSampleLevel(mask_top, mask_sampler, uv, i32(slot), level).a;
        }
        case MASK_TIER_MIDDLE: {
            alpha = textureSampleLevel(mask_middle, mask_sampler, uv, i32(slot), level).a;
        }
        default: {
            alpha = textureSampleLevel(mask_base, mask_sampler, uv, i32(slot), level).a;
        }
    }
    return alpha >= 0.5;
}
