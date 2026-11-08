// Updates the water-line map (ocean/waterline.rs) for one frame: where the water stands now at
// each spot, from the waves shrunk over the seabed exactly as the drawn surface is, and the
// highest it has stood there lately, which falls back towards it as the spot dries.

#include "ocean/surface_params.wgsl"
#include "ocean/waterline_params.wgsl"

@group(0) @binding(0) var<uniform> surface: SurfaceParams;
@group(0) @binding(1) var<uniform> waterline: WaterlineParams;
@group(0) @binding(2) var displacement: texture_2d_array<f32>;
@group(0) @binding(3) var wave_sampler: sampler;
@group(0) @binding(4) var seabed_map: texture_depth_2d;
// Last frame's map, and this frame's.
@group(0) @binding(5) var previous: texture_2d<f32>;
@group(0) @binding(6) var current: texture_storage_2d<rg32float, write>;

#include "ocean/seabed.wgsl"

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let n = u32(waterline.grid.y);
    if any(id.xy >= vec2u(n)) {
        return;
    }
    // The world texel this map texel holds within this frame's window.
    let first = waterline.window.xy;
    let nf = waterline.grid.y;
    let texel = first + (vec2f(id.xy) - first - floor((vec2f(id.xy) - first) / nf) * nf);
    let xz = (texel + vec2f(0.5)) * waterline.grid.x;

    // The water's height here now, as the surface is drawn: every cascade at the level whose
    // texels match the map's, shrunk over the seabed.
    var offset = vec3f(0.0);
    for (var cascade = 0u; cascade < 3u; cascade++) {
        let tile = surface.sizes[cascade];
        let level = max(log2(waterline.grid.x / (tile / surface.grid.x)), 0.0);
        offset += textureSampleLevel(displacement, wave_sampler, xz / tile, cascade, level).xyz;
    }
    // As a height in the world, not above the calm sea: when the sea falls, as a tide going
    // out, the line it reached stays behind and dries.
    let now = surface.sizes.w + shoal(offset, water_depth_at(xz)).y;

    // The highest it has stood here lately, if the map held this spot last frame; otherwise
    // the spot has only just come into view, and all that is known is where the water is now.
    var high = now;
    if waterline.grid.w > 0.5 && in_waterline_window(texel, waterline.window.zw) {
        let remembered = textureLoad(previous, vec2i(id.xy), 0).r;
        high = max(now, now + (remembered - now) * waterline.grid.z);
    }
    textureStore(current, vec2i(id.xy), vec4f(high, now, 0.0, 0.0));
}
