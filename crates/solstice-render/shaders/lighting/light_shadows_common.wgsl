// Each local light's shadow map: how far its light reaches in every direction, traced through
// the GI volume once a frame by lighting/light_shadows.wgsl. A pass lighting a point then looks
// its shadow up, one read per light, instead of tracing the volume once per pixel per light.
//
// Needs common/math.wgsl, and, declared by whatever includes this:
//   var light_shadows: texture_2d_array<f32>;

#include "common/math.wgsl"

// Texels along each side of a light's octahedral map. Twin of `LIGHT_SHADOW_SIZE` in lighting.rs.
const LIGHT_SHADOW_SIZE: i32 = 64;

// A texel coordinate past the map's edge, folded back the way the octahedron wraps: the far
// side of each edge is the mirror of the near side.
fn light_shadow_texel(texel: vec2i) -> vec2i {
    let last = LIGHT_SHADOW_SIZE - 1;
    var t = texel;
    if t.x < 0 || t.x > last {
        t = vec2i(select(2 * last + 1 - t.x, -1 - t.x, t.x < 0), last - t.y);
    }
    if t.y < 0 || t.y > last {
        t = vec2i(last - t.x, select(2 * last + 1 - t.y, -1 - t.y, t.y < 0));
    }
    return clamp(t, vec2i(0), vec2i(last));
}

// How much of light `index` reaches the point `from_light` away from it, `distance` off, on a
// surface the volume resolves with voxels `voxel` across: 1 in full view, 0 behind something,
// in between across a texel of the map. The point's own surface stands in the volume, and the
// trace stops most of a voxel short of any surface, so the light must fall two voxels short
// of the point before the point counts as shadowed.
fn light_shadow(index: u32, from_light: vec3f, distance: f32, voxel: f32) -> f32 {
    let p = (oct_encode(from_light / distance) * 0.5 + 0.5) * f32(LIGHT_SHADOW_SIZE) - 0.5;
    let base = vec2i(floor(p));
    let f = p - vec2f(base);
    let reach = distance - voxel * 2.0;
    var lit = 0.0;
    for (var i = 0; i < 4; i++) {
        let corner = vec2i(i % 2, i / 2);
        let weight = select(1.0 - f.x, f.x, corner.x == 1) * select(1.0 - f.y, f.y, corner.y == 1);
        let far = textureLoad(light_shadows, light_shadow_texel(base + corner), i32(index), 0).r;
        lit += weight * select(0.0, 1.0, far >= reach);
    }
    return lit;
}
