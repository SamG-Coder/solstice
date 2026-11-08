// The colour of a screen-trace hit, from last frame's final image.
//
// The including shader declares, in group 0:
//   motion_tex: texture_2d<f32>      G-buffer motion, to find a hit in last frame's image
//   scene_history: texture_2d<f32>   last frame's final colour

// Last frame's lit colour at a hit, found through the hit pixel's motion so that it stays
// put while the camera moves.
fn screen_hit_radiance(uv: vec2f) -> vec3f {
    let pixel = min(vec2i(uv * frame.viewport), vec2i(frame.viewport) - vec2i(1));
    let previous = uv + textureLoad(motion_tex, pixel, 0).xy;
    return textureSampleLevel(scene_history, clamp_sampler, clamp(previous, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
}
