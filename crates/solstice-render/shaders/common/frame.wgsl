// Per-frame constants for the main view. Twin of `FrameUniform` in frame.rs.

struct Frame {
    view_proj: mat4x4f,           // Includes the temporal jitter.
    inv_view_proj: mat4x4f,       // Inverse of the jittered matrix.
    unjittered_view_proj: mat4x4f,
    prev_view_proj: mat4x4f,      // Previous frame, without jitter.
    view: mat4x4f,
    camera_position: vec3f,
    frame_index: u32,
    viewport: vec2f,
    inv_viewport: vec2f,
    jitter_ndc: vec2f,
    near: f32,
    debug_mode: u32,
    sun_direction: vec3f,         // Points towards the sun.
    sun_illuminance: f32,
    sun_color: vec3f,
    sky_intensity: f32,
    light_count: u32,
    time: f32,
    gi_intensity: f32,
    // Zero makes the final image use only the current frame, with no history.
    temporal_accumulation: u32,
}

const DEBUG_NONE: u32 = 0u;
const DEBUG_CLUSTERS: u32 = 1u;
const DEBUG_TRIANGLES: u32 = 2u;
const DEBUG_RASTER_MODE: u32 = 3u;
const DEBUG_INSTANCES: u32 = 4u;
const DEBUG_ALBEDO: u32 = 5u;
const DEBUG_NORMALS: u32 = 6u;
const DEBUG_DIFFUSE_GI: u32 = 7u;
const DEBUG_REFLECTIONS: u32 = 8u;
const DEBUG_VOXELS: u32 = 9u;

fn pixel_to_ndc(pixel: vec2f, inv_viewport: vec2f) -> vec2f {
    let uv = pixel * inv_viewport;
    return vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// World position from a pixel centre and its reversed-Z depth.
fn reconstruct_world(frame: Frame, pixel: vec2f, depth: f32) -> vec3f {
    let ndc = pixel_to_ndc(pixel, frame.inv_viewport);
    let h = frame.inv_view_proj * vec4f(ndc, depth, 1.0);
    return h.xyz / h.w;
}
