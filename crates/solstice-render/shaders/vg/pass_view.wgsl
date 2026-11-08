// One view through the culling and rasterization passes: the main camera in either
// occlusion phase, a shadow cascade, or a voxelization volume. Twin of `PassView`
// in vg/culling.rs.

struct PassView {
    raster_view_proj: mat4x4f,
    occlusion_view: mat4x4f,
    planes: array<vec4f, 6>,
    camera_position: vec3f,
    // Largest allowed projected error, in pixels.
    lod_threshold: f32,
    viewport: vec2f,
    // Projection scale of the view used for the occlusion test.
    proj_scale: vec2f,
    hzb_size: vec2f,
    near: f32,
    // Pixels per world unit: at unit distance if perspective, everywhere if not.
    lod_scale: f32,
    is_ortho: u32,
    // 0 tests the previous frame's depth; 1 tests this frame's.
    phase: u32,
    occlusion_enabled: u32,
    // 0: hardware only. 1: small clusters go to software. 2: every cluster does.
    allow_software: u32,
    // For an orthographic view drawn for a camera: that camera, its pixels per world unit at
    // unit distance (0 for none), and the nearest it sees anything drawn here.
    eye_position: vec3f,
    eye_lod_scale: f32,
    eye_near: f32,
    // Further planes a cluster must lie inside, and how many of them are in use: for a shadow
    // view, the bounds of what it shadows, open towards the light.
    receiver_plane_count: u32,
    receiver_planes: array<vec4f, 6>,
    // Towards the light, for a shadow view.
    eye_light: vec3f,
}

// Written by the cull passes. Atomic view of `CullStateRead`.
struct CullState {
    read_begin: u32,
    read_end: u32,
    write_head: atomic<u32>,
    retry_count: atomic<u32>,
    sw_count: atomic<u32>,
    hw_count: atomic<u32>,
    sw_base: u32,
    hw_base: u32,
    triangle_count: atomic<u32>,
    // Queue entries and clusters that found no room in their lists, and so were lost. Nonzero
    // means the lists are too small for this view, and the image or shadow has holes.
    dropped: atomic<u32>,
    // Entries of the visible list claimed so far, from both ends together.
    reserved: atomic<u32>,
    _pad2: u32,
}

// Read by the raster passes, after culling has finished.
struct CullStateRead {
    read_begin: u32,
    read_end: u32,
    write_head: u32,
    retry_count: u32,
    sw_count: u32,
    hw_count: u32,
    sw_base: u32,
    hw_base: u32,
    triangle_count: u32,
    dropped: u32,
    reserved: u32,
    _pad2: u32,
}
