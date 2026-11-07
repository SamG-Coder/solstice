// Scene data as laid out in GPU buffers. Twin of solstice-asset's `gpu` module and
// solstice-render's `scene` module; change them together.

struct Cluster {
    center: vec3f,
    radius: f32,
    lod_center: vec3f,
    lod_radius: f32,
    lod_error: f32,
    vertex_offset: u32,
    triangle_offset: u32,
    counts: u32, // vertex_count | triangle_count << 8 | CLUSTER_CHILDREN_MISSING
}

struct Node {
    center: vec3f,
    radius: f32,
    lod_center: vec3f,
    lod_radius: f32,
    max_parent_error: f32,
    first: u32,
    count: u32,
    is_leaf: u32, // 0 for an interior node. See the NODE_ constants in vg/cull.wgsl.
}

// x y | z nx | ny u | v _
struct PackedVertex {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

struct Instance {
    // Affine transforms stored as the top three rows of the matrix.
    world_0: vec4f,
    world_1: vec4f,
    world_2: vec4f,
    prev_world_0: vec4f,
    prev_world_1: vec4f,
    prev_world_2: vec4f,
    normal_0: vec4f,
    normal_1: vec4f,
    normal_2: vec4f,
    quant_origin: vec3f,
    quant_step: f32,
    uv_origin: vec2f,
    uv_step: vec2f,
    root_node: u32,
    material: u32,
    max_scale: f32,
    _pad: u32,
}

struct Material {
    base_color: vec4f,
    emissive: vec4f,
    roughness: f32,
    metallic: f32,
    uv_scale: f32,
    normal_strength: f32,
    texture_layer: i32,
    macro_variation: f32,
    _pad0: u32,
    _pad1: u32,
}

struct Light {
    position: vec3f,
    range: f32,
    intensity: vec3f,
    _pad: u32,
}

// One selected cluster. The index of this entry is what the visibility buffer stores.
struct VisibleCluster {
    instance: u32,
    cluster: u32,
}

fn cluster_triangle_count(c: Cluster) -> u32 {
    return (c.counts >> 8u) & 0xffu;
}

fn cluster_vertex_count(c: Cluster) -> u32 {
    return c.counts & 0xffu;
}

fn transform_point(i: Instance, p: vec3f) -> vec3f {
    let h = vec4f(p, 1.0);
    return vec3f(dot(i.world_0, h), dot(i.world_1, h), dot(i.world_2, h));
}

fn transform_point_prev(i: Instance, p: vec3f) -> vec3f {
    let h = vec4f(p, 1.0);
    return vec3f(dot(i.prev_world_0, h), dot(i.prev_world_1, h), dot(i.prev_world_2, h));
}

fn transform_normal(i: Instance, n: vec3f) -> vec3f {
    let h = vec4f(n, 0.0);
    return normalize(vec3f(dot(i.normal_0, h), dot(i.normal_1, h), dot(i.normal_2, h)));
}

fn decode_position(i: Instance, v: PackedVertex) -> vec3f {
    let grid = vec3f(f32(v.a & 0xffffu), f32(v.a >> 16u), f32(v.b & 0xffffu));
    return i.quant_origin + grid * i.quant_step;
}

fn sign_extend_16(x: u32) -> f32 {
    return f32(i32(x << 16u) >> 16u);
}

fn decode_normal(v: PackedVertex) -> vec3f {
    let e = vec2f(sign_extend_16(v.b >> 16u), sign_extend_16(v.c & 0xffffu)) / 32767.0;
    return oct_decode(e);
}

fn decode_uv(i: Instance, v: PackedVertex) -> vec2f {
    return i.uv_origin + vec2f(f32(v.c >> 16u), f32(v.d & 0xffffu)) * i.uv_step;
}
