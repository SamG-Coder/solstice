// Bindings shared by the software and hardware rasterizers.

#include "common/math.wgsl"
#include "common/scene.wgsl"
#include "vg/pass_view.wgsl"

@group(0) @binding(0) var<uniform> view: PassView;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
// The page pool, as words: a page's vertices, four words each, and then its triangles. One
// binding rather than a view of each keeps the masked rasterizers within 8 storage buffers.
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(4) var<storage, read> visible: array<VisibleCluster>;
@group(0) @binding(5) var<storage, read> state: CullStateRead;

fn load_vertex(index: u32) -> PackedVertex {
    let word = index * 4u;
    return PackedVertex(pool[word], pool[word + 1u], pool[word + 2u], pool[word + 3u]);
}

// Visibility ID: visible-cluster index in the high bits, triangle in the low 7.
fn encode_id(visible_index: u32, triangle: u32) -> u32 {
    return (visible_index << 7u) | triangle;
}

fn local_vertex(packed_triangle: u32, corner: u32) -> u32 {
    return (packed_triangle >> (corner * 8u)) & 0xffu;
}
