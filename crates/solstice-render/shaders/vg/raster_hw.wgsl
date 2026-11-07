// Hardware rasterization of clusters by vertex pulling.
//
// WebGPU has no mesh shaders, so a cluster is one instance of a draw with enough
// vertices for the largest cluster. Vertices past the cluster's real triangle count
// collapse to a degenerate point, which the rasterizer discards cheaply.

#include "vg/raster_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) id: u32,
}

fn cluster_vertex(vertex_index: u32, instance_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(0.0);
    out.id = 0u;

    let visible_index = arrayLength(&visible) - 1u - (state.hw_base + instance_index);
    let entry = visible[visible_index];
    let cluster = clusters[entry.cluster];
    let triangle = vertex_index / 3u;
    if triangle >= cluster_triangle_count(cluster) {
        return out;
    }

    let instance = instances[entry.instance];
    let packed_triangle = triangles[cluster.triangle_offset + triangle];
    let local = local_vertex(packed_triangle, vertex_index % 3u);
    let vertex = vertices[cluster.vertex_offset + local];
    let world = transform_point(instance, decode_position(instance, vertex));

    out.position = view.raster_view_proj * vec4f(world, 1.0);
    out.id = encode_id(visible_index, triangle);
    return out;
}

@vertex
fn vs_visibility(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VertexOutput {
    return cluster_vertex(v, i);
}

// Depth-only entry point, used by shadow maps.
@vertex
fn vs_depth(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> @builtin(position) vec4f {
    return cluster_vertex(v, i).position;
}
