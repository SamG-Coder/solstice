// Reads a skinned vertex's skin. A shader that includes this defines `load_vertex`.

// A skinned cluster's vertices are followed by their skins, two to a record.
fn vertex_skin(cluster: Cluster, local: u32) -> vec2u {
    let record = load_vertex(cluster.vertex_offset + cluster_vertex_count(cluster) + local / 2u);
    if (local & 1u) == 0u {
        return vec2u(record.a, record.b);
    }
    return vec2u(record.c, record.d);
}

// Where a vertex of a cluster goes in the current pose, in the mesh's own space. Rigid instances
// stay as they are.
fn posed_position(instance: Instance, cluster: Cluster, local: u32, p: vec3f) -> vec3f {
    if !instance_skinned(instance) {
        return p;
    }
    return skin_point(skin_rows(instance.bones, vertex_skin(cluster, local), false), p);
}
