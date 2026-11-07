// Single-invocation steps that turn the counters in the cull state into indirect
// arguments. They run between the wider dispatches of cull.wgsl.

#include "vg/pass_view.wgsl"

// Must match the workgroup size of `process`.
const PROCESS_GROUP_SIZE: u32 = 32u;
// The most workgroups one dispatch dimension may hold. Consumers of the software list take
// one workgroup per cluster and stride over the rest when there are more.
const MAX_GROUPS: u32 = 65535u;

struct Args {
    // Dispatch: traversal of the next queue level.
    queue_x: u32,
    queue_y: u32,
    queue_z: u32,
    // Dispatch: software rasterization of the newly visible clusters.
    software_x: u32,
    software_y: u32,
    software_z: u32,
    // Draw: hardware rasterization of the newly visible clusters.
    draw_vertex_count: u32,
    draw_instance_count: u32,
    draw_first_vertex: u32,
    draw_first_instance: u32,
}

const MAX_CLUSTER_TRIANGLES: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> state: CullState;
@group(0) @binding(1) var<storage, read_write> args: Args;

fn groups_for(count: u32, size: u32) -> u32 {
    return (count + size - 1u) / size;
}

// Makes everything appended by the previous level the next range to read.
@compute @workgroup_size(1)
fn advance() {
    state.read_begin = state.read_end;
    state.read_end = atomicLoad(&state.write_head);
    args.queue_x = groups_for(state.read_end - state.read_begin, PROCESS_GROUP_SIZE);
    args.queue_y = 1u;
    args.queue_z = 1u;
}

// Publishes the clusters selected since the last call for rasterization.
@compute @workgroup_size(1)
fn finish() {
    args.software_x = min(atomicLoad(&state.sw_count) - state.sw_base, MAX_GROUPS);
    args.software_y = 1u;
    args.software_z = 1u;
    args.draw_vertex_count = MAX_CLUSTER_TRIANGLES * 3u;
    args.draw_instance_count = atomicLoad(&state.hw_count) - state.hw_base;
    args.draw_first_vertex = 0u;
    args.draw_first_instance = 0u;
}

// Starts occlusion phase 1. The bind group swaps queue and retry buffers, so the
// retry list written during phase 0 is now the queue.
@compute @workgroup_size(1)
fn begin_second_phase() {
    let retried = atomicLoad(&state.retry_count);
    state.read_begin = 0u;
    state.read_end = 0u;
    atomicStore(&state.write_head, retried);
    atomicStore(&state.retry_count, 0u);
    state.sw_base = atomicLoad(&state.sw_count);
    state.hw_base = atomicLoad(&state.hw_count);
}
