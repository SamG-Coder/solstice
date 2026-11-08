// The particle system's bookkeeping, a workgroup each: before the frame's spawn and update, what
// each emitter does and where its workgroups start (`plan`); after them, how many of each
// emitter's particles to draw (`finish`). One small pass for every emitter replaces a dispatch
// per emitter, which matters on the web, where each dispatch costs the browser CPU time.

#include "particles/common.wgsl"

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<storage, read> emitters: array<Emitter, MAX_EMITTERS>;
@group(0) @binding(2) var<storage, read_write> state: array<EmitterState, MAX_EMITTERS>;
// Dispatch arguments for spawn, then for update, then from word 8 a draw's arguments per
// emitter: vertex count, instance count, first vertex, first instance.
@group(0) @binding(3) var<storage, read_write> indirect: array<u32>;

const DRAWS_FIRST: u32 = 8u;

var<workgroup> spawn_groups: array<u32, MAX_EMITTERS>;
var<workgroup> update_groups: array<u32, MAX_EMITTERS>;

@compute @workgroup_size(64)
fn plan(@builtin(local_invocation_index) e: u32) {
    let read = sim.parity;
    let written = 1u - read;
    let emitter = emitters[e];
    var spawn = 0u;
    var update = 0u;
    if (emitter.flags & FLAG_LIVE) != 0u {
        // New particles take slots from the top of the free list, as many as it holds.
        let free = atomicLoad(&state[e].free_count);
        spawn = min(emitter.spawn_request, free);
        state[e].dropped += emitter.spawn_request - spawn;
        state[e].spawn_first = free - spawn;
        atomicStore(&state[e].free_count, free - spawn);
        state[e].spawn_index = state[e].spawned;
        state[e].spawned += spawn;
        update = atomicLoad(&state[e].alive[read]);
    }
    state[e].spawn_count = spawn;
    state[e].update_count = update;
    atomicStore(&state[e].alive[written], 0u);
    spawn_groups[e] = (spawn + GROUP_SIZE - 1u) / GROUP_SIZE;
    update_groups[e] = (update + GROUP_SIZE - 1u) / GROUP_SIZE;
    workgroupBarrier();

    if e == 0u {
        var spawn_total = 0u;
        var update_total = 0u;
        for (var i = 0u; i < MAX_EMITTERS; i++) {
            state[i].spawn_start = spawn_total;
            state[i].update_start = update_total;
            spawn_total += spawn_groups[i];
            update_total += update_groups[i];
        }
        indirect[0] = spawn_total;
        indirect[1] = 1u;
        indirect[2] = 1u;
        indirect[3] = update_total;
        indirect[4] = 1u;
        indirect[5] = 1u;
    }
}

@compute @workgroup_size(64)
fn finish(@builtin(local_invocation_index) e: u32) {
    let written = 1u - sim.parity;
    var count = 0u;
    if (emitters[e].flags & FLAG_LIVE) != 0u {
        count = atomicLoad(&state[e].alive[written]);
    }
    let at = DRAWS_FIRST + e * 4u;
    indirect[at] = 6u;
    indirect[at + 1u] = count;
    indirect[at + 2u] = 0u;
    indirect[at + 3u] = 0u;
}
