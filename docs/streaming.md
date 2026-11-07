# On-demand streaming: design

Status: proposed, awaiting approval. Nothing here is built yet.

## What the data says

Measured from the current browser cove (`cove_web.solstice`, file v3, textures at 1024):

| Part | Raw | Compressed |
|---|---|---|
| Cluster records (125,354 at 48 bytes) | 6.0 MB | 3.3 MB |
| Hierarchy nodes (11,562) | 0.6 MB | 0.3 MB |
| Vertices and triangles | 192 MB | about 127 MB (ratio 0.66, sampled pages) |
| Root clusters only (50,309 triangles) | 1.1 MB | about 0.7 MB |
| Textures, all mips | 96 MB | about 92 MB (BC barely compresses) |
| Textures, 128 and smaller | 1.5 MB | 1.4 MB |

- **The whole file** gzips to 209 MB today. (163 MB was the older 512 uncompressed bake.)
- **Sibling sets** (clusters with the same parent group): 8,622, averaging 14.5 clusters and 22 KB, largest 66 KB.
- **The hole-free rule has what it needs.** A scan of the file found 71,223 parent-to-child links, none
  missing and no duplicates, so every coarser cluster matches exactly one sibling set below it.
- **One view needs little.** From the cove's opening camera at 1 px error, 1600x900, with no occlusion
  culling and parent pages included, a simulated cut needs 230 pages, or 29 MiB of a 212 MiB total. A
  cut for every direction at once needs 62 MiB.
- **GitHub Pages serves Range requests.** A `GET` with `Range` on the live site returned
  `206 Partial Content` with a stable `ETag`.

## The hole-free rule

Call a sibling set `S(g)` and the clusters simplified from it `P(g)`. They cover the same surface with the
same locked border, so either may be drawn. The renderer draws a valid cut as long as it swaps whole sets, and
only swaps `P(g)` for `S(g)` once everything `P(g)` depends on has itself been swapped.

Today a cluster is drawn when its parent is too coarse and it is itself fine enough. This changes to:

```
draw C  =  parent_too_coarse(C) and (fine_enough(C) or children_not_installed(C))
```

That requires one invariant, enforced on the CPU: **a page installs only after every page holding the
parents of its clusters, and leaves only after every page that depends on it.** A cluster's parent set is
then always resident when it is, so a parent keeps drawing until its whole child set has arrived. Cracks can't
appear because a set's border is locked and all clusters share a mesh-wide position grid.

The GPU needs two extra bits, both in existing fields, so no struct grows:

- **Leaf node `is_leaf`:** leaf flag, installed flag and page id. A leaf whose set is not installed
  requests its page and stops.
- **Cluster `counts`:** a "children not installed" flag in a spare bit.

## Pages

- **Fixed 128 KiB of vertices and triangles.** The largest sibling set fits with room to spare, and it's
  a sensible size for one HTTP request.
- **A sibling set is never split**, so installing a page is all-or-nothing per set.
- **Each page holds sets from one build round.** A set's parents always come from a later round, so page
  dependencies cannot form a cycle.
- **Sets are ordered spatially within a round**, so one page tends to serve one area.
- **Small meshes share pages.** A page isn't tied to a mesh, since offsets are absolute. All root sets go
  together in pinned pages rather than one 128 KiB slot per mesh, which would waste about 10 MB.
- **Cluster records and hierarchy nodes stay resident**, about 7 MB here. This is much simpler than Nanite's
  scheme of keeping them in pages with fix-ups, because activating a page only rewrites records already in
  place. The cost is 48 bytes per cluster, or roughly 0.7 GB for a billion-triangle scene. This repo's
  scenes are far below that, and moving the records into pages later stays possible.

## GPU page pool and eviction

- **One storage buffer of fixed slots**, sized up front, so the budget cannot be exceeded. It's bound to
  the existing `vertices` and `triangles` bindings, which share it. Each page lays out vertices then triangles.
- **No extra indirection in shaders.** Installing rewrites `vertex_offset` and `triangle_offset` in the
  cluster records, so raster, resolve and voxelize shaders don't change.
- **Atomic with respect to frames.** All writes use `queue.write_buffer` between frames, and WebGPU orders them
  against submissions, so a frame never sees half a page.

Requests are one array of `u32` per page, cleared each frame:

- Traversal does `atomicMax(requests[page], priority)` at each leaf that passes the LOD test, where priority
  is projected error. Occluded leaves write the minimum value, which means "keep, but don't load".
- All views write to it, with shadow and GI views weighted lower.
- It can't overflow and needs no deduplication. It's about 8 KB to read back, using the same non-blocking
  pattern as `CullStats`.

Each frame the CPU, with no GPU code involved so it can be unit tested:

1. Adds the parents of requested pages, and sorts by priority with coarser pages first.
2. Issues up to a set number of fetches.
3. Installs arrived pages whose parents are in, up to about 1 MB per frame to avoid hitches.
4. When out of slots, evicts the least recently used page that has no installed dependants, is not pinned, and
   was not touched in the last 30 frames. If nothing qualifies, the load waits. Quality drops but nothing breaks,
   and a counter reports it.

**Shadows and GI.** Both are cached today, so they would keep the root-level shape forever. Page installs
will mark them stale, rate-limited and using the existing cascade turn-taking. GI levels re-voxelize one at a time.

## Textures

WebGPU has no sparse textures, and all layers of an array must share a size and mip count. So one array with
missing top mips is not possible, and creating it costs the full memory anyway. Instead, a material's layer
maps to a tier and slot:

| Tier | Edge | Layers | Lifetime |
|---|---|---|---|
| Base | 128 | All | In the initial download, never evicted |
| Middle | 512 | All | Fetched by demand, never evicted |
| Top | Largest bake size that gives at least 6 slots | As many as the budget allows | Least recently used |

- **Memory is fixed at startup**, as all arrays are allocated up front. At 1024 and 64 MiB that's 1.5 MB +
  24 MB + 8 slots of 4.2 MB. A 2048 bake gets 1024 in the browser on that budget, and full 2048 natively.
- **Promotion** copies lower mips between arrays with `copy_texture_to_texture`, fetches only the missing top
  mips, then rewrites the material. **Demotion** only rewrites the material.
- **Bindings:** resolve samples nine textures rather than three, which is under the limit of 16. Voxelize stays
  on the base tier, since 20 cm voxels gain nothing from more.
- **Demand comes from the cull pass, not resolve.** Resolve is already at the guaranteed limit of 8 storage
  buffers. Each cluster stores a quantised UV density in `counts`, each instance its layer and UV scale, and
  `try_emit` works out the mip wanted. Results go in the request array after the page entries. Occlusion
  is included, with no new binding.
- **GPUs without BC** keep the existing path: unpack on arrival, tiers capped at 512.

## Fetching: HTTP Range, in one file

| | One file with Range | One file per page |
|---|---|---|
| Works on GitHub Pages | Yes, tested | Yes |
| Files to deploy | 1 | About 1,700 pages plus texture mips |
| Merging adjacent pages into one request | Yes | No |
| Atomic update of a scene | Check `ETag` on each response | Files can mix mid-deploy |
| Works with `python -m http.server` | No, it ignores Range | Yes |

**Recommendation: Range.** If a server answers `200` to a Range request, the page downloads the whole file once
and serves ranges from memory, so nothing breaks. `scripts/serve.py` replaces the plain server in
`launch.json`, adding Range support and a `--throttle-mbps` option for measuring.

File v4:

```
header    magic, version, sizes and offsets
manifest  deflate: meshes, nodes, clusters, page table, set links, materials, instances,
          lights, viewpoint, root pages, texture mips of 128 and below
pages     each deflated on its own
textures  raw, by layer and mip
```

The whole-file `.gz` goes away, as pages and manifest are already compressed. Rust never fetches in the
browser: JS calls `viewer.page_requests()`, fetches, unpacks with `DecompressionStream`, and calls
`viewer.deliver_page(id, bytes)`. Natively, a background thread reads and inflates. Failed fetches retry with
back-off while rendering continues.

## Stages

Each stage keeps rendering correct and passes fmt, clippy and tests.

0. **Baseline.** Measure download, time to first image and GPU memory on current `main`, with a throttled server.
1. **Format.** Sets, rounds, page assignment and file v4. Loading still installs everything. Rebake; renders
   should match today's.
2. **Residency-aware culling.** Pool, flags and the new draw rule, still fully resident. Tests that random
   valid residency sets, from random viewpoints, are watertight. A GPU test with only roots installed
   and `temporal_accumulation: false` checks for holes.
3. **Streaming, native.** Requests, residency manager, eviction, budget, shadow and GI invalidation. Tests
   for page selection, parents-first install, children-first eviction, pinning, the budget and a randomised
   stress run checking the cut after every step.
4. **Browser.** Range fetching, whole-file fallback, HUD counters, `serve.py`, `build-web.sh`.
5. **Textures.** Tiers, demand and promotion.
6. **Report.** Before and after numbers, README, Status section.

## Expected results, to be replaced by measurements

| | Today | Target |
|---|---|---|
| Initial download, browser cove | 209 MB | About 6 MB |
| Time to first image at 50 Mbit/s | About 35 s | About 1 to 2 s, then detail fills in |
| Scene GPU memory | 294 MB | Fixed: 128 MiB geometry + 64 MiB textures + 7 MB records |
| Whole viewer GPU memory, browser | 1,133 MiB | About 1,040 MiB |

## Known limits

- **Whole-viewer memory falls only modestly.** The scene is about a quarter of it: the rest is render targets,
  cull lists and the GI volume, which this work doesn't touch. The memory gain is larger on bigger bakes,
  because the budget stays fixed as the file grows.
- **Detail pops in with no cross-fade**, softened only by temporal accumulation.
- **A fast flight outruns the network.** The result is coarser geometry until pages arrive, not holes.
- **The live demo stays procedural for now.** CI bakes without scans, so putting the cove on Pages is separate work.
