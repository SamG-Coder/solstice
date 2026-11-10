# On-demand streaming

This is the design as built. [Results](#results) are at the end.

## What the data said

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
   was not touched in the last 30 frames.
5. When that frees too little, a page left without a slot takes the place of the least wanted page still in,
   while it is wanted twice as much: most wanted first, against least wanted first, and of those the
   finest. So a full pool comes to hold what is wanted most, whatever came first. What a view only fetched
   ahead of need, or kept though hidden, makes way for what is seen, and never outbids anything itself. The
   factor of two keeps two pages wanted about as much from changing places for ever. Views ask by turns,
   the shadow maps and the voxels only when they are redrawn, so a page counts as wanted as much as it was
   asked for at most over the last 120 to 240 frames, and as much as a shadow map or the voxels asked for
   it over their last 3600 frames of redrawing: by its last asking alone it would lose its slot at each
   turn and take another's at the next. A page a shadow map or the voxels asked for also stands until
   views of its kind have been redrawn for 240 frames without asking for it again, however long that
   takes. Counted by the clock it left while its view stood still, the page that took its slot redrew the
   view, and the view asked for it back: thousands of pages went round like that for as long as one
   looked. A kind of view that asks for nothing for 3600 frames counts as gone, and its pages as anyone's.
6. If nothing qualifies, the load waits. Quality drops but nothing breaks, and a counter reports it.

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

Each stage kept rendering correct and passed fmt, clippy and tests. Textures came before the
browser in the end, as the page needed them to look right.

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

## Results

The browser cove at 1024, on an RTX 5080, in a 1024x768 canvas, with `scripts/serve.py` limiting
all traffic to 50 Mbit/s with 30 ms latency. Whole-viewer memory is `nvidia-smi` total used minus
idle, with both versions measured back to back. Idle drifted by about 70 MiB, so treat those two
as approximate.

| | Before | After |
|---|---|---|
| Download before the first image | 209 MB | **6.0 MB** |
| First image | 36.5 s | **1.4 s** |
| The opening view at full detail | At the first image | About 17 s, 61 MB in all |
| Scene data on the GPU | 280 MiB | **195 MiB, fixed**: 128 geometry, 60 textures, 6 records |
| Whole viewer on the GPU, browser | 855 MiB | **668 MiB** |
| On the server | 209 MB | 226 MB |

Scene data accounts for about 85 MiB of the 187 MiB fall in whole-viewer memory. The rest wasn't
traced. From the same camera, a settled streamed image differs from a whole load by 44 to 48 dB,
no more than two whole loads differ from each other (45 dB), using 396 of 1,640 pages.

The 2048 bake is 515 MB on the server, and starts the same way: 6.0 MB and 1.4 s to the first image.
With the budget cut to 24 MiB of geometry and 16 MiB of textures, flying between three viewpoints
evicted 46 pages with no GPU errors and stayed within the pool. Parts of the view were coarser
and textures softer, as intended.

A server without Range support, such as `python -m http.server`, also works: the page downloads
the file once, in that test 226 MB, and streams from memory.

**Not measured:** flyover frame times, real networks, and GitHub Pages serving a `.solstice` file.
Range was tested on Pages with a `.gz` file only. The on-screen frame rate looked the same as before
in the browser, at 102 to 110 fps, but with no controlled comparison.

## Known limits

- **Most of the viewer's memory isn't the scene.** The rest is render targets, cull lists and the GI
  volume, which this work doesn't touch.
- **A pool that's too small is coarse somewhere.** When one view wants more than the pool holds, the pages
  wanted least stay out, and pages wanted within twice of each other stay as they came. `geometry-mb`
  on the page, and `--geometry-budget-mb` natively, give the pool more room.
- **Detail pops in with no cross-fade**, softened only by temporal accumulation.
- **A fast flight outruns the network.** The result is coarser geometry until pages arrive, not holes.
- **The live demo stays procedural for now.** CI bakes without scans, so putting the cove on Pages is separate work.
