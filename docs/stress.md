# The stress scene

`scenes/stress.txt` is a scene built to find the renderer's limits: everything the engine does,
all at once, and more of it each round. When a round exposes a flaw or a slowdown, the cause is
fixed in the engine, never by lowering settings or taking content out, and the fix goes out as
its own small pull request. This is the log of those rounds.

## How it is measured

All timings are on an RTX 5080, native (Vulkan), at 1280×720 unless stated, with
`viewer assets/stress.solstice --capture OUT.png --time HH:MM --weather W --frames 120`. The GPU
times are those of the last frame, by pass, so a still camera reports no shadow cost: cascades
only redraw when their view changes. Moving captures add `--move 0.02`, which slides the camera
2 cm a frame. GPU contention from other work on the machine skews timings, so each number is the
better of two runs, and the first run after a build is discarded.

Scene file lines added for this scene, all in `apps/bake/src/scene_file.rs` and `flora.rs`:

- `hills HEIGHT WAVELENGTH FLAT_RADIUS` adds relief to the terrain, level around the origin and
  fading at the coast;
- `scatter MODEL COUNT x0 z0 x1 z1 HEIGHT MIN_SCALE MAX_SCALE SEED` throws copies of a model over
  a rectangle, seated on the terrain if `HEIGHT` is `ground` or `ground:FRACTION`;
- `trees SPECIES COUNT x0 z0 x1 z1 SPACING SEED` grows a species over dry land, no two closer than
  the spacing;
- `grass BREED x0 z0 x1 z1` tiles dry land with patches of grass, sheared to the slope;
- `lights COUNT x0 z0 x1 z1 HEIGHT RANGE R G B SEED` hangs point lights above the ground.

The island's radius now follows the terrain's width (0.3 of it, which is the cove's 270 m
exactly), so a wider terrain has more land rather than more seabed.

## Baseline: the cove

For scale, the cove at its own viewpoint, 10:00, fair weather: 114 instances, 22.0 M instanced
triangles, 348 k drawn. GPU 1.51 ms: screen gi 0.33, cull 1 0.20, raster 1 0.15, water 0.14,
clouds on screen 0.13, post 0.10, gi light 0.09, clouds 0.09.

## Rounds

### Round 1: the island

The cove's fort on a 1.8 km island with 45 m hills, 11,000 trees (oak, birch, fir, bush),
17,300 patches of meadow grass over 300 m, 4,900 scanned rocks and 40 cliffs scattered over the
land and into the surf, 600 barrels, vases and busts, 12 ships at anchor, and 64 torches in the
courtyard. Textures at 1024. **50,363 instances, 7.58 billion instanced triangles, 15.3 M
unique, 474 MB, baked in 4.5 minutes.**

Still camera, 10:00, fair, GPU ms (total, then the passes over 0.1 ms):

| View | Drawn | Total | Passes |
|---|---|---|---|
| Beach (the cove's camera) | 19.7 M tris, 177 k clusters | 2.66 | raster 1 0.84, screen gi 0.38, cull 1 0.32, lighting 0.18, water 0.16, gi light 0.14 |
| Hill, 60 m up, 230 m out | 38.7 M, 346 k | 2.79 | raster 1 1.23, cull 1 0.39, screen gi 0.24, clouds on screen 0.17, lighting 0.14 |
| Overview, 120 m up, 500 m out | 24.2 M, 223 k | 2.43 | raster 1 1.03, cull 1 0.33, screen gi 0.20, clouds on screen 0.17 |
| Rampart, looking over the wood | 5.7 M, 53 k | 2.35 | lighting 0.49, raster 1 0.39, screen gi 0.32, gi light 0.25, cull 1 0.21 |
| Courtyard, among the torches | 4.4 M, 39 k | 2.85 | lighting 0.96, raster 1 0.42, screen gi 0.34, gi light 0.29, cull 1 0.22 |

The 60 s flyover (`scenes/stress_flyover.txt`, 3,570 frames): **avg 4.5 ms, p95 8.8 ms, worst
18.3 ms**, one frame over 16.7 ms. Per pass, average / p95 / worst: shadows 0.96 / 5.19 / 8.95;
raster 1 0.75 / 1.07 / 1.46; cull 1 0.32 / 0.45 / 1.07; screen gi 0.28 / 0.46 / 0.78; lighting
0.19 / 0.55 / 1.15; clouds on screen 0.19 / 0.37 / 0.86. Up to 43 M triangles drawn.

What this round found:

- **Shadow cascade redraws cost 5 to 9 ms each.** A still camera pays nothing; a moving one
  redraws a cascade every few frames, and each redraw culls and draws the whole cascade, so
  the frame time swings between 3 and 12 ms. This is the limit the README already names on
  integrated GPUs; dense foliage makes it bite on the RTX 5080 too.
- **Local lights are brute force.** Every pixel loops over every light, and each light in
  range traces 16 steps through the GI volume for visibility; every GI voxel does the same with
  24 steps. 64 lights took the lighting pass from 0.05 to 0.96 ms in the courtyard.
- **The haze is a constant.** Extinction is 0.0011 per metre whatever the weather, so from a
  hill the wood 300 m away is already a quarter dissolved into the sky, and ships 1 km out
  are ghosts. Clear air has visibility of tens of kilometres.
- **The cloud shadow map ends 6 km out, and beyond it the edge texel is used.** From 80 m up
  the horizon is 32 km away, so the far sea shows the map's edge as a straight line, and its
  edge texels as dark streaks stretched to the horizon (the `ocean` debug view proved they were
  water, not geometry). At ground level the horizon is nearer than the map's edge, which is why
  the cove never showed it.
- Not bugs: the dark islets at the horizon in clear weather are the scattered cliffs standing
  on the seabed, and a camera 10 m up on the hills was inside them (the hills rise 45 m).

#### Fix 1: shadow cascades draw only what can shadow their slice

Measured with a cascade at a time, the far two cost 4.2 and 4.4 ms a redraw, the near two 0.9
and 1.3. Culling was 0.3 ms of that; the rest was the hardware rasterizer drawing what the
cull selected: **465,000 clusters and 50.5 million triangles for the far cascade**, more than
the main view draws, because the cascade is fitted to the bounding sphere of its slice of the
view frustum, which, for the slice from 120 to 420 m, is a square 1,060 m across: most of the
island, including everything beside and behind the camera, at 0.5 m texels throughout.

Two changes in the cull, both exact:

- **Receiver planes.** A caster can only shadow the slice if it lies inside the slice's
  frustum or towards the sun from it: inside every plane of the slice that faces the sun. Those
  planes go to the cull as extra frustum planes. The slice starts where receivers begin to
  dither into this cascade from the one before (85% of the way along it), its near plane is
  drawn back so that points at that *distance* in the frustum's corners are inside (cascades
  are chosen by distance, not depth), and each plane is pushed out by the furthest the shadow
  lookup offsets a receiver along its normal (7.6 texels). Without the last two, trees near the
  cascade's start lost their shadows: the difference image showed them.
- **Error the camera cannot see is not drawn.** A cascade also knows the camera's position and
  pixels per metre, and a cluster is fine enough when its error is under a texel *or* under a
  pixel at its distance from the camera, bounded below by the slice's start. This matters once a
  cascade's texels are finer than the camera's pixels, which for the far cascade is beyond
  325 m at 720p, so it gave 14% here and more at higher resolutions.

| | Far cascade redraw | Flyover: frame avg / p95 / worst | Shadows avg / redraw mean / max |
|---|---|---|---|
| Before | 465 k clusters, 50.5 M tris, 4.2 ms | 4.69 / 9.07 / 14.7 ms | 0.98 / 2.29 / 9.8 ms |
| After | 220 k clusters, 23.5 M tris, 2.25 ms | 3.95 / 5.56 / 8.9 ms | 0.36 / 1.35 / 4.7 ms |

Images are unchanged within noise: with TAA off, before and after agree to 53.8 dB on the hill
view (two runs of the same build agree to 58 dB) and 46.8 dB on the beach (floor 46.5 dB, the
water's reflection jitter). The cove agrees to 48.2 dB (floor 46.5) and the forest to 44.9 dB
(floor 44.9), and the cove's flyover is unchanged at 2.58 ms average, its shadows 0.085 → 0.073
ms. Tests cover the plane selection, the slice bounds and the padding, and each was checked by
breaking the code it tests.

What remains of the shadow cost is the hardware rasterizer drawing clusters of pixel-sized
triangles that the main view would send to the software rasterizer, and a redraw still culls
and draws the whole cascade. Both are candidates for later rounds.

### Round 2: more of everything

Textures at 2048. Trees 11,000 → 27,000 (oaks 7,000, birches 5,000, firs 7,000, bushes 8,000,
spaced closer), grass 17,300 → 48,800 patches (meadow 400 m, a tall-grass field to the west),
rocks 4,900 → 13,200, ships 12 → 40, and 448 lamps over the wood and beach besides the 64
torches. **122,361 instances, 18.8 billion instanced triangles, baked in 5.1 minutes.**

Still camera, 10:00, fair, with fix 1 (the receiver planes, see below):

| View | Total | Passes over 0.2 ms |
|---|---|---|
| Hill | 4.42 | raster 1 1.71, lighting 0.80, cull 1 0.73, screen gi 0.25 |
| Beach | 3.98 | lighting 1.04, raster 1 0.90, cull 1 0.38, screen gi 0.38, gi light 0.27 |
| Courtyard | 4.44 | lighting 1.77, gi light 0.82, raster 1 0.57, screen gi 0.33, cull 1 0.27 |

Flyover: **avg 6.0 ms, p95 10.7 ms, worst 16.3 ms.** Per pass, average / p95 / worst: shadows
0.86 / 5.69 / 9.99 (redraw mean 2.56 ms); raster 1 1.20 / 1.92 / 2.49; lighting 0.88 / 1.61 /
2.71; cull 1 0.47 / 0.65 / 0.90; gi light 0.25 / 0.72 / 1.69; screen gi 0.27 / 0.36 / 0.66;
raster 2 worst 3.22. Up to 94 M triangles drawn, 34.5 M on average.

What this round found:

- **The visible-cluster list overflows.** Each cull writes the clusters it selects into a list
  of 1,048,576 entries and silently drops the rest. Without the receiver planes the far shadow
  cascade selected 1,050,000 to 1,056,000 clusters from every viewpoint tried: the list is
  full, and whatever came after it in traversal order casts no shadow. With the planes the
  far cascade selects 540,000 to 960,000, under the cap, but only just in the courtyard. The
  main view is heading the same way: 94 M triangles is about 900,000 clusters. Nothing reports
  the overflow. That needs a counter at least, and the far cascades need to select far fewer
  clusters than the receivers' grass and leaves currently give them.
- **Local lights scale with how many overlap, not how many exist.** 512 lights over the island
  took the lighting pass to 0.80 ms on the hill and 1.77 ms in the courtyard, and the voxel
  light pass to 0.82 ms there. The per-light work is the visibility trace, 16 steps per pixel
  per light in range, so the cost is set by how many lights reach each pixel: the courtyard's
  64 torches 14 m apart overlap thirty deep. A tile or cluster list would cut the loop over the
  other 450, which costs little; the traces are the content's own cost.
- **Triangles drawn per pixel.** 94 M triangles at 720p is about a hundred per pixel, from
  foliage whose clusters stay fine. The software rasterizer carries it (raster 1 1.2 ms), but
  it is the ceiling for small GPUs.

#### Fix 2: cloud shadows beyond their map (PR #26)

A one-workgroup pass averages the cloud shadow map each frame, reading every fourth texel, and
`cloud_shadow()` blends from the map to that mean across the map's outer band (4.5 to 6 km) and
uses the mean beyond. The streaks and the line on the far sea are gone from 80 m and 300 m up;
a faint lens remains within the map where a cloud's shadow really falls 5 km out. Reviewed and
approved by a separate agent; its nits (a 4× subsample, wording) are in.

#### Fix 3: the cull lists report what they drop, and hold twice as much

`CullState` gains a `dropped` counter: every queue entry, retry entry and cluster that found no
room is counted, and a dropped cluster no longer adds its triangles to the count either.
`FrameStats` reports it for the main view and the last shadow cascade, and the viewer prints a
warning after a capture or a flyover when either is nonzero. Both visible lists grow from
1,048,576 to 2,097,152 entries (8 bytes each, so 16 MB more on the GPU in all): the far cascade
in the round-2 courtyard selects 988,000 clusters with the receiver planes and would have
overflowed within a camera turn. Checked by shrinking the lists to 65,536 for one build: the
courtyard then reports 887,927 clusters dropped in the main view and 920,079 in the cascade.

What this does not fix: a 2048² cascade over a dense wood legitimately asks for a million
clusters, four texels each, and 100 million triangles. That is the content's density, the same
as the main view's per pixel, over four and a half times the pixels. Fewer, larger triangles for
far shadow casters would need a coarser LOD than the texel for foliage, which is a design
question for the cluster builder's thinning, not a cull bug.

