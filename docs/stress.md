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

#### Fix 4: lights reach tiles, not every pixel

The lighting pass looped over every light in the scene for every pixel, 512 distance tests a
pixel here, before tracing visibility for the few in range. A compute pass now finds, per
16×16 tile of the screen, the lights whose range meets the tile's frustum (its four edge planes
through the camera, cut to the depth range the tile holds), and lists up to 127 of them; the
lighting pass loops over the tile's list. With no lights in the scene the pass is skipped and
the lists, zero from creation, say so.

| View, round 2 | Lighting before | After |
|---|---|---|
| Hill (512 lights in view, few in range) | 0.74 ms | 0.10 ms |
| Beach | 0.97 ms | 0.12 ms |
| Courtyard, day (64 torches 14 m apart) | 1.72 ms | 0.87 ms |
| Courtyard, 22:00 | 1.69 ms | 1.09 ms |

Images are unchanged within noise (TAA off; hill 58.3 dB, beach 46.8, courtyard 41.8 against
floors of 58, 46.5 and 40.7). What remains in the courtyard is the torches themselves: each pixel
there lies within range of some thirty, and each costs a 16-step visibility trace through the GI
volume. That is the content's cost, and a cheaper visibility (a shadow cache per light, or fewer
steps for near lights) would be the next step if it mattered. The voxel light pass still loops
over every light per voxel (0.8 ms in the courtyard); the same list idea per 8³ block of each
clipmap level would cut its loop, but not its traces.

A GPU test renders a ball under one lamp and six hundred lamps 200 m away, with sun, sky and
indirect light off so that only direct light shows, and checks that the lamp lights the ball and
the far lamps change nothing; the near lamp comes last in the list so that a tile holding the
wrong index would pick a far one. Four mutations fail it: no light listed, planes shrunk by the
range, the depth range shrunk, and the lighting pass reading the list's position as the index.

### Round 3: other sizes, other weather, other hours

The round-2 scene with fixes 1 to 4, at other sizes and under other skies. Still camera unless
said; the last frame's GPU times.

| View | Total | Passes over 0.3 ms |
|---|---|---|
| Hill, 1920×1080, fair 10:00 | 5.9 | raster 1 3.21, cull 1 0.68, screen gi 0.46, clouds on screen 0.27 |
| Hill, 2560×1440 | 10.3 | raster 1 4.80, screen gi 2.19, cull 1 0.94, clouds on screen 0.37, resolve 0.37, lighting 0.34 |
| Courtyard, 2560×1440 | 13.2 | lighting 7.28, screen gi 1.77, raster 1 1.65, gi light 0.81, resolve 0.37, post 0.33 |
| Hill, storm at 18:40 | 3.4 | raster 1 1.71, cull 1 0.47 |
| Beach, storm at 18:40 | 5.7 | screen gi 2.90, raster 1 0.90, raster 2 0.42, cull 1 0.34 |
| Courtyard, clear at 23:00 | 3.3 | lighting 0.85, gi light 0.80, raster 1 0.58, screen gi 0.33 |
| Overview, clear at 23:00 | 3.6 | raster 1 1.81, cull 1 0.53, resolve 0.32 |
| Hill, streamed (128 MB geometry, 64 MB textures) | 6.5 | raster 1 4.15, cull 1 0.49, post 0.37, gi light 0.36 |

Flyovers: **1080p, fair: avg 10.4 ms, p95 21.9, worst 57.7**, 335 of 3,570 frames over 16.7 ms;
shadows 1.18 / 8.05 / 19.4, raster 1 3.00 / 5.63 / 11.5, screen gi 0.63 / 1.93 / 7.8, up to 154 M
triangles. **720p, storm at 18:40: avg 12.3 ms, p95 34.0, worst 66.1**, 723 frames over 16.7 ms;
shadows 4.68 / 26.2 / 60.4, and the far cascade overflowed even the doubled list in one frame
(450,685 clusters dropped).

What this round found:

- **A low sun blows the far cascades up.** At 18:40 the sun is near the horizon, so a cascade's
  orthographic volume, 900 m deep towards the sun, lies along the ground, and the receiver
  planes, open towards the sun, admit the whole wood behind the slice: every tree for 900 m
  casts a long shadow into the view, as it truly does. The far cascade selected 1.24 million
  clusters standing still and more than 2.1 million at some point in the flight. Correct, and
  60 ms. The way out is not to cull casters that really shadow the view but to draw them more
  cheaply: the software rasterizer for the pixel-sized foliage clusters that make up most of
  the list (the main view sends 98% of its clusters there), and for the far cascade a coarser
  LOD than its texel for leaves, which the camera cannot resolve anyway at that distance.
- **The main view itself approaches the list.** 851,000 clusters at 1080p and 1,207,000 at
  1440p, against the new cap of 2,097,152; 4K would pass it. Those are 94 to 132 million
  triangles, forty to fifty per pixel, from foliage whose clusters stay fine under a one-pixel
  error threshold. The software rasterizer carries it at 3.2 and 4.8 ms, but this is the ceiling.
- **Local-light visibility scales with pixels times overlap.** The courtyard's torches cost
  0.87 ms at 720p and 7.3 ms at 1440p: thirty lights in range of every pixel, each traced 16 steps
  through the GI volume. The tiles (fix 4) removed the loop over the other 450 lights; what is
  left is the traces. A cheaper visibility for local lights (a small shadow map or a cached
  voxel occlusion per light, traced once per frame rather than once per pixel per light) is the
  next step there.
- **Screen GI at grazing light.** 2.9 ms on the beach at dusk in a storm, against 0.3 by day:
  the probes' screen traces run long when the whole view is dim and in shadow. Not investigated.
- **Streaming holds.** With 128 MB of geometry pages and 64 MB of textures, the hill view settled
  in 2.5 s with 615 of 4,720 pages, nothing evicted and nothing short of room, and the shadow
  cascade that frame drew 83,000 clusters from what had arrived.
- Textures at 2048 (31 layers, BC7 and BC5) cost nothing measurable over 1024 in these passes.

#### Fix 5: movable casters culled once (after #29)

Gameplay step 1 (#29) draws movable instances every frame into a quarter of a shadow layer per
cascade they fall in, each through its own cull, and the cull's fixed cost (a seed over every
instance and one dispatch per hierarchy level) came four times over: 0.47 ms a frame for the
cove's bobbing ship. Now the movable instances are culled once, through the widest touched
cascade's view with its near plane opened towards the light and the finest cascade's texel
scale, bounded by what the camera can see of each caster, and that one list is drawn into every
touched quarter. Up to 1,024 movable instances; beyond that each cascade culls its own, since
drawing the whole list into every quarter costs more than the traversals saved (19,500 bobbing
instances: 12.5 ms shared against 5 to 8 per cascade). A cascade is touched only where a movable
thing's shadow can land, by the same planes the receiver cull uses.

| Shadows pass, movable instances bobbing | Before | After |
|---|---|---|
| Cove, the ship (7 instances) | 0.47 ms | 0.19 ms |
| Cove, 102 instances within 400 m | 0.62 ms | 0.45 ms |
| Stress, 40 ships and the reefs | 0.68 ms | 0.45 ms |
| Stress, 19,500 instances (per-cascade path) | 5.9 ms | 4.7 to 8.0 ms, run to run |

Images unchanged within noise except the cove ship alone (41 dB): the camera-bounded detail
thins the rigging's shadow on the hull slightly, as fix 1 does for static casters.

#### Fix 6: shadow cascades through the software rasterizer

The cascades drew every cluster through the hardware rasterizer, which shades 2×2 quads and
wastes most of its work on the pixel-sized triangles a wood's leaves become in a shadow map;
the main view has sent such clusters to a compute rasterizer from the start. The cull now
marks small clusters software for the cascades too (by their size in texels, since an
orthographic view has no distance), a depth-only variant of the software rasterizer draws
them into a buffer of reversed depth with atomics, with the same slope-scaled bias as the
hardware pipeline so that surfaces do not shadow themselves, and a pass of one triangle lays
that buffer over the cascade's map under the usual depth test. Movable instances' quarters
stay on the hardware path for now.

| Flyover, shadows pass: redraw mean / p95 / max (average per frame) | Hardware only | With software |
|---|---|---|
| Stress, fair 10:00 | 2.63 / 5.22 / 10.9 ms (0.81) | 1.61–1.78 / 3.1–3.5 / 6.3–8.0 ms (0.54–0.60) |
| Stress, storm 18:40 (low sun) | 8.45 / 17.3 / 33.1 ms (2.91) | 6.41–6.88 / 11.9–12.6 / 28–33 ms (2.27–2.43) |
| Cove, fair 10:00 (two runs each) | 0.35 / 0.29 / 0.6 ms (0.076) | 0.36 / 0.33 / 0.6 ms (0.083) |

Measured against a build of `main` on the same day, cold frames discarded; the wood's two
figures are two runs of the same build. The software list is drawn with no clears and no work
driven from the CPU: each drawing stamps its depth words with a number of its own (24 bits of
reversed depth under an 8-bit stamp), the merge reads only its own stamp, and the buffer is
cleared once every 255 drawings; the merge is drawn indirectly from the cull's arguments, so
a cascade with no software clusters draws nothing. A cluster goes to software under 16 texels
across (40 pixels in the main view): at 40 the cove, whose clusters are mostly large, lost
0.03 ms a redraw to mid-sized clusters that the hardware draws faster without a pixel shader;
at 8 the wood lost a quarter of its gain.

What the cove still pays, 0.007 ms a frame or 13 µs a cascade redraw, is one more dependent
compute dispatch per redraw: timestamps around the sub-passes put the whole difference in the
cull and software pass, with the hardware pass and merge unchanged (+0.8 µs), and a build whose
cull never chose software (an empty dispatch) still cost 7 µs of it. Folding the dispatch into
the cull's compute pass, marking touched tiles from the cull so the merge draws only those (64
and 16 pixel tiles), and a lower threshold all measured the same or worse. Skipping the dispatch
on the CPU would need the software count read back, a frame late, and a cascade can have
pixel-sized casters whatever its size, so there is no safe bound without it.

Images are unchanged within noise (hill 56.5 dB against a 58 dB floor, beach 45.7 against
45.4, cove 46.4 against 46.5); without the slope bias the first attempt darkened the barrels
and leaves with self-shadowing, which the difference image showed. A GPU test renders a ball
over a floor with and without the ball, under a sun alone, and requires the floor to darken
under it; the ball's clusters all take the software path in the nearest cascade, so a software
rasterizer that wrote nothing, a merge that wrote the far plane or was never drawn, or a cull
that sent nothing to software fails it. The software path holds a 16 MB depth buffer for the
2048² cascades and 16 more bytes of indirect arguments.

The low sun's far cascade still asks for over two million clusters; drawing them twice as fast
does not change that count. The remaining lever is a coarser level of detail for leaves at
shadow-map distances, a cluster builder question.

### Where things stand

Fixed and in pull requests: shadow caster culling (#25, merged), cloud shadows beyond their map
(#26, merged), the cull lists' overflow counter and capacity (#27 and #31), light tiles (#30,
merged), movable casters culled once (#32), and the cascades through the software rasterizer
(fix 6).

Found and not yet fixed, in order of how much they would give:

1. Shadow cascades at full texel LOD for leaves the camera cannot resolve: with the software
   rasterizer (fix 6) a redraw is 1 to 23 ms in dense woods, worst with a low sun, and the far
   cascade still selects over two million clusters there.
2. Local-light visibility traced per pixel per light: 7 ms at 1440p where thirty torches overlap.
3. Haze fixed at 0.0011 per metre: a clear day has the air of a hazy one, which long views show.
4. The voxel light pass loops over every light per voxel; per-block lists would cut its loop
   as the tiles did the lighting pass's, though not its traces.
5. Forty to fifty triangles a pixel from foliage at a one-pixel error threshold, which is the
   main view's own ceiling on small GPUs.
