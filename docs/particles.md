# Particles: design

Rain that streaks past and splashes on the flagstones, spray thrown off the rocks, smoke from a
chimney leaning with the wind, sparks off a forge. This document is the plan for a particle
system in the spirit of Unreal's Niagara, agreed before any of it is built and kept up to date as
each step lands.

## Goals

- **Data-driven.** An effect is a short text file of emitters, each built from small reusable
  modules: spawn, initial attributes, forces, drag, wind, curl noise, collision, colour and size
  over life, kill. Like `.ocean` and `.weather` files, it changes without touching code.
- **Dynamic.** Effects have named parameters that the weather, the time of day or gameplay can
  drive at any moment: the weather's `rain` sets how hard it rains, a script sets how hot a fire
  burns.
- **Simulated on the GPU.** A persistent pool of particles, a free list, compute simulation and
  indirect draws. The CPU never touches a particle and never waits for the GPU.
- **Part of the whole system.** A particle is lit as the scene around it is:
  - the sun, its cascaded shadows and the clouds' shadow;
  - light bounced from the GI volume, and the sky;
  - the same aerial perspective as everything else.

  It also:
  - fades softly where it meets the scene;
  - holds steady under TAA;
  - glows into bloom when emissive;
  - can cast a shadow.

  It collides with the scene and the sea, and spray can wet the rocks it lands on.
- **Realistic.** Physical units throughout: drops fall at their measured terminal speed, rain
  rate sets how many there are, fire glows at its temperature.
- **Performant everywhere,** the web and integrated GPUs included. Budgets are fixed, LOD is by
  distance, and the cost is measured at every step (see [Costs](#costs)).

Not in the first version, though the design leaves room for each:
- ribbons and beams;
- particles driving the GI volume or lighting the scene, beyond an effect's own light;
- fluid simulation;
- particles seen through the water's refraction;
- particles under water.

## What there is today

- **No particles.** The weather's `rain` (`clouds/weather.rs`) only darkens cloud bases and
  greys the air. There is no snow setting. `Observation` already carries WMO codes for snow.
- **Frame order** (`renderer.rs`, profiler sections in `profiler.rs`):

  ```text
  sky → clouds → ocean (waves) → shadows → seabed map → wet line → GI volume
      → cull/raster ×2 → resolve → clouds on screen → screen GI → lighting → water → post
  ```

  Post (`post.rs`) is TAA and upscaling, then exposure, bloom and tonemapping.
- **The water pass is the precedent for compositing.** It is a compute pass after deferred
  lighting. It reads the lit HDR target, the G-buffer's depth and motion, and writes its own
  colour, depth and motion (`WaterTargets`), which post reads in place of the G-buffer's. The
  lit opaque scene stays untouched for refraction. Particles go after it, in the same way.
- **What a particle can be lit and collided with, all already on the GPU:**
  - `sun_irradiance_at` (atmosphere and cloud shadow) and `sun_shadow` (four 2048² cascades,
    cached and rarely redrawn), in `common/environment.wgsl` and `lighting/sun_shadow.wgsl`;
  - `sky_radiance`, the sky with its clouds;
  - the GI volume (`gi/volume.rs`): five 96³ clipmap levels from 0.2 m voxels, holding radiance
    and the distance to the nearest surface. That is a signed distance field of the scene near
    the camera, which can be traced anywhere, on screen or off;
  - the scene's depth and normals, this frame;
  - the seabed map (`ocean/seabed.rs`): 1 km of the scene seen from above, clipped 2 m above the
    sea;
  - the water-line map (`ocean/waterline.rs`): where the water stands now and has stood lately,
    over 128 m around the camera, at 25 cm;
  - the scene's static local lights (`Light` in `common/scene.wgsl`).
- **TAA** (`post/taa.wgsl`) reprojects with the closest depth's motion in a 3×3 neighbourhood
  and clips history to the neighbourhood's colour. It has no notion of translucency. Anything
  drawn over the scene without its own motion would ghost.
- **Limits** the renderer already requires (`gpu.rs`): 8 storage buffers and 5 storage textures
  per shader stage. Core WebGPU has no float atomics, no 64-bit atomics, no multi-draw indirect,
  and no non-zero `first_instance` in indirect draws without an optional feature.
- **Scene files** (`.solstice`) have optional tagged sections after the manifest, each a
  four-byte tag, a length and its contents. The sea is one (`solstice-asset/src/file.rs`). Bake
  scripts (`scenes/*.txt`) have one command per line.
- **Gameplay** (`docs/gameplay.md`, PR #24) asks the renderer for general APIs only. It never
  sees an actor. Particles follow that rule.

## References

- **Unreal Engine's Niagara.** Systems of emitters, emitters as stacks of modules in spawn and
  update stages, user parameters, GPU simulation, events between emitters, and collision against
  scene depth and distance fields.
- **Gareth Thomas, *Compute-Based GPU Particle Systems* (AMD, GDC 2014).** Dead and alive index
  lists, emission and simulation in compute, sorting, indirect drawing.
- **Lorach, *Soft Particles* (NVIDIA, 2007).** Fading sprites by their distance to the scene's
  depth.
- **Bridson, Hourihan and Nordenstam, *Curl-Noise for Procedural Fluid Flow* (SIGGRAPH 2007).**
  Divergence-free turbulence from the curl of a noise field, so smoke swirls without bunching up
  or thinning out.
- **McGuire and Bavoil, *Weighted Blended Order-Independent Transparency* (JCGT 2013).** The
  alternative to sorting; see [Decisions](#decisions-for-you).
- **Garg and Nayar, *Photorealistic Rendering of Rain Streaks* (SIGGRAPH 2006),** and
  **Tatarchuk, *Artist-Directable Real-Time Rain Rendering in City Environments* (ATI, 2006).**
  How a drop looks as a streak over one exposure, a top-down map so that rain doesn't fall
  indoors, and splashes.
- **Gunn and Kinzer (1949),** terminal velocities of raindrops. **Marshall and Palmer (1948),**
  the distribution of drop sizes by rain rate. **Locatelli and Hobbs (1974),** fall speeds of
  snowflakes, about 1 m/s.
- **Henyey and Greenstein (1941).** The phase function for light scattering through droplets
  and smoke: strongly forward for mist and spray, mildly for smoke.
- **Six-way lighting** (Unity's VFX Graph, with maps baked from simulations): smoke lit from any
  direction by six baked light maps. A later quality step for smoke.
- **Motion-vector flipbooks** (as in Unreal's SubUV motion-vector blending): flipbook frames
  blended along baked motion, so slow smoke animates smoothly from few frames.

## The design

### Effects as data

An effect is a text file of one or more emitters. The built-in ones are in
`crates/solstice-render/particles/`:

```text
# crates/solstice-render/particles/rain.particles
# Rain: streaks that fall with the wind and splash where they land.
param rain 0                       # 0..1, set by the weather
param size 1                       # drop size relative to the rate's typical one

emitter drops
  budget 40000                     # most particles alive at once
  follow camera                    # spawns in a box that moves with the camera
  spawn rate=14000*rain            # per second, scaled by the parameter
  shape box 50 24 50 offset=0 12 0
  lifetime 2.5
  size 0.0015*size 0.003*size      # drop radius in metres, uniform between the two
  velocity terminal                # starts at its terminal speed (Gunn and Kinzer)
  gravity
  drag terminal                    # quadratic, so that it holds its terminal speed
  wind 1                           # carried by the weather's wind at its height
  collide cover water kill event=splash
  render streak exposure=0.016 opacity=0.3 lit
  lod near=25 far=50

emitter splash
  budget 6000
  spawn event=splash count=4
  lifetime 0.25 0.4
  velocity cone up=30 speed=1.2 2.0
  size 0.001 0.002
  gravity
  render sprite opacity=0.5 lit
```

- **Lines are a keyword, then values,** as in weather files. `#` begins a comment. Every module
  has defaults, so a file names only what differs.
- **Parameters.** `param NAME DEFAULT` declares one. A value anywhere may be a number, a
  parameter, or a number times a parameter (`14000*rain`). That is all the expression language
  there is. A module that needs more takes a curve.
- **Curves** for anything over a particle's life, such as `color life 0=1 .6 .2 1 .5=.3 .3 .3 .6
  1=.2 .2 .2 0`, as keys from 0 to 1. Each is baked to 32 entries in a small curve atlas on the
  GPU.
- **Errors name the line,** as with weather and oceans, and a bad file falls back to no effect
  with a warning, never a crash.

**The modules**, by stage:

| Stage | Modules |
|---|---|
| Spawn | `spawn rate=` per second, `burst=` once, `event=` from another emitter, `distance=` per metre moved (trails); `follow camera` or `follow effect` |
| Shape | `point`, `sphere`, `box`, `disc`, `cone`, and two that join the scene: `sea crests` (where waves break) and `surf` (where the sea rises against a face) |
| Initial attributes | `lifetime`, `velocity` (cone, radial, `terminal`, inherited from the effect's motion), `size`, `rotation`, `color`, `flipbook` frame, `temperature` |
| Forces | `gravity`, `drag` (linear, quadratic or `terminal`), `wind`, `curl` noise, `buoyancy` (hot gas rising and cooling), `attract` to a point, `vortex` |
| Over life | `color`, `opacity`, `size` and `spin` by curve; `temperature` cooling for fire and sparks |
| Collision | `collide` against `depth`, `field` (the GI volume's distance field), `cover` (the top-down cover map) and `water`; then `bounce`, `stick`, or `kill`, with an optional `event=` |
| Kill | by age; by leaving the follow box; below the water |
| Render | `sprite` (facing the camera, the velocity, or up), `streak`, `flipbook`, later `mesh`; shading `lit`, `emissive` or `unlit`; `soft=` fade distance; `shadow` to cast one |

**How modules run.** Modules are not compiled into a shader per emitter, as Niagara does. Each
emitter becomes a fixed record on the GPU: a bit for each module it uses, its values, and its
curves' rows in the atlas. One simulation shader reads that record and branches on the bits.
Every particle in a workgroup belongs to the same emitter, so the branches are uniform and cost
almost nothing.
- **Why:** browsers compile pipelines slowly, and on the first use. One set of pipelines,
  created at start-up, means loading an effect never stalls a frame.
- **If it measures slow,** the hottest emitters can get specialised pipelines through WGSL
  `override` constants, without changing the format.

**Effects in scenes and from code.**
- **Bake scripts** gain `effect PATH X Y Z YAW SCALE [param=value ...]`. Placed effects go into a
  new tagged section, `PRTC`, holding each effect's text, as the sea's section holds its ocean.
  Older readers skip it.
- **The library** gets general calls only:

  ```rust
  let fire = renderer.particles().spawn(&Effect::by_name("campfire")?, transform);
  renderer.particles().set(fire, "heat", 0.8);       // eased, like the weather
  renderer.particles().set_transform(fire, transform);
  renderer.particles().stop(fire);                   // stops spawning; what is alive dies out
  ```

  Gameplay's `effect` component calls these. A footstep or a sword's hit is a `burst`.
- **The weather's own effects,** rain and snow, are owned by the renderer, which sets their
  parameters from the weather each frame. A scene does not place them.

### Simulation on the GPU

**The pool.** One storage buffer of particles, shared by every emitter. Each emitter has a fixed
slice of it, its `budget`, and keeps its own free list and alive list there.
- **A particle is 64 bytes:**
  - position and age;
  - velocity and lifetime;
  - the position last frame, for motion vectors, and packed size and rotation;
  - packed colour, packed temperature and flipbook frame, and its seed.
- **Its light is 16 bytes more,** in a separate buffer written by the lighting pass (below).
- **The pool's size** is set by quality:

  | Quality | Particles | Memory |
  |---|---|---|
  | High | 262,144 | 20 MB |
  | Medium | 131,072 | 10 MB |
  | Low | 65,536 | 5 MB |

  Effects that ask for more than the pool holds have their budgets scaled down to fit.

**Each frame** is five dispatches and the draws, however many emitters there are:

1. **Plan**, one workgroup. For each live emitter, it works out:
   - how many particles to spawn: rate times time, plus bursts and queued events, limited by its
     free list;
   - the emitter's LOD;
   - where its workgroups start in the spawn and update dispatches.

   It writes the indirect arguments for everything after it. One small pass replaces a dispatch
   per emitter, which matters on the web, where each dispatch costs CPU time in the browser.
2. **Spawn.** Each new particle pops a slot from its emitter's free list with an atomic. It is
   seeded from the emitter's running spawn count, not its slot, so a particle's random values
   are the same however the GPU orders the atomics.
3. **Update.** Ages each live particle and applies its emitter's forces and collisions. Then it
   kills the particle, returning its slot to the free list, or appends it to next frame's alive
   list. The two alive lists take turns, one read while the other is written. Collisions append
   events for other emitters to spawn from next frame.
4. **Light**, per particle, described under [Rendering](#rendering).
5. **Sort**, for emitters that need it (below), and **cull**: particles outside the frustum or
   smaller than a third of a pixel are left out of the draw lists, and the indirect draw counts
   are written.

Then one `draw_indirect` per visible emitter. The vertex shader pulls its particle from the
emitter's draw list by `instance_index`, so no instance buffer is built. The emitter's offset
into the list comes from a uniform array indexed by the draw, since a non-zero `first_instance`
needs an optional feature.

**Integration** is semi-implicit Euler at the frame's time step. Steps longer than 1/30 s are
split, so a slow frame doesn't throw sparks through walls. Drag is integrated exactly for the
linear case, and stably for the quadratic case.

**The CPU never reads a particle.** Statistics come back asynchronously, as culling statistics
do: alive, spawned and dropped for lack of budget, per emitter.

**A CPU reference** in Rust runs the same modules for one particle at a time. GPU tests compare
against it, as the waves' tests compare against `spectrum.rs`.

### Budgets and LOD

- **Hard budgets.** Each emitter has its slice; when the free list is empty, spawns are dropped
  and counted. Nothing grows, nothing overflows, and memory is fixed at start-up.
- **LOD by distance.** Between an emitter's `lod near` and `far`, its spawn rate falls to a
  fraction, and the particles that are spawned grow by its inverse square root, so the effect
  covers the screen as much as before with fewer particles. Beyond `far`, or outside the frustum
  by its bounds, it stops spawning. It keeps simulating what is alive until that dies, and wakes
  with a short pre-warm when it comes back.
- **Camera-following emitters,** rain and snow, are densest near the camera and thin with
  distance. Beyond about 50 m, rain is the weather's grey haze, which the clouds already draw.
- **Quality** scales every spawn rate: 1 on High, 0.6 on Medium, 0.35 on Low. Low also draws
  large translucent particles at half resolution (step 7).
- **Overdraw** is the real cost on integrated GPUs, more than particle count. So:
  - sprites fade out as they reach the camera, which avoids a full-screen layer of smoke;
  - each emitter's on-screen size can be capped;
  - a debug view, `particles`, shows overdraw as a heat map.

### Rendering

**Where it goes in the frame.** Particles are simulated after the scene is resolved, so they can
collide with this frame's depth, and lit after the sun's shadows, the GI volume and the clouds
are ready. They are drawn after the water, into the same colour target that post reads, the
water pass's when there is a sea and the lit HDR target when not. Two new profiler sections:

```text
… → resolve → clouds on screen → screen GI → lighting → water → particles → particle draw → post
```

**Depth.** The opaque scene's depth with the water in it is R32Float storage, which the hardware
can't depth-test against. A tiny full-screen pass copies it into a Depth32Float attachment, so
hidden particle fragments are rejected before shading. That matters most for overdraw on
integrated GPUs. Particles test against it but never write it.

**Lit translucent particles.** Light is worked out once per particle, in compute, not per pixel.
That is the main saving over shading each fragment, and on a small sprite it looks the same.
- **Sun:** the sun's light through the atmosphere and the clouds' shadow, times its cascaded
  shadow at the particle. Particles larger than 2 m sample four points across themselves, so a
  smoke column is lit along its length.
- **Sky:** the sky's light, clouds included, as a first-order spherical harmonic, made once a
  frame by a tiny pass. It is darkened by how enclosed the particle is, from the GI volume's
  distance field, so smoke in an archway is darker than smoke in the open.
- **Bounce:** two short traces of the GI volume per particle per frame, in directions that
  rotate from frame to frame, averaged in the particle over about eight frames. Spray by a sunlit
  cliff picks up its colour. Beyond the volume, the sky alone.
- **Local lights:** the scene's lights within range, as the lighting pass has them.
- **Emissive:** fire glows by temperature, from Planck's law, in the same physical units as the
  sun. Bloom and exposure then treat it as they treat any bright pixel, with no special case.
- **What each pixel does:** the stored light is applied with the emitter's Henyey-Greenstein
  phase towards the eye, so mist and spray glow when backlit by a low sun. Flipbooks with
  normals shade by them. Then the particle's colour is premultiplied by its opacity.

**Soft particles.** Each fragment fades by its distance to the scene's depth over the emitter's
`soft` distance, so smoke never shows a hard line where it crosses the ground or the water. It
also fades near the camera's near plane.

**Aerial perspective.** The same `aerial_perspective` as everything else, per particle, applied
before premultiplying. A distant chimney's smoke fades into the haze exactly as the chimney does.

**Motion and TAA.** Translucent sprites over a moving scene are what TAA handles worst: it
reprojects by the opaque scene's motion and keeps the particle's colour as stale history.
- **Sprites are drawn with the jittered projection,** so TAA anti-aliases their edges.
- **A reactive mask.** Each particle writes its opacity into a small R8 target, with a max
  blend. TAA lowers the weight of history and widens its colour clip where that is high. This is
  what Unreal's responsive AA and FSR's reactive mask do. Rain streaks and sparks then stay sharp
  instead of smearing.
- **Solid particles** (alpha-tested sparks, debris, later meshes) write their own depth and
  motion, from the particle's position this frame and last, over the composite's. TAA then
  follows them as it follows any surface.

**Sorting.** Only where it shows.
- **Not sorted:** additive emitters (sparks, fire's glow), and tiny particles such as rain,
  splashes, spray and snow. Their order is invisible.
- **Sorted back to front:** large, dark, alpha-blended emitters such as smoke and dust. Each
  sorted emitter's draw list is sorted by depth with a bitonic sort in workgroup memory:
  32-bit keys of 16-bit depth and a 16-bit index, at most 16,384 particles per sorted emitter.
- **Between emitters:** drawn far to near by the centres of their bounds.

**Shadows from particles** (step 8). The sun's shadow maps are cached and rarely redrawn, so
particles cannot go into them. Instead, a small per-frame map of opacity seen from the sun:
- 512², over the nearest cascade's area;
- particles splat their opacity into it with a multiplicative blend.

The lighting pass, the water pass and particle lighting all multiply the sun by it. Smoke then
shades the ground below it and shades itself. Only emitters marked `shadow` cast one, and it is
off on Low.

### Collision

Each test is cheap, and an emitter picks the ones it needs:
- **`depth`.** This frame's depth and normals, so a particle bounces off exactly what is drawn.
  It is detailed, but blind off screen and behind things.
- **`field`.** The GI volume's distance field: everywhere within the volume, on screen or not,
  at 0.2 m near the camera and coarser further off. This is Niagara's distance-field collision.
  It is the default for sparks and debris.
- **`cover`.** A new top-down map of the scene's highest surfaces: 512² over 256 m, half a metre
  a texel. It is drawn by the shadow maps' depth-only pass, like the seabed map but with nothing
  clipped away, and only when the camera has moved far or geometry has streamed in. Rain and
  snow stop where they first meet a surface, so it doesn't rain indoors or under the gate's arch.
  Splashes spawn where the drops land. Later, rain wetness reads the same map to know what is
  sheltered.
- **`water`.** The water's height now, from the water-line map within 64 m of the camera, and
  the calm level beyond it. A drop or spray hits the waves that are drawn, not a flat plane. The
  map's height is what the water reaches at each texel, so it is a few centimetres generous
  against cliff faces, which is good enough for a landing.

**What a hit does:** `bounce` with restitution and friction, `stick` (snow settling, for its
lifetime), or `kill`. Any of these can send an `event` to another emitter, which spawns from
the hit's position and normal next frame: splashes from raindrops, rings from drops landing on
the sea, sparks from sparks.

**Wetness.** Spray that lands on the rocks wets them. Its impacts raise the remembered water
level at that texel of the water-line map, with an `atomicMax` on the height, as a quantised
`u32`. The water-line update then folds it in, so it dries as the sea's own wet band does, with
no new wetness logic. Rain wetness is its own later design. It will use the same wet material
and the cover map, not individual drops.

### The first real uses

Each one proves part of the system, in the cove:

| Effect | Proves | Driven by |
|---|---|---|
| **Rain**: streaks, splashes on stone, rings on the sea | Camera-following emitters, cover collision, water collision, events, streak rendering, the reactive mask | The weather's `rain` and wind |
| **Snow**: flakes drifting, settling and melting | Wind, curl noise, `stick` | A new weather setting, `snow` |
| **Sea spray**: thrown off breaking crests and off the rocks in the surf | Shapes that read the sea, backlit phase, wetting rocks | The waves' folding and the water-line map's rise against faces |
| **Smoke**: from a chimney or a fire | Sorting, soft particles, sky and bounce light, buoyancy, curl noise, casting shadows | Wind, and a script's `heat` |
| **Fire**: flames over the smoke | Emissive by temperature, bloom, a light of its own | A script's `heat` |
| **Sparks**: from the fire, and from impacts | Additive, field collision with bounce, solid motion vectors | Bursts from gameplay |
| **Dust**: lifted from dry ground in a strong wind | Wind, LOD, cover-map spawning, wetness keeping it down | Wind speed, and wetness |

- **Rain is realistic,** not a texture:
  - the rain rate in mm/h sets the number and size of drops, by Marshall and Palmer;
  - each drop falls at its terminal speed, about 4–9 m/s, and leans with the wind;
  - a streak is the drop's path over one exposure, lit by the sun and the sky behind it, as Garg
    and Nayar describe.
- **Snow** falls at about 1 m/s and wanders with curl noise. Flakes settle and melt; settling as
  a white layer on the ground is out of scope here.
- **Spray** comes from the joins with the sea that already exist:
  - from crests, where the waves' surface folds over: the Jacobian of their displacement falls
    below a threshold. Ocean step 4, foam, needs the same quantity, and the two will share it;
  - from the surf, where the water-line map shows the water rising fast against a steep face.
- **Fire, a fire's light.** A fire carries one point light, which flickers with its heat. It uses
  the moving-light API that gameplay is adding (`docs/gameplay.md`), not one of its own.

### Debug views

- **`particles`:** overdraw as a heat map, the thing to watch on integrated GPUs.
- **`particle light`:** each particle's stored light, without its texture or colour.
- **Statistics overlay:** alive, spawned and dropped per emitter, and the pool's use.

## Steps

Each step is its own PR off `main`, and none starts until the last is merged. Each has:
- GPU tests that are mutation-tested: the bug is put back, and the test must fail;
- costs measured on the RTX 5080 and the integrated GPU, repeated, with cold first runs
  discarded;
- `cargo test --workspace` passing;
- clippy clean with `-D warnings`, natively and for `wasm32-unknown-unknown`;
- a review by a separate AI agent, and your go-ahead before it is merged.

1. **The core.**
   - The `.particles` format and its parser, and the CPU reference.
   - The pool, free lists, alive lists, and the plan, spawn and update passes.
   - Indirect drawing of unlit, soft sprites; the depth copy.
   - The basic modules: rate, burst, point, sphere and box shapes, lifetime, velocity, size,
     colour, gravity, drag, and colour and size over life.
   - Profiler sections, the `particles` overdraw view, and statistics.
   - A test fountain effect for the viewer (`--effect NAME`).
   - Tests:
     - The GPU's trajectories match the CPU reference.
     - Particles are conserved: alive plus free always equals each budget, over thousands of
       frames of spawning and dying. A double free or a leaked slot fails it.
     - Spawns beyond the budget are dropped and counted, and never write outside the slice.
     - A sprite fades to nothing where it meets a surface.
     - Uneven frame times give the same result as even ones.
2. **Lighting and haze.**
   - Per-particle light: sun, cascaded shadow and cloud shadow; the sky's harmonics with
     enclosure; GI bounce; local lights.
   - Phase functions, emissive by temperature, and aerial perspective.
   - The `particle light` view.
   - Tests:
     - A particle in a shadow is darker than one in the sun, by the shadow's amount.
     - At equal distance, a particle hazes exactly as an opaque surface does.
     - A particle beside a red wall picks up red.
     - Emissive particles reach bloom.
3. **Motion, TAA and sorting.**
   - The reactive mask and its use in TAA.
   - Solid particles' depth and motion.
   - The bitonic sort, and ordering between emitters.
   - Tests:
     - A moving solid particle's motion vector lands it on its last frame's pixel.
     - Streaks don't leave a trail under a moving camera, measured against TAA off.
     - Sorted output matches a CPU sort.
     - With the reactive mask removed, the ghosting test fails.
4. **Forces and collision.**
   - Wind from the weather, brought down to the particle's height with the log profile the sea
     uses, plus gusts; curl noise, buoyancy, attract and vortex.
   - Collision against depth, the distance field, the cover map and the water, with bounce,
     stick and kill.
   - Events between emitters.
   - Tests:
     - Nothing falls through the floor, the cliff, or the sea.
     - Bounces keep the right energy for their restitution.
     - Curl noise is divergence-free.
     - Particles under the gate's arch stay dry of rain.
5. **Rain and snow.**
   - The renderer's own rain and snow, driven by the weather.
   - A new weather setting, `snow`, in the weather files and in `Observation`.
   - Streak rendering, splashes and rings on the sea.
   - Tuned against photographs and against the rain-rate figures.
   - Tests:
     - Drop counts and speeds match Marshall and Palmer, and Gunn and Kinzer.
     - No rain falls beneath cover.
     - `rain 0` costs nothing, as no emitter runs.
6. **Sea spray.**
   - Spawning from folding crests and from surf against faces.
   - Backlit phase, and wetting the rocks through the water-line map.
   - Tests:
     - A calm sea throws no spray, and a gale throws plenty.
     - Spray lands where it should, and the rocks it lands on are wet, then dry.
7. **Smoke, fire, sparks and dust.**
   - Flipbooks with normals and motion vectors.
   - Fire by temperature with its light; sparks; dust over dry ground.
   - Half-resolution drawing of large translucent particles on Low, upsampled by depth.
   - Example effects placed in the cove.
8. **Particle shadows.** The sun's opacity map, cast by marked emitters and received by the
   lighting pass, the water and particles. Tests: smoke's shadow darkens the ground under it, by
   its opacity.
9. **README, screenshots and costs.**
   - Measured in a heavy storm with spray and a fire, on the RTX 5080, on the integrated GPU,
     and in Chromium, Edge and Firefox.
   - Viewer controls for rain and effects.

Mesh particles, such as debris and splinters, forward-shaded with their own depth and motion,
come after, if you want them.

## Costs

Targets, at 1280×720, measured each step against them. The integrated GPU already takes about
30 ms a frame at Low, so particles must take only a small share of that.

| Scene | RTX 5080, High | Integrated, Low |
|---|---|---|
| No effects running | 0 ms: nothing is dispatched | 0 ms |
| Heavy rain: about 40k drops, with splashes | 0.25 ms | 1.2 ms |
| Storm: rain, spray and a fire with smoke | 0.6 ms | 2.0 ms |
| Simulation alone, per 100k particles | 0.1 ms | 0.8 ms |

Each step reports, separately:
- plan, spawn and update;
- lighting;
- sort;
- the depth copy;
- the draw;
- overdraw, for the draw;

plus memory. Timings come from the profiler natively. In browsers, only the frame time can be
measured.

## Decisions for you

1. **Which first use to build first, after the core.** I recommend rain. The weather already has
   `rain`, and it proves cover collision, water collision, events, streaks and the reactive mask
   in one go, in the cove, where it shows. Spray is the alternative: it ties into the sea the
   most.
2. **Snow in the weather.** I recommend a `snow` setting from 0 to 1 beside `rain`, set from
   `Observation`'s snow codes. It changes the weather files and the weather type that the web
   page and gameplay read.
3. **Sorting or order-independent blending.** I recommend sorting only the emitters that need it,
   and ordering emitters by distance, as Unreal does. Weighted blended OIT needs no sort at all,
   but it muddies dark smoke in front of bright fire, which is the case that matters most.
4. **Where smoke and fire textures come from.** I recommend baking our own flipbooks: a small 2D
   smoke and fire simulation in `solstice-procgen`, written out as frames with normals and
   motion vectors. There is then no question over licences, and they can be retuned. The
   alternatives are noise shaded in the shader, which is cheaper but looks procedural, or CC0
   flipbooks from elsewhere, if suitable ones can be found.
5. **The budget on integrated GPUs.** I propose the targets in [Costs](#costs): about 1.2 ms for
   heavy rain and 2 ms for a storm at Low, on a frame that is already about 30 ms. Do you want
   tighter, or effects held back further on Low?
6. **Particle shadows (step 8) and mesh particles.** Both are optional. I recommend shadows,
   because smoke without them looks pasted on, but off on Low. Mesh particles can wait until
   gameplay needs debris.

Publishing the cove with effects is a separate question, for when there is something worth
showing.

## Risks

- **Particles behind the water.** The water's refraction sees the lit scene without particles,
  so spray behind a wave's surface vanishes rather than showing dimly through. It is rarely
  visible, as spray is above the water. The fix, drawing particles into the lit target before
  the water pass, is possible later.
- **TAA and thin streaks.** If the reactive mask isn't enough, rain streaks can be drawn wider
  and fainter, or kept out of history altogether.
- **Overdraw on integrated GPUs.** Smoke near the camera can cost more than all the simulation.
  The defences are near fades, on-screen size caps and half-resolution drawing; the overdraw
  view shows whether they are enough.
- **WebGPU limits.** The update pass binds the pool, both alive lists, the free lists, the
  emitter table, events and counters: 7 storage buffers, within the 8 required. Collision adds
  textures, not buffers. Everything counts with `u32` atomics only.
- **Light per particle, not per pixel.** Large sprites could look flatly lit. Four samples over
  large particles, flipbook normals, and later six-way lighting are the answers, in that order.
- **The GI volume's reach.** Bounce light and field collision only work within it: its coarsest
  level is about 300 m across, centred on the camera. Beyond it, particles get sky light and the cover map.
- **Determinism.** Atomics hand out slots in any order. Tests check invariants and compare
  trajectories by spawn index, never by slot. Seeding by spawn index keeps the look the same
  from run to run.
- **Shared code.** Lighting, TAA, the water-line map and the weather all gain small changes.
  Each change is in the step that needs it, with the tests of the system it touches still passing.
