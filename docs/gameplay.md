# Gameplay: design

A design for making Solstice's worlds playable: things that move and collide, characters that
are skinned and animated, and game logic written in a Python-like language. The same game runs
in the browser and natively. This document is the plan, written before any of it is built, and
it will be kept up to date as each step lands.

## Goals

- **Built like Unreal.** A world of actors made of components, ticked in a fixed order of tick
  groups, talking through events. Engine systems are written in Rust where speed matters, and
  game logic is written in scripts. Unreal splits its work the same way: C++ beneath, with
  Blueprint and Verse above.
- **Python-like scripting**, the same language natively and on the web, with:
  - hot reload while the game runs, keeping each actor's state;
  - a sandbox: a script can reach only what the engine hands it, and a runaway script is
    stopped rather than freezing the page;
  - errors that name the file and line.
- **Modular.** Each part is its own crate and can be tested without a GPU. The renderer knows
  nothing about gameplay: it gains a few small, general APIs (things that move, skinned meshes),
  and nothing more. A scene without gameplay loads and costs exactly what it does today.
- **Fast on the web, integrated GPUs included.** The web build has one thread: GitHub Pages
  cannot send the headers that wasm threads need. The budget below assumes that, and every step
  measures against it.
- **Deterministic simulation.** Physics and gameplay advance in fixed steps, so a test can run a
  scene for N steps and check where everything ended up, and the native and web builds agree.
- **Gameplay drives the world.** Scripts can set the time of day, the weather and the sea, and
  react to them. The renderer already eases all three; gameplay only sets where they should go.

Not in the first version, with room left for each:
- networking and multiplayer;
- an editor beyond text files and hot reload;
- audio;
- navigation meshes and AI;
- cloth, soft bodies and destruction;
- ragdolls, which come after the IK step.

## What there is today

- **The scene is static.** `.solstice` files hold meshes as cluster DAGs, materials, instances
  and lights. Instances are uploaded once (`scene.rs`, `GpuInstance`) and never change. Only the
  ocean's surface writes its own instance buffer each frame.
- **Motion vectors are half ready.** `GpuInstance` already has `prev_world` next to `world`, and
  resolve uses it for each pixel's motion (`vg/resolve.wgsl`). Today the two are always equal.
- **Several passes assume nothing moves:**
  - sun shadow cascades are cached and only redrawn when their view moves (`shadow.rs`);
  - GI voxel levels are rebuilt only when the camera has moved far enough (`gi/volume.rs`);
  - the seabed map is redrawn only when the camera moves 128 m or the sea level changes;
  - the wetness work, in progress on its own branch, keeps a water-line map in the world.

  Each of these has to be told when something moves, or ignore things that move.
- **Two-phase occlusion culling** (`vg/culling.rs`) already copes with motion: anything the old
  depth hid wrongly is retried against the new depth in the same frame.
- **Vertices** are 16 bytes on one 16-bit grid per mesh: position, a packed normal, UV. There are
  no bone weights. The importer warns that it ignores skins and animations
  (`solstice-import/src/lib.rs`), and imports models in their rest pose.
- **Dynamic attributes**, each eased by the renderer over a given number of seconds:
  - `set_weather(Weather, seconds)`, a weather cycle (`set_weather_changing`), and reports from
    real observations (`Weather::observed`);
  - `set_ocean(Option<Ocean>, seconds)` and `set_sea_level`;
  - `TimeOfDay`: `advance(hours)`, and from it the sun, moon, stars and `Environment`;
  - each has a text form (`.weather`, `.ocean` files) with `parse` and `to_text`.
- **The viewers** drive these directly. The native viewer has command-line flags, and the web
  page calls `set_time`, `set_weather` and the rest from JavaScript (`apps/viewer/src/web.rs`,
  `apps/viewer/web/index.html`). The camera is a free-flying viewer camera. There is no input
  system beyond it.
- **The ocean exists only on the GPU.** `spectrum.rs` has a CPU reference: the spectrum, its
  seeded amplitudes (`pcg`, `gaussian_pair`) and `direct_sum`. Tests use it to check that the GPU
  draws exactly the waves the reference does. Nothing reads wave heights back during play.
- **Scene files** take optional tagged sections, such as `SEA `, which older readers skip. So
  actors, collision and scripts can be added without a new file version.

## References

- **Unreal Engine.** The gameplay framework: `UWorld`, `AActor`, `UActorComponent` and
  `USceneComponent`, tick groups (`TG_PrePhysics`, `TG_DuringPhysics`, `TG_PostPhysics`,
  `TG_PostUpdateWork`), tick intervals, timers and delegates, GameMode, PlayerController and
  Pawn, and Enhanced Input. Also Mass, Unreal's ECS for crowds; Blueprints, a visual language
  run by a VM and slower than C++, which is why Epic advises keeping heavy loops out of
  Blueprint ticks; Verse, a language Epic designed itself; and the Animation Blueprint, blend
  spaces, Control Rig, IK Rig and the Animation Budget Allocator.
- **Nanite skinning.** Unreal 5.5 and later draw skeletal meshes through Nanite. The cluster
  hierarchy is built in bind pose, and bounds are computed from the bones' transforms, so the
  hardware rasterizer can be used (Epic's *Nanite Foliage* documentation). Epic has published
  little more, so the skinning design here is our own, and step 8 proves it out before anything
  depends on it.
- **Godot.** A tree of nodes, signals for events, and GDScript: a Python-like language written
  for the engine, with optional static types and hot reload. It is the clearest example of an
  engine building its own language, and of what that costs: years of work on the language, its
  tools and its editor support.
- **Starlark** (Bazel's configuration language) and its Rust implementation `starlark-rust`,
  from Meta. **Monty**, Pydantic's sandboxed Python subset in Rust. **RustPython**. **Pyodide**,
  CPython compiled to wasm. **Rhai**. **Luau**, Roblox's sandboxed Lua.
- **Rapier**, from Dimforge: rigid bodies, joints, character controllers and scene queries in
  pure Rust. It compiles to `wasm32-unknown-unknown`, and its npm builds are what most web games
  use.
- **Animation.** Kavan et al., *Skinning with Dual Quaternions* (2007). Frechette's Animation
  Compression Library, for quantising clips. ozz-animation's sampling and blending. Aristidou and
  Lasenby, *FABRIK* (2011). Clavet, *Motion Matching* (GDC 2016), for later.
- **Water.** *The Technical Art of Sea of Thieves* (GDC 2018): ships float by sampling the same
  waves that are drawn.

## How the engine fits together

```text
                 scripts (.star)        actor files (.actor)       scene (.solstice)
                       │                        │                         │
                       ▼                        ▼                         ▼
 input ──▶ ┌──────────────────────────────────────────────────────────────────┐
           │ solstice-world   entities, components, transform tree,           │
           │                  tick groups, events, timers, fixed clock        │
           ├──────────────┬───────────────┬────────────────┬──────────────────┤
           │ solstice-    │ solstice-     │ solstice-anim  │ environment:     │
           │ script       │ physics       │ clips, blends, │ time of day,     │
           │ (Starlark)   │ (Rapier)      │ state machines,│ weather, sea     │
           │              │               │ IK             │                  │
           └──────────────┴───────────────┴────────────────┴──────────────────┘
                                         │ extract, once per frame
                                         ▼
           solstice-render: moving instances, bone palettes, lights, environment
```

- **`solstice-world`**: the framework. It has no GPU, no physics and no scripting, so every
  other crate can depend on it, and it tests headless.
- **`solstice-script`**, **`solstice-physics`** and **`solstice-anim`** each add their
  components and systems to a world. Each is optional, and each tests on its own.
- **`solstice-game`**: the glue. It runs the frame, owns input, and extracts what the renderer
  needs at the end of each frame. The viewers use it only when a scene has gameplay.
- **The renderer gains general APIs only:** move an instance, hide it or change its glow, upload
  a skeleton's bone matrices. It never sees an actor.
- **Two web builds.** The viewer stays as it is (319 KB gzipped today). A second build, with
  gameplay, is loaded only for scenes that have it.

## The gameplay framework

### Actors and components

Actors are entities, and components are plain Rust structs stored by an ECS. This is how
Unreal's actors and components behave, with Mass's speed when there are thousands of them.

| Unreal | Solstice |
|---|---|
| `UWorld` | `World` |
| `AActor` | an entity with a `Transform`, a `Name`, and optionally a `Parent` |
| `UActorComponent`, `USceneComponent` | components: `Mesh`, `RigidBody`, `Collider`, `Skeleton`, `Animator`, `Script`, `Light`, `Camera`, `Trigger`… |
| Blueprint class | an actor file (`.actor`): its components, their settings, and a script |
| Blueprint graph, Verse | a Starlark script |
| C++ gameplay code | Rust systems, in game crates |
| Tick groups, tick interval | the same, below |
| Delegates, events | events, delivered between tick groups |
| `FTimerManager` | timers |
| GameMode, PlayerController, Pawn | a game-rules script, a `Controller`, and a `Pawn` component |
| Enhanced Input | actions mapped to keys, buttons and touches in a text file |

An actor file follows the style of the `.ocean` and `.weather` files:

```text
# scenes/cove/gate.actor
name gate
mesh large_iron_gate                 # a mesh already in the scene file
collider box 1.7 2.4 0.1
script gate.star
property open_seconds float 2.5      # editable per placement in the scene
property locked bool false
```

A scene places it with its own settings: `actor gate.actor 20.7 0 0 90 locked=true`.

**Which ECS.** Both candidates were measured (below). They run at about the same speed:
- `hecs` is small and stable: 31 KB gzipped on wasm;
- `bevy_ecs` brings schedules, change detection and observers, at 160 KB, and breaks its API
  every few months.

The framework only needs fixed tick groups on one thread, which is a few hundred lines on top of
`hecs`. So the recommendation is `hecs`, with Solstice's own framework over it.

### A frame

```text
 1. Input        gather keys, mouse, pads and touch; map them to actions
 2. Events       deliver input actions and due timers to scripts
 3. PrePhysics   Rust systems, then script ticks for the actors that tick
 4. Physics      0 to 3 fixed 60 Hz steps; contacts and triggers are queued
 5. PostPhysics  deliver hits and overlaps to scripts
 6. Animation    state machines read gameplay's values; sample, blend, IK
 7. Late         the camera follows, and environment targets are set
 8. Extract      changed transforms, bone palettes, lights and environment go to the renderer
```

- **Fixed steps.** Physics and fixed ticks run at 60 Hz from an accumulator. Rendering
  interpolates between the last two physics states, which also gives each instance its
  `prev_world` for motion vectors. A frame runs at most three steps; past that, the game slows
  down rather than falling behind in a spiral.
- **Most actors never tick.** As in Unreal, ticking is opt-in, with an interval. Scripts mostly
  react to events: an overlap, an input action, a timer, the hour changing.
- **Events are queued** and delivered between tick groups. A handler never runs halfway through
  another system, so the order is always the same and replays give the same result.
- **Spawning and destroying** are deferred to the end of the tick group that asks for them.

### Where work runs

The rule, borrowed from Unreal: **scripts decide, Rust does the work.** A script sets a
character's desired speed, opens a gate or starts an animation state. The Rust systems then do
the per-frame, per-entity work: integration, physics, sampling, IK. This keeps the number of
crossings between script and Rust down to tens or hundreds per frame, not thousands, whichever
language is chosen.

## Scripting

### What was measured

Each candidate ran the same three scripts:
- an empty function called from Rust;
- `tick(dt)`, which calls two Rust functions, `get_x()` and `set_x(x + dt)`, as a script that
  moves something does;
- a loop adding up a million floats, for raw interpreter speed.

Setup:
- **Native:** Rust 1.98 release builds with thin LTO, on a Ryzen 7 9800X3D under Windows 11.
- **Web:** the same code built for `wasm32-unknown-unknown`, run in Node 24 on the same machine.
  Node uses V8, Chrome's engine.
- **Sizes:** the wasm after `wasm-bindgen`, with name sections removed, gzipped. There is no
  `wasm-opt` pass, which would shrink them a little. Pyodide's are its release files, gzipped.
- **Runs:** each figure is the best of three, and each benchmark was run two or three times;
  where the runs differed, the table gives the range. The benchmark's code is kept with this
  session's notes, and moves into the repository in step 3.
- **Start-up** on the web includes V8 compiling the module on its first run.

| | Download (gz) | Start-up | Empty call | `tick`, 2 Rust calls | Loop, per iteration | Python |
|---|---|---|---|---|---|---|
| **Starlark** 0.14, native | | 3.3 ms | 18 ns | 47 ns | 28 ns | syntax, a subset |
| **Starlark**, web | 1.33 MB | 26–31 ms | 33–43 ns | 89 ns | 54–72 ns | |
| **RustPython** 0.6, native | | 11 ms | 54 ns | 204 ns | 60 ns | nearly all of the language |
| **RustPython**, web | 4.15 MB, no stdlib | 77 ms | 63 ns | 233 ns | 105 ns | |
| **Monty** 1.5, native | | 0.9 ms | 1,730 ns | 4,470 ns | 80 ns | most of it: classes, try, while |
| **Monty**, web | 3.07 MB | 19–24 ms | 2,400–3,800 ns | 6,300–11,700 ns | 80–144 ns | |
| **Pyodide** 0.28.3, web only | 5.4 MB | 1.7–2.6 s | 140–240 ns | 810 ns | 55–82 ns | all of CPython 3.13 |
| **Rhai** 1.27, native | | 1.3 ms | 41 ns | 444 ns | 193 ns | no, Rust-like |
| **Rhai**, web | 0.56 MB | 15–25 ms | 44–70 ns | 600–670 ns | 260–320 ns | |
| Plain Rust, web, for scale | | | | 3 ns | 1 ns | |

The viewer's own wasm is 319 KB gzipped, so every interpreter here is larger than the rest of the
engine. Next to the scenes, though, it is small: the cove streams up to 209 MB. Size matters for
time to first frame, not for bandwidth.

What each one means:

- **Pyodide** is real CPython with its whole ecosystem, but it is the wrong fit for an engine.
  - It takes 1.7–2.6 s to start, and downloads 5.4 MB (2.8 MB of wasm, 2.4 MB of standard
    library).
  - It is a separate Emscripten module, so every call between a script and the engine goes
    through JavaScript twice.
  - It has no native counterpart: natively the engine would embed CPython instead, a different
    runtime with different packaging.
  - It has no sandbox. A script can reach everything the page can, through its `js` module.
  - It could still serve later as an optional console for tools, never as the game's language.
- **RustPython** runs nearly all of Python, with reasonable calls (233 ns on the web).
  - It is the largest download at 4.15 MB, even with no standard library, and takes 77 ms to
    start.
  - Its release did not build as published: one of its dependencies broke compatibility in a
    patch release, and had to be pinned back by hand.
  - Its embedding API offers no limit on running time that we could find, so an endless loop
    would hang the page.
- **Monty** is the most promising real Python. It runs classes, dataclasses, `try`, `while`,
  `global` and `math` (each checked), though not yet `yield`. It has time and memory limits, and
  starts in under a millisecond natively.
  - Its interface between Rust and scripts is built for an AI agent's tool calls, not a game
    loop. Each call from Rust compiles a small snippet, at 2.4 µs or more on the web. Each call
    from a script back into Rust suspends the interpreter and resumes it, at 2–4 µs on the
    web. That is 70 to 130 times Starlark's cost.
  - It is 3 MB, calls itself experimental, and needs Rust 1.96, newer than the project's
    pinned 1.95.
- **Starlark** is the fastest at every crossing and has the best tools.
  - Calls in both directions cost tens of nanoseconds, a thousand ticking scripts cost about
    0.09 ms a frame on the web, and on the web its loops are as fast as Pyodide's CPython.
  - It has optional type annotations with a checker, and a language server (the `starlark_lsp`
    crate) for editor support.
  - Its sandbox is complete. There is no I/O, no clock and no randomness unless the engine
    provides them, and `Evaluator::set_check_cancelled` lets the engine stop a script that
    runs too long.
  - It is not full Python. Checked against `starlark-rust` 0.14: no `class`, `while`, `try` or
    `global`. Recursion, lambdas, list comprehensions, type annotations and (with a dialect
    flag) f-strings all work. Values made at load time are frozen: a script cannot keep state in
    a global.
- **Rhai** is small and mature, but its syntax is Rust-like rather than Python-like, and it was
  slower than Starlark at every call and loop measured. It is here for contrast.
- **Luau**, for contrast, was not measured. Its VM is C++, which does not link into a
  `wasm32-unknown-unknown` build. It would need its own Emscripten module and JavaScript glue,
  with Pyodide's double crossing. It also isn't Python-like.
- **A language of our own**, Python syntax compiled to the engine's own bytecode (or to wasm),
  is the GDScript path. With static types it could beat every number above and be small. But it
  means writing a parser, type checker, VM, error messages, debugger and editor support: years
  of work, as Godot and Epic (Verse) both found. Compiling to wasm has a second problem: the
  browser can run wasm, but the native build would need a wasm runtime such as Wasmtime as well.

### The recommendation: Starlark

**Starlark, through `starlark-rust`.**
- It is Python to read and write: `def`, indentation, lists, dicts, comprehensions, `if`, `for`.
- It is the fastest at the crossings a game makes all the time.
- It is fully sandboxed, can be stopped, and has types and an editor language server.
- It behaves the same natively and on the web.

Its gaps fit the framework, and are filled by it:

- **No classes:** an actor's class is its `.actor` file. The script is the functions that
  handle its events, and `self` is a handle to the actor.
- **No mutable globals:** an actor's state lives in its components, as typed `property` values
  declared in its `.actor` file, and in a per-actor `self.vars` for anything else. This is also
  what makes hot reload keep state: none of it lives in the script.
- **No `while`:** loops run over ranges, lists and queries, so every handler finishes. The
  cancellation check catches the rest.
- **No `try`:** a failing handler stops, its error and line go to the on-screen log, and the
  actor carries on with its next event.

```python
# scenes/cove/gate.star
def on_begin(self):
    self.vars.open = False

def on_interact(self, who):
    if self.locked:
        say(who, "The gate is locked.")
        return
    self.vars.open = not self.vars.open
    target = 90.0 if self.vars.open else 0.0
    tween(self, "rotation.yaw", target, self.open_seconds)

def on_hour(self, hour):
    if hour == 21:
        self.locked = True   # the fort closes at night
```

**How scripts run:**
- **Loading.** Each script file is parsed, type-checked, run once and frozen. Its handler
  functions become frozen values shared by every actor that uses the script.
- **Each frame,** events are handled in a fresh temporary heap, which is dropped when the frame
  ends. So nothing a frame allocates can outlive it, and there is no garbage collector to tune.
  Step 3 confirms that frozen handlers run this way at the measured cost.
  Anything worth keeping goes into `self.vars` or a component, where it is stored as an engine
  value (number, bool, string, vector, entity, list, dict).
- **Hot reload.** When a file changes (a file watcher natively, a reload button on the web), it
  is reloaded and the actors switch to the new handlers on their next event. State stays,
  because it lives in components. If the new file fails to load, the old one keeps running and
  the error is shown.
- **The engine's API** is written once in Rust, as a table of functions with typed arguments.
  Starlark's bindings are generated from it, and so is a `.pyi`-style stub for editors. If
  Monty's interface becomes fit for a game, or Solstice ever grows its own language, only the
  generated layer changes; the API and the game's scripts stay put.

## Physics

**Rapier**, version 0.36, in `solstice-physics`.
- It is pure Rust, so it is the same code natively and on the web, inside the one wasm module.
- It has rigid bodies, joints, sensors, a kinematic character controller, ray and shape casts,
  and cross-platform determinism as an option.
- The alternatives don't fit. Jolt and PhysX are C++ and would need Emscripten, with the same
  problem as Luau. Avian is built around Bevy.

### What was measured

Boxes dropped in a jumbled pile on the ground. The table gives the mean step over the first two
seconds, while most are still moving (afterwards they fall asleep and cost almost nothing), and
1,000 raycasts down into the pile:

| Boxes | Native | Web | Web with SIMD | Worst step, web with SIMD | 1,000 rays, web |
|---|---|---|---|---|---|
| 100 | 0.16 ms | 0.59–0.66 ms | 0.35–0.37 ms | 11 ms (V8 still compiling) | 0.4–0.5 ms |
| 500 | 0.87–0.89 ms | 2.5–2.6 ms | 1.13–1.17 ms | 4.5–4.9 ms | 0.4–0.5 ms |
| 1,000 | 1.85–2.02 ms | 5.1–5.3 ms | 2.35–2.44 ms | 9.4–10.3 ms | 0.4 ms |

- **SIMD halves the web cost.** Built with `-C target-feature=+simd128`, so the compiler may use
  wasm's vector instructions, the web build ran 500 boxes only about 30% slower than native.
  Every browser that has WebGPU also has wasm SIMD, so the gameplay build turns it
  on. Size barely changes: 761 KB with it, 753 KB without.
- **Worst steps** come when a whole pile lands at once, and from V8 compiling on the first
  frames. Real scenes spread out their impacts. The first-frame cost can be paid behind the
  loading screen, by stepping an empty world.

### Budget

On the web, a laptop with integrated graphics has perhaps half the 9800X3D's single-thread
speed. To hold 60 fps there, gameplay gets 4 ms of each frame's CPU, and the renderer keeps the
rest:

| Part | Integrated-GPU laptop, web | This machine, web |
|---|---|---|
| Physics | 1.5 ms: about 300 moving bodies, any number asleep | 0.75 ms |
| Animation | 1.0 ms: about 20 characters at full rate | 0.5 ms |
| Scripts | 1.0 ms | 0.5 ms |
| World, events, extract | 0.5 ms | 0.25 ms |

Every step that adds a system measures it against its line here, and adds it to the profiler's
CPU sections.

### Colliding with the scene

- **The bake makes collision.** For each static instance it takes a coarse cut of the mesh's
  cluster DAG, at about 2–5 cm of error, as a triangle mesh. Small props get convex hulls
  instead. Terrain becomes a heightfield. All of it goes into a new tagged section, `COLL`, so
  the shapes the player touches match what is drawn, without the full detail.
- **Moving actors** use primitives or convex hulls, from their `.actor` files.
- **The character controller** is Rapier's kinematic one: steps, slopes, snapping to the ground,
  pushing bodies.
- **Interpolation.** Rendering interpolates each body between its last two steps.
- **Determinism.** Rapier's `enhanced-determinism` feature is measured in step 5, and kept if
  its cost fits the budget, so that tests and replays give the same result natively and on the
  web.
- **Versions.** Rapier 0.36 uses `glam` 0.33. Solstice is on 0.30, and moves to 0.33 so the two
  share one copy.

### Floating on the sea

A ship that floats must ride the waves that are drawn. The waves exist only on the GPU, but
`spectrum.rs` already reproduces them exactly on the CPU, seeds and all, for its tests. So the
sea gets a CPU height query:
- a direct sum over the few hundred most energetic waves of the largest cascade, which hold
  nearly all the height that moves a hull;
- shrunk in the shallows by the same `shoal()` the surface uses, with the seabed depth read
  from a small CPU copy of the seabed map.

Buoyancy samples the query at a handful of points on each hull, and pushes each one up by the
water it displaces, as Sea of Thieves does. It is deterministic and has no delay. Reading
heights back from the GPU was the other option; it would arrive frames late and make the
physics depend on the GPU's timing.

## Rigging and animation

### Content

- Rigs and clips are made outside the engine, in Blender, and imported as glTF skins and
  animations. The importer already parses glTF, and today only warns that it ignores them.
- **Sources.** The cove's models come from Poly Haven, which has no characters. CC0 rigged
  characters with animation sets exist (Quaternius, for example). Which to use is your decision.

### Skinning in virtual geometry

Skinned meshes go through the same pipeline as everything else: the cluster DAG, streaming, both
rasterizers, the visibility buffer and resolve. They do not get a separate path.

- **Weights.** Each vertex gets up to four bone influences, as 8-bit bone indices into its
  cluster's own list of bones and 8-bit weights: 8 more bytes per vertex. They are stored only
  for skinned meshes, in their own part of each page.
- **Building the DAG** in bind pose:
  - clusters are grouped by the bones they use as well as by position, so each cluster's list
    stays short, mostly 8 bones or fewer;
  - simplification keeps the weights: collapsing an edge blends them, and the quadric error
    counts a change of weights, so joints aren't simplified away;
  - borders between clusters stay locked, as now, so a shared vertex keeps exactly the same
    weights on both sides and skins to exactly the same place, and no cracks open.
- **Culling.** Each cluster stores its bounding sphere in each of its bones' bind space. Each
  frame, its bounds are the union of those spheres, moved by the current bones, which is cheap
  for a list of 8. LOD error is measured in bind pose and scaled by the largest bone scale, with
  a margin for bending.
- **Rasterizing.** Both rasterizers, resolve and the shadow pass call one shared function,
  `skinned_position`. It applies linear blend skinning from a buffer of bone matrices: this
  frame's, and last frame's for motion vectors. Normals are skinned with the same matrices.
  - Vertices are skinned where they are used, as Unreal does: there is no pre-skinned copy, so
    no extra memory, and it works whatever the LOD cut and whatever has streamed in.
  - If the cost proves too high, the fallback is a compute pass that skins only the visible
    clusters into a scratch buffer each frame. Step 8 measures both.
  - Dual-quaternion skinning, for shoulders and wrists that would otherwise pinch, is an option
    to judge on real characters.
- **Moving things in the rest of the renderer:**
  - *Shadows:* cascades keep their cache for static geometry. Each frame, moving instances are
    drawn over a copy of it.
  - *GI voxels:* moving things are left out at first. They receive GI and appear in screen-space
    GI, but don't occlude or bounce light in the voxels.
  - *Seabed and water-line maps:* moving things are left out.

### The animation runtime

`solstice-anim`, on the CPU, in Rust:

- **Skeletons and clips.** Clips are resampled to 30 Hz and quantised, as in the Animation
  Compression Library: rotations as their three smallest components, translations within each
  track's range. Bones are sampled in batches of four (structure-of-arrays) for SIMD.
- **Blending:**
  - linear and additive blends;
  - per-bone masks, so the upper body can aim while the legs run;
  - 1D and 2D blend spaces, by speed and direction;
  - state machines with cross-fades, written in a small text format like the others;
  - sync groups, so blended walk and run cycles keep their feet in step.
- **Root motion,** and **notifies**: events at set times in a clip, such as a footstep or a
  sword's hit frame, delivered to scripts like any other event.
- **IK:**
  - two-bone analytic IK, with a pole vector, for legs and arms;
  - foot placement, with a raycast down from each foot and the pelvis lowered to suit;
  - look-at and aim, with limits;
  - FABRIK for longer chains, such as tails and ropes.
- **Secondary motion:** spring chains (Verlet), with sphere and capsule colliders, for hair,
  straps and ropes.
- **Animation LOD:** distant or small characters update less often, as Unreal's budget allocator
  does, within the 1 ms budget.
- **Ragdolls,** from Rapier joints, blended in and out of animation, come in a later step.

## Gameplay and the world around it

The renderer already eases the time of day, the weather and the sea. Gameplay gets them as one
`Environment` in the world, and sets targets, not values each frame:

```python
def on_begin(self):
    env.time.day_length = 1200          # a 20-minute day
    env.weather.set("fair", over=0)

def on_storm_bell(self):
    env.weather.set("storm", over=90)   # the renderer eases it in
    env.sea.set("ocean", over=180)      # and the sea builds more slowly

def on_hour(self, hour):
    if hour == 19:
        for lamp in find("lantern"):
            lamp.light.on = True
```

- **Reading:** `env.sun_direction`, `env.is_night`, `env.wind`, `env.weather.rain`,
  `env.sea.height_at(x, z)` (the CPU query above), and later how wet a surface is.
- **Events:** `on_hour`, `on_weather_changed`, `on_sunset`, `on_sunrise`.
- **New renderer APIs** that gameplay needs, all general:
  - per instance: moving it, hiding it, and scaling its glow, for opening gates and lit
    windows;
  - lights switched on and off, and moved. GI is fully dynamic, so it follows.
- **Who is in charge.** While a game runs, it owns the environment. The web page's time and
  weather controls become debug overrides, marked as such, and hand control back when released.
  A scene file can still set the starting values.

## Viewers

- **Native:** `viewer scenes/cove.solstice --play` runs the scene's game, if it has one, and
  watches its scripts for changes.
- **Web:** the page offers *Play* on scenes with gameplay, and loads the gameplay build. It
  captures the pointer and the keyboard, and shows script errors in a small overlay.
- **Input actions** are a text file mapping actions to keys, mouse, gamepad and touch, so the
  same game works on a phone.

## Steps

Each step is its own PR, with mutation-tested tests, measured costs (natively and on the web),
clippy clean on native and wasm32, and a review by a separate AI agent. It is merged only with
your go-ahead, and nothing is published without asking.

1. **Things that move.** The renderer API for moving instances, with `prev_world` filled in;
   hiding instances and scaling their glow; moving instances drawn over the cached shadow
   cascades; moving things left out of the GI voxels and the seabed map.
   - Demo: the cove's ship bobs at anchor, driven by plain Rust for now.
   - Tests: a moved instance's motion vectors and shadow follow it; cached shadows of static
     geometry stay as they were; nothing static is redrawn.
2. **The world.** `solstice-world`: entities and components on `hecs`, the transform tree, tick
   groups, events, timers and the fixed clock. Headless.
   - Tests: tick order; event order and replay; the clock under slow and uneven frames.
3. **Scripting.** `solstice-script` with Starlark: actor scripts, properties and `self.vars`,
   events, timers, the engine API table and its generated bindings, hot reload, time limits, and
   errors with file and line. The benchmark above moves into the repository, so its numbers can
   be checked again.
   - Tests: hot reload keeps state; a runaway script is stopped and reported; a broken reload
     keeps the old script running.
4. **Scenes, actors and play mode.** `.actor` files, `actor` lines in scene files and their
   tagged section, input actions, a player camera, and *Play* in both viewers, with the second
   web build.
5. **Physics.** `solstice-physics`: collision baked into `COLL`, bodies, sensors, the character
   controller, a debug view of colliders, wasm SIMD, and determinism measured.
6. **The sea in physics.** The CPU wave query, checked against the GPU's waves, and buoyancy: the
   ship floats, and a rowing boat can be pushed about.
7. **Gameplay and the environment.** The `Environment` API and its events; lights switched by
   gameplay; viewer controls as overrides.
8. **Skinned virtual geometry.** Importing skins, weights through the DAG, per-cluster bone
   lists and bounds, and skinning in both rasterizers, resolve and shadows, with motion vectors.
   - First a character in its bind pose, compared with the static import; then one looping clip.
   - The cost is measured, against the scratch-buffer fallback.
9. **Animation.** Clips and their compression, blends, blend spaces, state machines, root motion,
   notifies, and animation LOD.
10. **IK and secondary motion.** Two-bone IK, foot placement, look-at, FABRIK, and spring chains.
11. **A playable cove.** Walk a character through the fort, open the gate, row out to the ship,
    and see the lanterns lit at dusk. README, screenshots and costs on the RTX 5080 and on the
    integrated-GPU tier. Published only with your go-ahead.

Steps 1–4 come first in that order. After them, physics (5–6) and characters (8–10) depend only
on 1–4, so either can go first, as you prefer. Step 7 can go anywhere after 4.

## Risks

- **Starlark isn't Python.** People who know Python will reach for `class`, `while` and `try`.
  The framework covers each: actor files, loops over ranges and queries, and errors that stop a
  handler. But it must be taught, with good error messages. The bindings are generated from one
  API table, so changing language later (to Monty, if its interface grows fit for games) changes
  the bindings, not the game's scripts' meaning.
- **Download.** The gameplay build adds about 1.3 MB for Starlark and 0.8 MB for Rapier, gzipped,
  to the viewer's 0.3 MB. Hence the second build, so that scenes without gameplay don't pay for
  it.
- **One thread on the web.** Everything in the budget shares one thread with the renderer's own
  CPU work. Physics on wasm, even with SIMD, ran about 1.3 times slower than native at 500 boxes,
  and its first frames are slow while V8 compiles.
- **The renderer assumes a static world.** Shadow caching, the GI voxels, the seabed map and the
  wetness water-line map all do. Step 1 handles each one, and anything new must say how it treats
  moving things.
- **Skinned virtual geometry is new ground.** Unreal only shipped it in 5.5, and has published
  little about it. Step 8 starts with a bind-pose comparison, so any crack or LOD error shows up
  before animation depends on it.
- **Content.** Characters and animations need a CC0 source, or yours.
- **Determinism across native and the web** depends on Rapier's option and on avoiding
  platform-specific maths in gameplay code. Tests compare the two builds.
- **Dependencies move.** Starlark and Rapier are both pre-1.0. Both are pinned, and upgraded on
  purpose, as the toolchain is.
