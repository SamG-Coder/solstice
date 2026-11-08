# Gameplay: design

A design for making Solstice's worlds playable: things that move and collide, characters that
are skinned and animated, and game logic written in a Python-like language. The same game runs
in the browser and natively. This document is the plan, written before any of it is built, and
it will be kept up to date as each step lands.

## Goals

- **Built like Unreal.** A world of actors made of components, ticked in a fixed order of tick
  groups, talking through events. Engine systems are written in Rust where speed matters, and
  game logic is written in scripts. Unreal splits its work the same way: C++ beneath, with
  Blueprint above (and Verse, so far only in Unreal Editor for Fortnite).
- **Python-like scripting**, the same language natively and on the web, with:
  - hot reload while the game runs, keeping each actor's state;
  - a sandbox: a script can reach only what the engine hands it, and a runaway script is
    stopped rather than freezing the page;
  - errors that name the file and line.
- **Modular.** Each part is its own crate and can be tested without a GPU. The renderer knows
  nothing about gameplay: it gains a few small, general APIs (things that move, skinned meshes),
  and nothing more. A scene without gameplay loads and costs exactly what it does today.
- **Fast on the web, integrated GPUs included.** The web build has one thread. Wasm threads
  need the page to be cross-origin isolated, and GitHub Pages cannot send the headers for that.
  The usual workaround, a service worker that adds them, reloads the page on the first visit and
  blocks cross-origin fetches, and the scenes are fetched from their own Pages sites. The budget
  below assumes one thread, and every step measures against it.
- **Deterministic simulation.** Physics, script ticks and gameplay events advance in fixed
  steps, from input recorded per step. So a test can run a scene for N steps and check where
  everything ended up, a replay gives the same result, and the native and web builds agree.
  Only what is drawn (animation poses, the camera) follows the frame rate.
- **Gameplay drives the world.** Scripts can set the time of day, the weather and the sea, and
  react to them. The renderer already eases the weather and the ocean; gameplay sets where they
  should go, and eases the rest itself.

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
  - wetness (`docs/wetness.md`, `ocean/waterline.rs`) remembers how high the water has stood
    at each spot, in a map fixed in the world. It draws no geometry, but the wetness it gives
    belongs to a place, not to an object: a hull that sails away leaves its wet band behind.

  Each of these has to be told when something moves, or ignore things that move.
- **Two-phase occlusion culling** (`vg/culling.rs`) already copes with motion: anything the old
  depth hid wrongly is retried against the new depth in the same frame.
- **Vertices** are 16 bytes on one 16-bit grid per mesh: position, a packed normal, UV, and one
  spare 16-bit value (`x y z nx ny u v _`, `solstice-asset/src/gpu.rs`). There are no bone
  weights. The importer warns that it ignores skins and animations
  (`solstice-import/src/lib.rs`), and imports models in their rest pose.
- **Dynamic attributes:**
  - `set_weather(Weather, seconds)`, eased by the renderer over that many seconds, a weather
    cycle (`set_weather_changing`), and reports from real observations (`Weather::observed`);
  - `set_ocean(Option<Ocean>, seconds)`, eased the same way;
  - `set_sea_level(level)`, which takes effect at once;
  - `TimeOfDay`: `advance(hours)`, which just adds hours, and from it the sun, moon, stars and
    `Environment`;
  - the weather and the ocean have text forms (`.weather`, `.ocean` files) with `parse` and
    `to_text`. The time of day is read from the viewers' flags (`parse_time` in
    `apps/viewer/src/lib.rs`).
- **The viewers** drive these directly. The native viewer has command-line flags, and the web
  page calls `set_time`, `set_weather` and the rest from JavaScript (`apps/viewer/src/web.rs`,
  `apps/viewer/web/index.html`). The camera is a free-flying viewer camera. There is no input
  system beyond it.
- **The ocean exists only on the GPU.** `spectrum.rs` has a CPU reference: the spectrum, the
  seeded starting amplitudes (`pcg`, `gaussian_pair`, `initial_amplitude`) and `direct_sum`.
  Tests use it to check that the GPU draws exactly the waves the reference does (`waves.rs`).
  Waves shrink over the seabed in `shoal()`, which exists only in WGSL
  (`shaders/ocean/seabed.wgsl`). Nothing reads wave heights back during play.
- **Scene files** take optional tagged sections, such as `SEA `, which older readers skip. So
  actors, collision and scripts can be added without a new file version.

## References

- **Unreal Engine.** The gameplay framework: `UWorld`, `AActor`, `UActorComponent` and
  `USceneComponent`, tick groups (`TG_PrePhysics`, `TG_DuringPhysics`, `TG_PostPhysics`,
  `TG_PostUpdateWork`), tick intervals, timers and delegates, GameMode, PlayerController and
  Pawn, and Enhanced Input. Also Mass, Unreal's ECS for crowds; Blueprints, a visual language
  run by a VM and slower than C++, which is why Epic advises keeping heavy loops out of
  Blueprint ticks; Verse, a language Epic designed itself, so far only in Unreal Editor for
  Fortnite; and the Animation Blueprint, blend spaces, Control Rig, IK Rig, IK Retargeter and
  the Animation Budget Allocator.
- **Nanite skinning.** Unreal 5.5 added experimental support for drawing skeletal meshes
  through Nanite. Epic's documentation on Nanite foliage, which is animated by the same
  skinning, says that bounds can be computed from the skinning matrices, so the hardware
  rasterizer can be used. Epic has published little more, so the skinning design here is our
  own, and step 7 proves it out before anything depends on it.
- **Godot.** A tree of nodes, signals for events, and GDScript: a Python-like language written
  for the engine, with optional static types and hot reload. It is the clearest example of an
  engine building its own language, and of what that costs: years of work on the language, its
  tools and its editor support.
- **Starlark** (Bazel's configuration language) and its Rust implementation `starlark-rust`,
  from Meta. **Monty**, Pydantic's sandboxed Python subset in Rust. **RustPython**. **Pyodide**,
  CPython compiled to wasm. **Rhai**. **Luau**, Roblox's sandboxed Lua.
- **Rapier**, from Dimforge: rigid bodies, joints, character controllers and scene queries in
  pure Rust. It compiles to `wasm32-unknown-unknown`, and Dimforge also publishes it to npm
  for JavaScript games.
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
- **Two web builds**, chosen on 8 November 2026. The viewer stays as it is (319 KB gzipped
  today). A second build, with gameplay, is loaded only for scenes that have it, when *Play* is
  pressed.

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
mesh ../../target/scan/large_iron_gate/large_iron_gate.gltf 1.15   # a model, and its scale
collider box 1.7 2.4 0.1             # from step 5
script gate.star
property open_seconds float 2.5      # editable per placement in the scene
property locked bool false
```

A scene places it as it places models, at x, y, z and a yaw in degrees, with its own settings:
`actor cove/gate.actor 20.7 0 0 90 locked=true`.

**Which ECS.** Both candidates were measured, by moving 10,000 entities by their velocity 100
times, natively and on the web, set up as for the scripting benchmarks below (one run each):

| | Download (gz) | Native, per entity | Web, per entity |
|---|---|---|---|
| `hecs` 0.11 | 31 KB | 0.90 ns | 1.58 ns |
| `bevy_ecs` 0.20, a plain query | 160 KB | 0.67 ns | 1.85 ns |
| `bevy_ecs` 0.20, a system in a schedule | | 0.92 ns | 1.28 ns |

They run at about the same speed:
- `hecs` is small and stable;
- `bevy_ecs` brings schedules, change detection and observers, and breaks its API every few
  months.

The framework only needs fixed tick groups on one thread, which is a few hundred lines on top of
`hecs`. So the choice, made on 8 November 2026, is `hecs`, with Solstice's own framework over
it.

### A frame

```text
 1. Input        gather keys, mouse, pads and touch; map them to actions
 2. Fixed steps  0 to 3 steps of 1/60 s, each one:
      a. Start        the step's input is taken; timers due on it deliver their events
      b. PrePhysics   Rust systems, root motion, then script ticks for the actors that tick
      c. Physics      one Rapier step; contacts and triggers are queued
      d. PostPhysics  hits and overlaps delivered to scripts
      e. Transforms   every entity's place in the world, keeping the step before
 3. Animation    state machines read gameplay's values; sample, blend, IK
 4. Late         the camera follows
 5. Extract      changed transforms, bone palettes, lights and environment go to the renderer
```

As built (step 2, `crates/solstice-world`): `Game::frame` runs items 1 to 4 of this list.
Extraction (item 5) comes with roadmap step 4, *Scenes, actors and play mode*. Items 3 and 4
are the per-frame tick groups `Animation` and `Late`.

- **Fixed steps.** Everything that decides what happens runs inside the steps, at 60 Hz from an
  accumulator: events, script ticks, Rust gameplay systems and physics. Input is gathered per
  frame but handed out per step, and recorded that way for replays. A frame runs at most three
  steps; past that, the game slows down rather than falling behind in a spiral.
- **Per frame.** Animation, the camera and extraction run once per frame, at the frame's own
  time. They only affect what is drawn, so they need not be deterministic.
- **Interpolation.** Rendering draws each body between its last two physics states, at the
  accumulator's fraction. Each instance's `prev_world` is the transform it was actually drawn
  with last frame, kept by the extract step, not the previous physics state: the two differ
  whenever a frame runs no step or several.
- **Root motion** is sampled in PrePhysics, for the step's length, and drives the character
  controller in that step's physics, as Unreal does for characters. The full pose is still
  sampled per frame.
- **Most actors never tick.** As in Unreal, ticking is opt-in, with an interval. Scripts mostly
  react to events: an overlap, an input action, a timer, the hour changing.
- **Events are queued** and delivered between tick groups. A handler never runs halfway through
  another system, so the order is always the same and replays give the same result.
  - Each event is stamped with the step and group that sent it. Every fixed-step group sees it
    exactly once, within a step: what was sent before it in this step, and what was sent from
    its own place on in the step before. So a script's event in PrePhysics reaches handlers in
    PostPhysics of the same step, and in PrePhysics of the next.
  - Events sent before a step (while setting up, or by timers as the step starts) are seen by
    all of that step's groups.
  - The per-frame groups see everything sent since they last ran, once, however many steps the
    frame held. They send no events, set no timers and spawn nothing, and that is checked: they
    run as often as frames come, which differs from run to run, and a replay doesn't run them.
    They may write only what is drawn (poses, the camera).
- **Input** is the frame's values, plus the actions pressed since the last step, so a tap
  between two steps isn't lost. The per-frame groups also see the frame's own input, for a
  camera that turns even in frames with no step.
- **Replays are exact on the same build and platform.** Native and web builds may round some
  maths differently, such as sines in `glam` and `libm`.
- **Spawning and destroying** are deferred to the end of the tick group that asks for them.
- **Timers** count steps, not seconds, so they fire on the same step in every run.
- **Teleports** mark the entity `Snap`, so it is drawn at its new place at once, not sliding there.
- **Later steps will need:** animation notifies (footsteps, hit frames) worked out in the fixed
  step, alongside root motion, since the per-frame groups can't send events; and, for physics,
  places worked out again before the physics group when something moved earlier in the step.

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
  Starlark 0.14, Rapier 0.36 and `hecs` 0.11 also build with the project's pinned Rust 1.95.
- **Web:** the same code built for `wasm32-unknown-unknown`, run in Node 24 on the same machine.
  Node uses V8, Chrome's engine.
- **Sizes:** the wasm after `wasm-bindgen`, with name sections removed, gzipped. There is no
  `wasm-opt` pass, which would shrink them a little. Pyodide's are its release files, gzipped.
- **Runs:** each figure is the best of three, and each benchmark was run two or three times;
  where the runs differed, the table gives the range. The code is in `tools/gameplay-bench`,
  with how to run it.
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
    web. A whole `tick` costs 70 to 130 times as much as Starlark's.
  - It is 3 MB, calls itself experimental, and needs Rust 1.96, newer than the project's
    pinned 1.95.
- **Starlark** is the fastest at every crossing and has the best tools.
  - Calls in both directions cost tens of nanoseconds, a thousand ticking scripts cost about
    0.09 ms a frame on the web, and on the web its loops are as fast as Pyodide's CPython.
  - It has optional type annotations with a checker, and a language server (the `starlark_lsp`
    crate) for editor support.
  - It is sealed off from the system: there is no I/O, no clock and no randomness unless the
    engine provides them. The engine can cap each call's work with `set_max_tick_count`, which
    counts steps and so stops a script at the same point natively and on the web, and its memory
    with `set_max_heap_size`. `starlark-rust` does not claim to be safe against hostile code,
    which is fine here: the scripts ship with the scene, written like the rest of it, and are
    not uploads from strangers.
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

### The choice: Starlark

**Starlark, through `starlark-rust`.** Chosen on 8 November 2026, from the measurements above.
- It is Python to read and write: `def`, indentation, lists, dicts, comprehensions, `if`, `for`.
- It is the fastest at the crossings a game makes all the time.
- It is sealed off from the system, can be stopped, and has types and an editor language
  server.
- It behaves the same natively and on the web.

Its gaps fit the framework, and are filled by it:

- **No classes:** an actor's class is its `.actor` file. The script is the functions that
  handle its events, and `self` is a handle to the actor.
- **No mutable globals:** an actor's state lives in its components, as typed `property` values
  declared in its `.actor` file, and in a per-actor `self.vars` for anything else. This is also
  what makes hot reload keep state: none of it lives in the script.
- **No `while`:** loops run over ranges, lists and queries. Recursion is allowed, so a handler
  can still run away; the step cap stops it, and the error is reported like any other.
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
  value (number, bool, string, vector, entity, list, dict). Lists and dicts there are owned by
  the engine and changed in place, so `self.vars.path.append(p)` keeps `p`.
- **Hot reload.** When a file changes (a file watcher natively, a reload button on the web), it
  is reloaded and the actors switch to the new handlers on their next event. State stays,
  because it lives in components. If the new file fails to load, the old one keeps running and
  the error is shown.
- **The engine's API** is written once in Rust, as a table of functions with typed arguments.
  Starlark's bindings are generated from it, and so is a `.pyi`-style stub for editors. If
  Monty's interface becomes fit for a game, or Solstice ever grows its own language, only the
  generated layer changes; the API and the game's scripts stay put.

**As built (step 3, `crates/solstice-script`):**
- **Handlers.** `on_begin(self)` runs once, the first step an actor has the script.
  `on_tick(self, dt)` runs every step, but only for scripts that have one. `on_<name>(self, ...)`
  runs for each `ScriptEvent` named `<name>`, sent to the actor or to every actor, by a script
  or by Rust.
  - Each step they run in `PrePhysics`: beginnings, then events in the order they were sent,
    then ticks, each in the world's order of entities.
  - An event a script sends, even to itself, is handled the next step: one step, 1/60 s, of
    latency between one actor and the next.
- **`self`** is a copy of the actor on the call's heap:
  - its properties, which can be set but not added;
  - `self.vars`, where plain data can be kept: numbers, strings, `vec3`s, entities, and lists
    and dicts of those, at most 64 deep and 100,000 values, and never containing themselves;
  - `self.position` and `self.yaw`, written back to its `Transform`, local to its `Parent` if it
    has one; an actor without a `Transform` gets one when it first moves, and both must stay
    finite;
  - `self.entity`, which can be a dict key.

  Lists in `self.vars` are changed in place within a call, and kept.
- **A failing handler changes nothing but the log.** Its copy is thrown away, and what it asked
  of the world (events to send, timers to set or cancel) is held back until it succeeds, so an
  actor is never left half-changed. Its log lines are kept, to show why it failed; what failed,
  with file and line, goes to the host's errors.
- **Timers** set by `after` and `every` belong to the actor: their handles are its own, and
  `cancel` refuses, returning `False`, another actor's. `begin` and `tick` are the engine's own
  events, and can't be sent.
- **Loading.** Each script is parsed, run once and frozen, and its handlers are shared. A script
  that fails to load, or whose `on_*` names aren't functions, leaves the old one running. Each
  call gets a temporary heap of its own, dropped when the call ends; sharing one heap across a
  step was tried, and was slower. Type checking is left to editors for now.
  - Reloads aren't part of a `Recording`: a replay runs the scripts loaded when it starts.
- **Limits.** Each call, and each load, may do 100,000 of Starlark's steps of work, and hold
  16 MiB.
  - Counting work, not time, stops a runaway script at the same point natively and on the web:
    tested, at the same 100 lines of its log, plus one counting those dropped, on both.
  - It stops in about 4 ms on the web: one hitch, not a hang.
  - The memory limit is best effort. Starlark checks it now and then, and the host once more as
    the call ends, from Starlark's own count of what it allocated. That count is of values,
    which are larger on 64-bit than on wasm32, so a script near the limit may pass on the web
    and fail natively.
  - A call keeps 100 lines of log, of up to 1,000 characters each.
- **The API** is the one `#[starlark_module]` in `api.rs`: `vec3`, `send`, `after`, `every`,
  `cancel`, `find`, `input`, `held`, `pressed`, `now` and `log`. Starlark's own documentation,
  `api_reference()`, renders it for people and editors, in place of a hand-written stub.
- **Cost** (`tools/gameplay-bench/b_script`, 9800X3D):
  - about 1.1 to 1.3 µs a ticking actor on the web, and 1.9 to 2.4 µs natively, most of it
    setting up each call;
  - so the 1 ms scripts line of the budget holds about 400 actors ticking every step on an
    integrated-GPU laptop. Most actors don't tick: they wait for events.
  - Starlark adds 1.4 MB, gzipped, to the gameplay web build.
- **No `unsafe` code.** A frozen handler is called from a temporary heap by allocating it on the
  call's frozen heap, which keeps its own heap alive for the call.
- **`Name`** moved into `solstice-world`, since the world's own tools will find entities by it.

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
rest. At 60 fps a frame runs one fixed step:

| Part | Integrated-GPU laptop, web | This machine, web |
|---|---|---|
| Physics, per step | 1.5 ms: about 300 moving boxes, any number asleep | 0.75 ms |
| Scripts, per step | 1.0 ms | 0.5 ms |
| Animation, per frame | 1.0 ms: perhaps 20 characters at full rate, an estimate until step 8 measures it | 0.5 ms |
| World, events, extract | 0.5 ms | 0.25 ms |

- **The per-step lines are paid per step.** A device that can't hold 60 fps runs two steps a
  frame at 30 fps, and so pays them twice. The budget is set so the integrated-GPU tier holds 60.
- **Boxes are a best case.** Contacts against the scene's triangle meshes cost more than box on
  box. Step 5 measures the cove's own collision.
- Every step that adds a system measures it against its line here, and adds it to the
  profiler's CPU sections.

### Colliding with the scene

- **The bake makes collision.** For each mesh it takes a coarse cut of the cluster DAG, at
  about 2–5 cm of error, as a triangle mesh. Small props get convex hulls instead. Terrain
  becomes a heightfield. So the shapes the player touches match what is drawn, without the full
  detail.
  - Shapes are made once per mesh and shared by every instance of it (Rapier's `SharedShape`).
  - They are stored in the file's data area, where streamed pages are, and listed by a small
    `COLL` section in the manifest. The manifest is read whole when a scene loads, so the shapes
    themselves stay out of it: they are fetched only when *Play* is pressed, and a scene that is
    only looked at never downloads them.
- **Moving actors** use primitives or convex hulls, from their `.actor` files.
- **The character controller** is Rapier's kinematic one: steps, slopes, snapping to the ground,
  pushing bodies.
- **Determinism.** Rapier's `enhanced-determinism` feature is measured in step 5, and kept if
  its cost fits the budget, so that tests and replays give the same result natively and on the
  web.
- **Versions.** Rapier 0.36 uses `glam` 0.33 (through `glamx`). Solstice is on 0.30, and moves
  to 0.33 so the two share one copy.

### Floating on the sea

A ship that floats must ride the waves that are drawn. The waves exist only on the GPU, but
`spectrum.rs` already reproduces their starting amplitudes exactly on the CPU, seeds and all
(`initial_amplitude`, checked against the GPU in `waves.rs`). So the sea gets a CPU height
query:
- **The waves:** a direct sum over the few hundred most energetic waves, advanced to the step's
  time. Cascade 0 holds only waves longer than about 21 m, which is enough for the ship, but a
  rowing boat also rides the shorter waves of cascade 1, so the query takes both.
- **Sideways motion:** the waves move the water sideways as well as up, to sharpen their
  crests. The height at a point is found by a few iterations that find which undisplaced point
  moved there.
- **The shallows:** heights shrink by a Rust twin of `shoal()`, tested against the WGSL one, with
  the depth from a small CPU copy of the seabed map.
- **Timing:** the query follows the same eased sea state the GPU draws, advanced by the fixed
  steps' clock, so it stays deterministic.

Buoyancy samples the query at a handful of points on each hull, and pushes each one up by the
water it displaces, as Sea of Thieves does. It has no delay. Reading heights back from the GPU
was the other option; it would arrive frames late and make the physics depend on the GPU's
timing. The query's cost counts in the physics line of the budget, and step 11 measures it.

## Rigging and animation

### Content

Chosen on 8 November 2026: a MakeHuman character, animated with Quaternius's Universal
Animation Library.

- **The character** comes from MakeHuman's Blender plugin, MPFB 2. What it makes is CC0, and
  so are its bundled assets: bodies, skins, hair and clothes. The plugin's own code is GPL, but
  none of it ends up in Solstice. MakeHuman makes realistic people, so the character suits the
  photoscanned cove better than a stylised one would.
  - The character is made by a Blender Python script kept in the repository, written like every
    other script here. Its body, clothes, rig and export settings are all in that script, so it
    can be remade or changed by editing it, the way scenes are made from `scenes/*.txt`.
  - It is exported by Solstice's own export tool (below), with MPFB 2's game-engine rig.
  - Blender 5.2.1 LTS is installed at `D:\Blender`, and the script runs it headless
    (`blender --background --python`). MPFB 2 is not installed yet. Installing it from Blender's
    extensions site waits for your go-ahead when step 6 begins, and that step first checks that
    it supports Blender 5.2.
- **The animations** come from the Universal Animation Library, volumes 1 and 2: over 250 CC0
  clips in glTF on one humanoid rig, with walking in eight directions, jogging, sprinting,
  sitting, climbing, swimming and more. The free tier is enough. The *Source* tier, with its
  Blender files, is not needed.
- **Retargeting.** The two rigs differ: they name, count and orient their bones differently,
  and their proportions differ. So the bake retargets each clip onto the character, as Unreal's
  IK Retargeter does:
  - each rig gets a small text file naming the bone that plays each role: pelvis, spine chain,
    neck, head, and each limb's chain;
  - rotations are carried across in each bone's rest frame, so a difference in rest pose cancels
    out;
  - the pelvis's movement is scaled by the ratio of leg lengths, so feet don't slide;
  - IK cleans up the hands and feet afterwards (step 9).

  Doing it in the bake, in Rust, keeps it reproducible and tested. Changing the character later
  means changing its role file, and no clip needs touching.
- **Licences** are recorded with each asset, as for the Poly Haven scans.

### The export tool

MakeHuman's glTF export has known rough edges, as its users report. Clothing's helper vertex
groups come out unless modifiers are applied, and materials come out grey when Blender's shader
nodes are too complex for its exporter. So Solstice gets its own export tool: a Blender Python
script in the repository (`tools/blender/solstice_export.py`), run headless. It doesn't write
glTF itself. It prepares the character, then hands it to Blender's own glTF writer with one fixed
set of options, then checks the result with Solstice's importer.

1. **Hidden geometry.** MPFB 2 hides the body under clothes with helper modifiers and vertex
   groups. The tool applies them for real, keeping the armature live, deletes the hidden faces,
   and drops every vertex group that isn't a bone.
2. **Materials.** Each material is baked, with Cycles, down to the textures the importer reads:
   base colour (with cavity), normal, and roughness and metallic. Every material then comes out
   right, whatever its node graph.
3. **Weights.** At most four bones per vertex, renormalised; tiny weights dropped; bones that
   deform nothing removed, except those the retargeting role file names.
4. **Space.** Metres, Y up, transforms and rest pose applied, so the import needs no
   corrections.
5. **Export** with fixed options, so every export of the same script comes out the same.
6. **Checks,** by reading the file back with Solstice's importer. The tool fails loudly unless:
   - no vertex group is left that isn't a bone;
   - every material has its textures;
   - every vertex has at most four weights, summing to one;
   - every bone the role file names is present.

It is a separate tool, so the engine only ever sees clean glTF, and it works for any rigged
character, not just MakeHuman's. Its first job, before any of it is written, is to export a
default MakeHuman character as it stands and see what actually goes wrong, since the reports
above are secondhand.

### Skinning in virtual geometry

Skinned meshes go through the same pipeline as everything else: the cluster DAG, streaming, both
rasterizers, the visibility buffer and resolve. They do not get a separate path.

- **Weights.** Each vertex gets up to four bone influences, as 8-bit bone indices into its
  cluster's own list of bones and 8-bit weights: 8 more bytes per vertex. They are stored only
  for skinned meshes, in their own part of each page. The vertex's spare 16 bits stay free.
- **Building the DAG** in bind pose:
  - clusters are grouped by the bones they use as well as by position, so each cluster's list
    stays short, mostly 8 bones or fewer, with a hard cap of 16. Coarse levels may want more,
    such as a hand with all its fingers. There, the smallest influences are dropped and the rest
    renormalised until the cluster fits, and that change counts in its simplification error;
  - simplification keeps the weights: collapsing an edge blends them, and the quadric error
    counts a change of weights, so joints aren't simplified away;
  - borders between clusters stay locked, as now, so a shared vertex keeps exactly the same
    weights on both sides.
  - Equal weights are not quite enough for no cracks: the skinned position must come out the
    same to the last bit. Two clusters can list the same bones in different orders, so each
    vertex's influences are stored sorted by the skeleton's own bone numbers, and summed in that
    order everywhere.
- **Culling.** Each cluster stores its bounding sphere in each of its bones' bind space. Each
  frame, its bounds are the union of those spheres, moved by the current bones, which is cheap
  for a list of 8. Hierarchy nodes get posed bounds the same way.
- **LOD.** LOD error is measured in bind pose and scaled by the largest bone scale, with a
  margin for bending. The cut stays free of holes and overlaps only if a group's clusters and
  the clusters made from them decide from the same sphere. So each group's LOD sphere is posed
  once, by one function over one set of bones stored with the group, and both sides of the
  decision read that result.
- **Rasterizing.** Both rasterizers, resolve and the shadow pass call one shared function,
  `skinned_position`. It applies linear blend skinning from a buffer of bone matrices: this
  frame's, and last frame's for motion vectors. Normals are skinned with the same matrices.
  - Vertices are skinned where they are used, as Unreal does: there is no pre-skinned copy, so
    no extra memory, and it works whatever the LOD cut and whatever has streamed in.
  - Resolve skins each pixel's three vertices, with four bones each, for this frame and the
    last: a real cost on integrated GPUs.
  - If the cost proves too high, the fallback is a compute pass that skins only the visible
    clusters into a scratch buffer each frame. Step 7 measures both.
  - Dual-quaternion skinning, for shoulders and wrists that would otherwise pinch, is an option
    to judge on real characters.
- **Moving things in the rest of the renderer** (step 1, for every moving instance, skinned or
  not):
  - *Marking them.* Each instance gets a "moving" flag. Culling for the cached views (the shadow
    cascades' cache, the seabed map and GI voxelisation) skips flagged instances, so none of them
    keeps a ghost of something at its old place.
  - *Shadows.* The cached cascades keep only static geometry. Moving instances are drawn each
    frame into a map of their own for each cascade they touch, and shading multiplies the light
    the two let through. Nothing is copied: copying all four 2048² cascades each frame would move
    about 128 MB, too much for integrated GPUs.
    - As built (step 1): the four moving maps share one more layer of the cascades' array, a
      quarter each, at half their resolution. The shaders that sample shadows already use
      WebGPU's limit of 16 textures, so a new texture was not an option, and one layer costs
      16 MiB rather than four layers' 64 MiB.
    - A cascade with nothing movable in it draws nothing more and is sampled once, as before.
    - The layer costs 16 MiB even in a scene with nothing movable: an array's layers can't be
      added later without making it again.
    - Cost: 0.45 ms on the RTX 5080 with the cove's ship moving in all four cascades, nearly all
      of it one culling pass per cascade. Culling once for all four is left to the stress-test
      work.
  - *GI voxels.* Moving things are left out at first. They receive GI and appear in screen-space
    GI, but don't occlude or bounce light in the voxels.
  - *Wetness.* The water-line map draws no geometry, and needs nothing. But its wetness belongs
    to places, so a moving hull would wrongly lose its wet band, and carry none. Moving things
    get their own wetness later, with the rain work.

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

The renderer already eases the weather and the ocean. Gameplay gets them, with the time of day
and the sea level, as one `Environment` in the world, and sets targets, not values each frame.
The time of day and the sea level take effect at once in the renderer, so the `Environment`
eases those itself, on the fixed steps' clock: a tide is a sea level eased over hours.

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
  - lights switched on and off, and moved. GI's lighting is updated every few frames
    (`LIGHT_SCHEDULE` in `gi/volume.rs`), so it follows a changed light.
- **A moving sun breaks the shadow cache.** A cascade is redrawn whenever its view changes, and
  its view includes the sun's direction. The 20-minute day above turns the sun about 0.3° a
  second, so every cascade would be stale every frame, and the most expensive pass on small GPUs
  would run all the time. So the sun's direction for shadows moves in small steps, a step every
  second or so, and the cascades catch up over the frames after it, far ones taking turns, as
  they already do. Step 10 measures how small a step looks smooth.
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

**As built (step 4, `crates/solstice-game`):**
- **Actor files** are as above, with these differences from the plan:
  - `mesh` names a glTF file, relative to the actor file, and a scale. The bake places it
    wherever the scene places the actor, as one or more instances, and they move with it.
  - `camera X Y Z` makes an actor the player's, with its eye there.
  - `collider` comes with physics, in step 5.
  - Properties are `bool`, `int`, `float`, `text` (one word) and `vec3`. A scene line sets a
    vector as `x,y,z`.
- **In the scene file.** The bake carries a scene's gameplay in an optional `GAME` section. It
  holds named text files: each actor file and script, the input actions, and `scene.game`, the
  scene's `actor` lines with the instances each actor's mesh became. The section knows nothing
  of what is in it. A scene without actors has no section, so it is written byte for byte as
  before. The bake checks every actor, setting and script, and stops at the first mistake, with
  its line.
- **Input** (`.input` files, or built-in actions). An action's value is the sum of its
  controls, times their scales.
  - Buttons and sticks add up to between -1 and 1.
  - Movement (the mouse, and dragging on the screen) is in pixels a second. That keeps
    `self.yaw += input("turn") * dt` right however the frames and steps fall.
  - Sticks have a dead zone.
  - A tap presses its action without holding it.
- **Play mode** (`Play`) runs the frame, then says which instances moved and where the
  player's eye is.
  - Each actor is drawn between its last two steps, and only when it moved.
  - Looking up and down (`look_up`) moves the camera alone, per frame. Turning is the
    player's own yaw, which its script sets.
- **Viewers.**
  - Natively, `viewer SCENE --play` plays the scene, with keys and the mouse. It reloads
    scripts from `--scripts DIR` (by default `scenes`) when they change, checking twice a
    second. `--play --capture` runs a game headless, with no input.
  - On the web, the plain build offers *Play* on scenes with a game, which reloads the page
    with the gameplay build. That build reads keys, the mouse, standard-layout gamepads and
    touch: a stick under the left thumb, looking under the right, and taps. It shows what
    scripts log, and their errors, in a panel.
  - Gamepads natively are left for later: they need a crate of their own (`gilrs`).
- **Maths.** Starlark has no maths module, so the engine gives `sin`, `cos`, `atan2` and
  `sqrt`.
- **Until physics,** the player walks at a fixed height, through walls.
- **Cost** (`tools/gameplay-bench/b_play`, 9800X3D, warm runs):
  - **Play mode's own work.** Each frame it maps input, works out where each actor is drawn,
    and moves the instances of those that moved. That is about 0.09 µs an actor that stands
    still, on the web and natively.
  - **Turning actors.** With a thousand actors, each with a ticking script that turns it, a
    frame takes 0.75 µs an actor on the web, and 1.3 µs natively, scripts included.
  - **Downloads.** The plain web viewer grows by 879 bytes gzipped (387,396 to 388,275), for
    reading the gameplay section. The gameplay build is 2.15 MB gzipped, and is fetched only to
    play.

## Steps

Each step is its own PR, with mutation-tested tests, measured costs (natively and on the web),
clippy clean on native and wasm32, and a review by a separate AI agent. It is merged only with
your go-ahead, and nothing is published without asking.

1. **Things that move.** *Done.* The renderer API for moving instances, with `prev_world`
   set to the transform each was drawn with last frame; hiding instances and scaling their glow;
   the moving flag, with moving instances left out of the cached shadows, the seabed map and the
   GI voxels, and drawn into per-cascade shadow maps of their own.
   - Demo: the cove's ship bobs at anchor, driven by plain Rust for now (`viewer --bob`).
   - Checked, and decided: with its hull out of the GI voxels, the water around the ship stopped
     reflecting it, and the image moved by 4.5 levels in 255 on average. So an instance can now
     *stay in GI* while movable, held where it stood when the voxels were last drawn: right for
     things that move only a little, such as a ship at anchor. With it, the difference from a
     static ship is 0.4 levels in 255. The demo uses it; a character walking about would not.
   - Tests: a moved instance's motion vectors and shadow follow it; cached shadows of static
     geometry stay as they were; nothing static is redrawn; nothing moving is left behind in a
     cache.
2. **The world.** *Done.* `solstice-world`: entities and components on `hecs`, the
   transform tree, tick groups, events, timers, and the fixed clock with input per step.
   Headless.
   - Tests: tick order; event order; a recorded run replays to the same state; the clock under
     slow and uneven frames.
   - Cost: a step of the benchmark (`tools/gameplay-bench/b_world`: turning a thousand roots,
     working out every place, and an event per root) takes about 16 ns an entity on the web
     (wasm with SIMD, 9800X3D) and 13 ns natively. So 9,000 entities with transforms take
     0.15 ms a step on the web; perhaps 0.3 ms on an integrated-GPU laptop, within the world's
     line of the budget. Every entity with a transform is worked out every step. If scenes grow
     past that, skipping what hasn't moved comes next.
3. **Scripting.** *Done.* `solstice-script` with Starlark: actor scripts, properties and
   `self.vars`, events, timers, the engine API table and its generated bindings, hot reload,
   step and memory caps, and errors with file and line. The benchmarks are run again in place.
   - Tests: hot reload keeps state; a runaway script is stopped and reported, at the same point
     natively and on the web; a broken reload keeps the old script running.
4. **Scenes, actors and play mode.** *In review.* `.actor` files, `actor` lines in scene files and their
   tagged section, input actions, a player camera, and *Play* in both viewers, with the second
   web build.
   - Tests: actor files and scene lines read back what they wrote, and report mistakes clearly;
     an action reaches a script from each kind of input; a scene without gameplay loads into the
     plain viewer exactly as before.
   - Demo: in the cove, walk up to the fort's gate and press E, and it rises
     (`viewer assets/cove.solstice --play`, or *Play* on the web page).
5. **Physics for walking.** `solstice-physics`: collision baked and fetched on *Play*, bodies,
   sensors, the character controller, a debug view of colliders, wasm SIMD, and determinism
   measured.
   - Demo: a capsule walks the fort, climbs the stair to the rampart, and opens the gate.
6. **The character and the export tool.** The Blender script that makes the MakeHuman
   character, and `tools/blender/solstice_export.py`. Installing MPFB 2 waits for your go-ahead.
   - First, a default character exported as MPFB 2 does it, to see what goes wrong.
   - Tests: the export's checks fail on a deliberately broken character, and pass on the real
     one; two exports of the same script come out the same.
7. **Skinned virtual geometry.** Importing skins, weights through the DAG, per-cluster bone
   lists and bounds, posed node and group bounds, and skinning in both rasterizers, resolve and
   shadows, with motion vectors.
   - First the character in its bind pose, compared with the static import; then one looping
     clip.
   - Tests: no cracks between clusters in any pose; the LOD cut has no holes or overlaps while a
     clip plays.
   - The cost is measured, against the scratch-buffer fallback.
8. **Animation.** Clips and their compression, retargeting from the Universal Animation Library's
   rig onto the character's, blends, blend spaces, state machines, root motion, notifies, and
   animation LOD. Its budget line is measured.
   - Tests: a clip retargeted onto its own rig comes back unchanged; a retargeted walk's feet
     slide less than a set distance while they bear weight. IK in step 9 removes the rest.
   - Demo: the capsule becomes a character that idles, walks and runs.
9. **IK and secondary motion.** Two-bone IK, foot placement, look-at, FABRIK, and spring chains.
   - Demo: feet planted on the stair and the uneven courtyard. This is the first playable
     milestone: a character in the fort.
10. **Gameplay and the environment.** The `Environment` API and its events, with the time of day
    and sea level eased by gameplay; the sun stepped for shadows; lights switched by gameplay;
    viewer controls as overrides.
11. **The sea in physics.** The CPU wave query, checked against the GPU's waves, with its Rust
    `shoal()`; buoyancy: the ship floats, and a rowing boat can be pushed about.
12. **A playable cove.** Walk a character through the fort, open the gate, row out to the ship,
    and see the lanterns lit at dusk. README, screenshots and costs on the RTX 5080 and on the
    integrated-GPU tier. Published only with your go-ahead.

The steps run in this order. The first playable thing is a character in the fort (chosen on
8 November 2026), so walking, the character, skinning and animation (5–9) come before the
environment and the sea (10–11).

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
- **The renderer assumes a static world.** Shadow caching, the GI voxels and the seabed map all
  do, and wetness belongs to places rather than objects. Step 1 handles the first three, wetness
  on moving things waits for the rain work, and anything new must say how it treats moving
  things. A moving sun is the same problem in another form, handled in step 10.
- **Skinned virtual geometry is new ground.** Unreal added it in 5.5, as an experimental
  feature, and has published little about it. Step 7 starts with a bind-pose comparison, so any
  crack or LOD error shows up before animation depends on it.
- **Content.** MakeHuman's glTF export has rough edges: users report exporting every clothing
  vertex group unless modifiers are applied, and materials coming out grey when Blender's shader
  nodes are too complex. The export tool (step 6) exists to fix both, and checks its output
  before anything depends on it. Retargeting
  between two different rigs can leave feet sliding or limbs twisted, which the retargeting
  tests and IK are there to catch.
- **Determinism across native and the web** depends on Rapier's option and on avoiding
  platform-specific maths in gameplay code. Tests compare the two builds.
- **Dependencies move.** Starlark and Rapier are both pre-1.0. Both are pinned, and upgraded on
  purpose, as the toolchain is.
