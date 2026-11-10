# Porsche 911 GT3 R (992): the model

Builds the car the player drives, from Porsche's published dimensions and from measurements of
its studio photographs. Blender 4.2 or later.

```bash
blender --background --factory-startup --python-exit-code 1 --python tools/blender/gt3r/make.py -- target/gt3r/gt3r.glb --engine target/circuit/gt3r.glb --audit
```

- `target/gt3r/gt3r.glb`: every part a named mesh of its own (about 180 of them).
- `target/circuit/gt3r.glb`: what the game bakes (`scenes/circuit/gt3r.actor`). Everything
  fixed to the body is one mesh, and the lamps' clear lenses are left out, as Solstice draws
  blended materials solid.

To look at the work, `--views DIR [name,name]` renders inspection views with Cycles, and
`sheet.py` tiles them into one image. `go.sh` does both.

## How it is built

| File | What it makes |
|---|---|
| `spec.py` | The dimensions, and the body's lines along the car: each one a curve through measured points. |
| `body.py` | The shell, lofted through those lines and subdivided; the wheel arches; a check of the shell against the measurements. |
| `glazing.py` | The cabin's hollow, the window openings cut through to it, the glass and its seals. |
| `wheels.py` | Tyres, forged centre-lock rims, slotted discs, and calipers. |
| `front.py` | Intake, splitter, dive planes, lamps, bonnet outlets, wing louvres. |
| `sides.py` | Side intakes, arch vents, skirts, doors, mirrors, fillers, seams, race numbers, roof. |
| `rear.py` | Ducktail, light bar, bumper, tailpipes, diffuser, engine cover, wing and its mounts. |
| `cockpit.py` | Roll cage, seat and harness, wheel, dash, switch panel, nets. |
| `common.py` | Curves, meshes, materials, cuts, and fitting things to the body's surface. |
| `render.py`, `sheet.py` | Inspection views. |
| `audit.py` | Looks for floating parts, an open or inside-out shell, and tyres off the ground. |

Details are fitted to the body rather than placed by hand. `Surface` casts rays at the skin
as it was before anything was cut, and lays outlines on it: `patch` for glass and lenses,
`ribbon` for shut lines and seals, `recess` to sink an opening into the shell, `fin` for
dive planes and louvres. So moving one of the body's lines in `spec.py` carries every detail
with it.

## What the engine needs of it

- The car points along +Y in Blender (-Z in the file), with its origin on the ground under the
  middle of its wheelbase.
- `wheel_fl`, `wheel_fr`, `wheel_rl`, `wheel_rr` each have their origin at their hub, where
  `scenes/circuit/gt3r.vehicle` has it. The engine finds them by that, and turns them.
- `caliper_*` each have their origin 30 mm inboard of their hub. The engine carries those with
  the wheel, steering and riding the suspension, without turning them.

The model carries no badges, sponsors' marks or tyre makers' lettering.
