"""Builds the model of the Porsche 911 GT3 R (992), for `scenes/circuit/gt3r.vehicle`.

    blender --background --factory-startup --python-exit-code 1 \
        --python tools/blender/gt3r/make.py -- OUT.glb [--engine ENGINE.glb]
        [--views DIR [NAME,...]] [--until STAGE] [--paint R,G,B] [--outdoor] [--samples N]

`OUT.glb` holds every part as a mesh of its own, named. `--engine` also writes the file the
game bakes: the same model with everything fixed to the body joined into one mesh, and without
the lamps' clear lenses, which Solstice would draw solid.

Built from Porsche's published dimensions and from measurements of its studio photographs:
see `spec.py`, and target/gt3r/REFERENCES.md for which photograph supports what. The model
carries no badges or sponsors' marks.

In Blender the car points along +Y, with +X to its right and +Z up, and its origin on the ground
under the middle of its wheelbase. The glTF writer turns that into the engine's own: -Z ahead,
+Y up. The four wheels are objects of their own, `wheel_fl`, `wheel_fr`, `wheel_rl` and
`wheel_rr`, each with its origin at its hub: the engine finds them by that, and turns them.

`--until` stops after a stage, to look at the work so far: shell, arches, glazing, wheels,
front, sides, rear, cockpit.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

import body  # noqa: E402
import render  # noqa: E402
from common import Surface, material, smooth  # noqa: E402

STAGES = ["shell", "arches", "glazing", "wheels", "front", "sides", "rear", "cockpit"]


def materials(paint):
    light = 0.3 * paint[0] + 0.59 * paint[1] + 0.11 * paint[2] > 0.25
    return {
        # Numbers in whichever of black and white shows on the paint.
        "number": material("number", (0.012, 0.012, 0.012) if light else (0.85, 0.85, 0.83), 0.4),
        "paint": material("paint", paint, 0.30, 0.0, coat=1.0),
        "liner": material("liner", (0.012, 0.012, 0.013), 0.85),
        "carbon": material("carbon", (0.022, 0.022, 0.024), 0.36, coat=0.6),
        "black": material("black", (0.010, 0.010, 0.011), 0.55),
        "rubber": material("rubber", (0.018, 0.018, 0.019), 0.82),
        "seal": material("seal", (0.008, 0.008, 0.008), 0.7),
        "accent": material("accent", (0.62, 0.02, 0.025), 0.32, coat=0.8),
        # GT3R_SOLID_GLASS=1 draws the panes solid green, to look at their shape.
        "glass": (material("glass", (0.05, 0.8, 0.1), 0.6) if os.environ.get("GT3R_SOLID_GLASS")
                  else material("glass", (0.05, 0.06, 0.065), 0.04, alpha=0.32)),
        "lens": material("lens", (0.75, 0.78, 0.8), 0.03, alpha=0.22),
        "chrome": material("chrome", (0.80, 0.80, 0.80), 0.12, 1.0),
        "aluminium": material("aluminium", (0.62, 0.63, 0.64), 0.35, 1.0),
        "steel": material("steel", (0.32, 0.32, 0.33), 0.42, 1.0),
        "wheel": material("wheel", (0.022, 0.022, 0.024), 0.34, 0.6, coat=0.5),
        "disc": material("disc", (0.20, 0.19, 0.18), 0.48, 1.0),
        "caliper": material("caliper", (0.025, 0.025, 0.027), 0.38, 0.4),
        "nut_red": material("nut_red", (0.55, 0.04, 0.03), 0.3, 0.9),
        "nut_blue": material("nut_blue", (0.04, 0.12, 0.55), 0.3, 0.9),
        "lamp": material("lamp", (0.95, 0.95, 1.0), 0.1, emission=((1.0, 0.97, 0.90), 4.0)),
        "tail": material("tail", (0.45, 0.01, 0.01), 0.25, emission=((1.0, 0.03, 0.02), 2.5)),
        "mesh": material("mesh", (0.03, 0.03, 0.032), 0.6, 0.5),
        "fabric": material("fabric", (0.02, 0.02, 0.022), 0.9),
        "belt": material("belt", (0.5, 0.03, 0.03), 0.8),
        "screen": material("screen", (0.02, 0.03, 0.05), 0.2, emission=((0.15, 0.45, 0.9), 1.2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--until", default=STAGES[-1], choices=STAGES)
    parser.add_argument("--paint", default="0.82,0.82,0.80")
    parser.add_argument("--outdoor", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--engine")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--probe", action="append", default=[],
                        help="VIEW,U,V: say what a view sees at a point of its picture")
    parser.add_argument("--hide", default="", help="leave out parts whose names start with these, to look behind them")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--size", default="800x500")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    started = time.time()
    last = STAGES.index(args.until)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = materials(tuple(float(v) for v in args.paint.split(",")))

    parts, wheels = [], []
    shell = body.build_shell(mats["paint"])
    parts.append(shell)
    # The skin as it was before anything was cut in it: what everything else is fitted to.
    skin = shell.copy()
    skin.data = shell.data.copy()
    skin.name = "skin"
    bpy.context.collection.objects.link(skin)
    surface = Surface(skin)
    print("make: the body against the photographs")
    body.check(surface)
    if last >= 1:
        body.cut_arches(shell, mats["liner"])
    if last >= 2:
        import glazing
        parts += glazing.build(shell, surface, mats)
    if last >= 3:
        import wheels as wheel_maker
        wheels = wheel_maker.build(mats)
    if last >= 4:
        import front
        parts += front.build(shell, surface, mats)
    if last >= 5:
        import sides
        parts += sides.build(shell, surface, mats)
    if last >= 6:
        import rear
        parts += rear.build(shell, surface, mats)
    if last >= 7:
        import cockpit
        cabin = cockpit.build(shell, surface, mats)
        parts += cabin
        import audit
        audit.inside(skin, cabin)
    bpy.data.objects.remove(skin, do_unlink=True)
    smooth(shell, 38.0)

    if args.hide:
        hidden = tuple(args.hide.split(","))
        for obj in [p for p in parts if p is not None and p.name.startswith(hidden)]:
            parts.remove(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
    everything = [p for p in parts if p is not None] + wheels
    if args.audit:
        import audit
        audit.audit(shell, [p for p in parts if p is not None], wheels)
    triangles = sum(len(p.vertices) - 2 for o in everything for p in o.data.polygons)
    print(f"make: {len(everything)} parts, {triangles} triangles, {time.time() - started:.0f} s")
    for obj in sorted(everything, key=lambda o: o.name):
        count = sum(len(p.vertices) - 2 for p in obj.data.polygons)
        print(f"  {obj.name:24s} {count:7d} triangles")

    for spot in args.probe:
        view, u, v = spot.split(",")
        render.probe(view, float(u), float(v))
    if args.views:
        folder = os.path.abspath(args.views[0])
        names = args.views[1].split(",") if len(args.views) > 1 else None
        size = tuple(int(v) for v in args.size.split("x"))
        render.shoot(folder, names, size=size, outdoor=args.outdoor, samples=args.samples)
    if not args.no_export:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in everything:
            obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=output, export_format="GLB", use_selection=True,
                                  export_yup=True, export_apply=True)
        print(f"make: wrote {output}")
    if args.engine:
        # For the game: one mesh for all that is fixed to the body, and each wheel and caliper
        # its own, as the engine finds those by where their origins are.
        output = os.path.abspath(args.engine)
        carried = [o for o in everything if o.name.startswith(("wheel_", "caliper_"))]
        fixed = [o for o in everything if o not in carried and not o.name.startswith("lens_")]
        for obj in [o for o in everything if o.name.startswith("lens_")]:
            bpy.data.objects.remove(obj, do_unlink=True)
        from common import join
        whole = join(fixed, "body")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in [whole] + carried:
            obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=output, export_format="GLB", use_selection=True,
                                  export_yup=True, export_apply=True)
        print(f"make: wrote {output}")
    print(f"make: done in {time.time() - started:.0f} s")


main()
