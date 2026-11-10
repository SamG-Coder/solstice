"""The 911 GT3 R's front end: intake, splitter, dive planes, lamps, bonnet outlets, wing louvres.

From the front view M22_3540, the front three-quarter M22_3539 and the side view M22_3548.
"""

import math

import bpy

import spec
from common import box, boolean, ellipse, join, lathe, line, new_object, prism, rounded, smooth, tube

# The opening across the bumper, seen from ahead: (x, z).
INTAKE = [(-0.72, 0.40), (-0.805, 0.31), (-0.77, 0.19), (-0.40, 0.165), (0.0, 0.16),
          (0.40, 0.165), (0.77, 0.19), (0.805, 0.31), (0.72, 0.40), (0.0, 0.418)]
# The splitter, seen from above: (x, y), its right half from the wheel forward to the middle.
SPLITTER = [(1.035, 1.66), (1.045, 1.90), (1.0, 2.06), (0.90, 2.16), (0.72, 2.215),
            (0.40, 2.248), (0.0, spec.SPLITTER_TIP)]
# Each lamp's lens, seen from ahead: a round lamp, laid back along the wing.
LAMP_AT, LAMP_RADIUS, LAMP_HEIGHT = (0.707, 0.66), 0.122, 0.100
# One of the two outlets in the bonnet, seen from above: the right-hand one.
OUTLET = [(0.045, 1.26), (0.20, 1.265), (0.355, 1.30), (0.33, 1.50), (0.275, 1.69),
          (0.045, 1.725)]


def build(shell, surface, mats):
    parts = []
    liner, black, carbon, accent = mats["liner"], mats["black"], mats["carbon"], mats["accent"]
    above = lambda x, y: ((x, y, 3.0), (0, 0, -1))  # noqa: E731
    ahead = lambda x, z: ((x, 4.0, z), (0, -1, 0))  # noqa: E731

    def add(obj):
        if obj is not None:
            parts.append(obj)
        return obj

    # ---- The intake: one opening the width of the bumper, with the coolers' mesh set back in
    # it, uprights dividing it, and a bar across.
    intake = rounded(INTAKE, 8)
    boolean(shell, prism("cut", intake, "y", 1.90, 2.4, liner))
    add(surface.rim("intake_lip", intake, ahead, 0.02, 0.0015, carbon))
    grown = [(x * 1.04, 0.29 + (z - 0.29) * 1.08) for x, z in intake]
    add(prism("intake_mesh", grown, "y", 1.955, 1.962, mats["mesh"]))
    for x in (-0.52, -0.24, 0.24, 0.52):
        reach = surface.cast((x, 4.0, 0.28), (0, -1, 0))
        front = (reach[0].y if reach else 2.1) - 0.035
        add(box("intake_upright", (x - 0.009, 1.96, 0.165), (x + 0.009, front, 0.41), carbon, 0.003))
    add(surface.fin("intake_bar", [ahead(x, 0.29) for (x,) in line((-0.76,), (0.76,), 40)],
                    lambda t: -0.045, 0.014, carbon, out=(0, 1, 0)))

    # ---- The splitter: a blade under the nose, out to the wheels, its edge picked out.
    outline = SPLITTER + [(-x, y) for x, y in reversed(SPLITTER[:-1])]
    edge = rounded(outline, 6, closed=False)
    blade = edge + [(-1.035, 1.66), (-0.86, 1.60), (0.86, 1.60)]
    blade = [(x, y) for x, y in blade]
    add(prism("splitter", blade, "z", 0.043, 0.058, carbon, 0.003))
    add(tube("splitter_edge", [(x, y, 0.0505) for x, y in edge], 0.0095, accent, 10))
    for side in (-1, 1):
        # The fence at each end of it.
        fence = [(1.60, 0.05), (2.02, 0.05), (2.0, 0.125), (1.82, 0.165), (1.62, 0.165)]
        add(prism("splitter_fence", fence, "x", side * 1.036 - 0.004, side * 1.036 + 0.004,
                  carbon, 0.002))

    # ---- Dive planes: two curved blades on each corner of the bumper.
    for side, tag in ((1, "right"), (-1, "left")):
        beside = lambda y, z, side=side: ((side * 3.0, y, z), (-side, 0, 0))  # noqa: E731
        for k, (start, end) in enumerate((((1.775, 0.385), (2.05, 0.275)),
                                          ((1.775, 0.232), (2.05, 0.158)))):
            path = [(y, z - 0.03 * math.sin(math.pi * t / 30))
                    for t, (y, z) in enumerate(line(start, end, 30))]
            add(surface.fin(f"dive_plane_{tag}_{k}", [beside(y, z) for y, z in path],
                            lambda t: 0.035 + 0.05 * math.sin(math.pi * min(t * 1.3, 1.0)),
                            0.006, carbon, out=(side, 0, 0.12)))

    # ---- The lamps: a bowl sunk in the front of each wing, four LED units in it, and a clear
    # lens over it that carries on the wing's surface.
    for side, tag in ((1, "right"), (-1, "left")):
        cx, cz = side * LAMP_AT[0], LAMP_AT[1]
        lens = ellipse(cx, cz, LAMP_RADIUS, LAMP_HEIGHT, 48)
        surface.recess(shell, ellipse(cx, cz, LAMP_RADIUS - 0.006, LAMP_HEIGHT - 0.006, 40),
                       ahead, 0.085, black)
        add(surface.patch(f"lens_{tag}", lens, ahead, mats["lens"], 0.0005))
        add(surface.rim(f"lamp_rim_{tag}", lens, ahead, 0.012, 0.0012, mats["seal"]))
        units = []
        for dx in (-0.050, 0.050):
            for dz in (-0.042, 0.042):
                found = surface.cast((cx + dx, 4.0, cz + dz), (0, -1, 0))
                if not found:
                    continue
                at, normal = found
                # Each unit sits on the bowl's floor, facing the way the car is going.
                base = at - normal * 0.07
                unit = box("unit", (base.x - 0.032, base.y - 0.015, base.z - 0.030),
                           (base.x + 0.032, base.y + 0.030, base.z + 0.030), mats["black"], 0.004)
                units.append(unit)
                for ix in (-0.016, 0.016):
                    for iz in (-0.013, 0.013):
                        units.append(box("led", (base.x + ix - 0.009, base.y + 0.028, base.z + iz - 0.007),
                                         (base.x + ix + 0.009, base.y + 0.033, base.z + iz + 0.007),
                                         mats["chrome"], 0.002))
                units.append(box("led", (base.x - 0.006, base.y + 0.030, base.z - 0.004),
                                 (base.x + 0.006, base.y + 0.035, base.z + 0.004), mats["lamp"]))
        if units:
            add(join(units, f"lamp_units_{tag}"))

    # ---- The bonnet: two outlets the coolers' air leaves by, either side of a spine; the
    # duct that feeds the cockpit; and the lid's own outline.
    for side in (1, -1):
        outline = rounded([(side * x, y) for x, y in OUTLET], 8)
        surface.recess(shell, outline, above, 0.075, liner)
        add(surface.rim("outlet_lip", outline, above, 0.014, 0.0015, carbon))
        for y in (1.37, 1.49, 1.61):
            add(surface.fin("outlet_vane", [above(side * x, y) for (x,) in line((0.06,), (0.31 - (y - 1.3) * 0.12,), 12)],
                            lambda t: -0.05, 0.006, carbon, out=(0, 0.5, 1)))
    # The fuel cell is in the nose: its filler stands in the right-hand outlet.
    floor = surface.top(0.27, 1.40)
    if floor:
        z = floor - 0.07

        def filler(bm):
            lathe(bm, [(z - 0.01, 0.050), (z + 0.022, 0.050), (z + 0.030, 0.043), (z + 0.030, 0.030),
                       (z + 0.018, 0.026), (z + 0.018, 0.001)], 32, axis="z", centre=(0.27, 1.40, 0.0))
        add(smooth(new_object("fuel_filler", filler, [mats["aluminium"]]), 40.0))
    duct = rounded([(0.215, 2.0), (0.235, 1.86), (0.285, 1.80), (0.335, 1.86), (0.30, 2.0)], 6)
    surface.recess(shell, duct, above, 0.028, liner)
    for side in (-1, 1):
        # The lid's edge, down the valley beside each wing, and two round openings in the nose.
        rays = [above(side * (spec.XV(y) + 0.012), y) for (y,) in line((1.03,), (2.035,), 60)]
        add(surface.ribbon("bonnet_gap", rays, 0.005, 0.0012, mats["seal"]))
        oval = ellipse(side * 0.30, 2.045, 0.05, 0.02, 20)
        surface.recess(shell, oval, above, 0.02, mats["mesh"], cuts=1)
        # Quick-release pins holding the lid.
        for y in (1.12, 1.92):
            x = side * (spec.XV(y) - 0.07)
            add(surface.patch("bonnet_pin", ellipse(x, y, 0.016, 0.016, 14), above, mats["aluminium"], 0.002, 1))
    add(surface.ribbon("bonnet_gap", [above(x, 2.045 - 0.05 * (x / 0.46) ** 2) for (x,) in line((-0.45,), (0.45,), 40)],
                       0.005, 0.0012, mats["seal"]))

    # ---- Louvres over each front wheel: slots through the wing's top that let the arch breathe.
    for side in (-1, 1):
        for k in range(5):
            y = spec.FRONT_AT - 0.19 + k * 0.095
            inner, outer = 0.735, 0.955 - 0.012 * abs(k - 2)
            slot = [(side * inner, y), (side * outer, y + 0.012), (side * outer, y + 0.05),
                    (side * inner, y + 0.038)]
            surface.recess(shell, slot, above, 0.035, liner, cuts=2)

    # ---- The tow strap, on the nose.
    add(tube("tow_strap", [(0.36, 2.10, 0.45), (0.36, 2.19, 0.455), (0.36, 2.20, 0.42),
                           (0.36, 2.11, 0.415)], 0.007, accent, 8, closed=True))
    return parts
