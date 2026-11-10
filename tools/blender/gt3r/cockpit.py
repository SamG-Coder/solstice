"""The 911 GT3 R's cockpit, as it shows through the glass: cage, seat, wheel, dash, nets.

A single seat on the left, as the car is built. From the cockpit photographs M22_3541 to
M22_3544. Everything here sits inside the cabin's hollow (see glazing.py).
"""

import math

import bpy
from mathutils import Matrix, Vector

import glazing
from common import box, join, lathe, new_object, plate, prism, smooth, transform, tube

DRIVER_X = -0.36


# How far the middle of a cage bar is from the body's skin: the shell, the bar's own radius, and
# a finger's width between them.
CLEAR = glazing.SHELL + 0.046
HOOP_AT = -0.50


def rail(y):
    """Where a bar under the roof's rail, or up the windscreen pillar, runs at y: (x, y, z)."""
    x, z = glazing.inner_ring(y, CLEAR)[13]
    return (x, y, z)


def beside(y, z):
    """Where a bar along the cabin's side runs at y and a height: (x, y, z)."""
    side = glazing.inner_ring(y, CLEAR)[3:14]
    for (x0, z0), (x1, z1) in zip(side, side[1:]):
        if min(z0, z1) <= z <= max(z0, z1) and z1 != z0:
            return (x0 + (x1 - x0) * (z - z0) / (z1 - z0), y, z)
    return (side[-1][0], y, z)


def cage(mats):
    """The welded-in roll cage. Every bar is placed from the cabin's own shape, a fixed way in
    from the skin, so that it runs up the windscreen pillar and under the roof inside them."""
    steel = mats["cage"]
    r = 0.020
    bars = []

    def both(path, radius=r):
        for side in (1, -1):
            bars.append(tube("bar", [(side * x, y, z) for x, y, z in path], radius, steel, 12))

    def across(path, radius=r):
        whole = [(-x, y, z) for x, y, z in reversed(path[:-1])] + path
        bars.append(tube("bar", whole, radius, steel, 12))

    # The main hoop behind the door: up each side, round under the roof, across.
    ring = glazing.inner_ring(HOOP_AT, CLEAR)
    hoop = [(x, HOOP_AT, z) for x, z in ring[3:17]]
    across([(hoop[0][0], HOOP_AT, glazing.CABIN_FLOOR + 0.02)] + hoop)
    # Up each windscreen pillar from the floor, and back under the roof's rail to the hoop.
    pillar = [beside(0.76, 0.16), beside(0.76, 0.45), beside(0.75, 0.72)]
    pillar += [rail(y) for y in (0.66, 0.58, 0.50, 0.42, 0.34, 0.26, 0.18, 0.08, -0.06, -0.22,
                                 -0.36, HOOP_AT)]
    both(pillar)
    # Across the top of the windscreen, and across the roof half-way back.
    for y, radius in ((0.21, r), (-0.14, r * 0.85)):
        across([(x, y, z) for x, z in glazing.inner_ring(y, CLEAR)[13:17]], radius)
    # The door bars: a cross each side, bowed out to the door's skin as real ones are.
    for z0, z1 in ((0.66, 0.30), (0.26, 0.64)):
        both([beside(0.75 + (HOOP_AT - 0.75) * t, z0 + (z1 - z0) * t) for t in (0, 0.25, 0.5, 0.75, 1)])
    both([beside(0.75 + (HOOP_AT - 0.75) * t, 0.46) for t in (0, 0.5, 1)], r * 0.85)
    # The hoop's diagonals, and the bar the harness is fixed to.
    foot, shoulder = hoop[0], hoop[9]
    for side in (1, -1):
        bars.append(tube("bar", [(side * foot[0], HOOP_AT, 0.20), (-side * shoulder[0], HOOP_AT, shoulder[2])],
                         r * 0.9, steel, 12))
    harness = beside(HOOP_AT - 0.03, 0.80)
    bars.append(tube("bar", [(-harness[0], harness[1], 0.80), (harness[0], harness[1], 0.80)], r * 0.9, steel, 12))
    # The stays, back and down from the hoop's shoulders to the rear turrets.
    # They end on top of the rear wheel housings, inboard of the wells.
    turret = beside(-1.20, 0.80)
    turret = (min(turret[0] - 0.12, 0.58), turret[1], turret[2])
    # Half-way back the cabin has narrowed: the stay follows it in.
    x, z = glazing.inner_ring(-0.86, CLEAR + 0.04)[11]
    both([shoulder, (min(x, 0.5 * (shoulder[0] + turret[0])), -0.86, z), turret])
    both([hoop[12], (0.12, -0.90, 0.96), turret], r * 0.85)
    return join(bars, "roll_cage")


def seat(mats):
    """A carbon racing shell with head restraints, its pads, and a six-point harness."""
    carbon, fabric, belt = mats["carbon"], mats["fabric"], mats["belt"]
    lean = math.radians(18)
    pieces = []
    # Built upright about the hip point, then leaned back.
    back = [
        box("shell", (-0.235, -0.07, 0.0), (0.235, 0.0, 0.66), carbon, 0.03),
        box("pad", (-0.17, -0.005, 0.05), (0.17, 0.03, 0.60), fabric, 0.02),
        box("bolster", (-0.25, -0.05, 0.10), (-0.20, 0.13, 0.42), carbon, 0.02),
        box("bolster", (0.20, -0.05, 0.10), (0.25, 0.13, 0.42), carbon, 0.02),
        box("shoulder", (-0.27, -0.05, 0.40), (-0.21, 0.10, 0.62), carbon, 0.02),
        box("shoulder", (0.21, -0.05, 0.40), (0.27, 0.10, 0.62), carbon, 0.02),
        box("head", (-0.16, -0.07, 0.64), (0.16, -0.01, 0.90), carbon, 0.03),
        box("head_wing", (-0.20, -0.06, 0.66), (-0.15, 0.17, 0.88), carbon, 0.02),
        box("head_wing", (0.15, -0.06, 0.66), (0.20, 0.17, 0.88), carbon, 0.02),
        box("head_pad", (-0.12, -0.012, 0.68), (0.12, 0.02, 0.86), fabric, 0.02),
        # The shoulder straps, over the top and down the front.
        box("strap", (-0.115, 0.03, 0.10), (-0.065, 0.042, 0.64), belt),
        box("strap", (0.065, 0.03, 0.10), (0.115, 0.042, 0.64), belt),
    ]
    turn = Matrix.Rotation(lean, 4, "X")
    for piece in back:
        transform(piece, turn)
    pieces += back
    pieces += [
        box("base", (-0.235, -0.02, -0.06), (0.235, 0.47, 0.0), carbon, 0.025),
        box("base_pad", (-0.17, 0.02, -0.005), (0.17, 0.44, 0.035), fabric, 0.02),
        box("thigh", (-0.25, 0.02, -0.03), (-0.20, 0.42, 0.10), carbon, 0.02),
        box("thigh", (0.20, 0.02, -0.03), (0.25, 0.42, 0.10), carbon, 0.02),
        box("lap_strap", (-0.20, 0.10, 0.035), (0.20, 0.15, 0.047), belt),
        box("buckle", (-0.035, 0.105, 0.045), (0.035, 0.155, 0.062), mats["aluminium"], 0.004),
    ]
    whole = join(pieces, "seat")
    transform(whole, Matrix.Translation(Vector((DRIVER_X, -0.33, 0.25))))
    return whole


def wheel_and_dash(mats):
    """The steering wheel with its screen and buttons, the dash, and the switch panel."""
    carbon, black, screen = mats["carbon"], mats["black"], mats["screen"]
    made = []
    # The wheel: a flat-topped, flat-bottomed rim with a hub full of buttons.
    tilt = Matrix.Rotation(math.radians(-22), 4, "X")
    at = Vector((DRIVER_X, 0.36, 0.80))
    rim = []
    for k in range(33):
        a = math.tau * k / 32
        x, z = 0.145 * math.cos(a), 0.105 * math.sin(a)
        # Squared off: a racing wheel is nearer a rectangle than a ring.
        x = math.copysign(abs(x / 0.145) ** 0.6 * 0.145, x)
        z = math.copysign(abs(z / 0.105) ** 0.6 * 0.105, z)
        rim.append((x, 0.0, z))
    wheel = [tube("rim", rim[:-1], 0.016, mats["fabric"], 10, closed=True),
             box("hub", (-0.085, -0.02, -0.06), (0.085, 0.012, 0.06), carbon, 0.012),
             box("hub_screen", (-0.045, 0.012, -0.012), (0.045, 0.016, 0.045), screen),
             box("spoke", (-0.14, -0.012, -0.012), (-0.08, 0.008, 0.012), carbon, 0.004),
             box("spoke", (0.08, -0.012, -0.012), (0.14, 0.008, 0.012), carbon, 0.004)]
    colours = [mats["tail"], mats["lamp"], mats["button_green"], mats["button_blue"], mats["button_yellow"]]
    for i, (bx, bz) in enumerate(((-0.066, 0.035), (0.066, 0.035), (-0.066, 0.0), (0.066, 0.0),
                                  (-0.066, -0.035), (0.066, -0.035), (-0.03, -0.04), (0.03, -0.04))):
        wheel.append(box("button", (bx - 0.009, 0.012, bz - 0.009), (bx + 0.009, 0.019, bz + 0.009),
                         colours[i % len(colours)], 0.003))
    wheel = join(wheel, "steering_wheel")
    # Facing the driver: its face towards -Y.
    transform(wheel, Matrix.Rotation(math.pi, 4, "Z"))
    transform(wheel, tilt)
    transform(wheel, Matrix.Translation(at))
    made.append(wheel)
    made.append(tube("steering_column", [tuple(at), (DRIVER_X, 0.62, 0.74), (DRIVER_X, 0.86, 0.66)],
                     0.022, black, 10))

    # The dash: a carbon cowl across the car under the windscreen, with the driver's display.
    dash = [box("dash", (-0.70, 0.60, 0.70), (0.70, 0.95, 0.80), carbon, 0.03),
            box("dash_hood", (DRIVER_X - 0.17, 0.52, 0.78), (DRIVER_X + 0.17, 0.70, 0.86), carbon, 0.02),
            box("display", (DRIVER_X - 0.11, 0.515, 0.795), (DRIVER_X + 0.11, 0.522, 0.85), screen)]
    made.append(join(dash, "dashboard"))

    # The switch panel beside the driver: rows of lit buttons on a carbon console.
    console = [box("console", (-0.08, -0.25, 0.14), (0.12, 0.50, 0.40), carbon, 0.02),
               box("panel", (-0.07, 0.10, 0.395), (0.11, 0.46, 0.41), black, 0.004)]
    for row in range(5):
        for col in range(3):
            bx, by = -0.045 + col * 0.045, 0.14 + row * 0.07
            console.append(box("switch", (bx - 0.014, by - 0.014, 0.408), (bx + 0.014, by + 0.014, 0.418),
                               colours[(row + col) % len(colours)], 0.003))
    made.append(join(console, "switch_panel"))
    return made


def build(shell, surface, mats):
    import bmesh  # noqa: F401
    from common import material
    mats = dict(mats)
    mats["cage"] = material("cage", (0.55, 0.55, 0.56), 0.4, 0.2)
    mats["button_green"] = material("button_green", (0.02, 0.4, 0.05), 0.3, emission=((0.1, 1.0, 0.2), 1.5))
    mats["button_blue"] = material("button_blue", (0.02, 0.08, 0.5), 0.3, emission=((0.15, 0.4, 1.0), 1.5))
    mats["button_yellow"] = material("button_yellow", (0.5, 0.4, 0.02), 0.3, emission=((1.0, 0.8, 0.1), 1.5))
    parts = [cage(mats), seat(mats)] + wheel_and_dash(mats)
    carbon, black = mats["carbon"], mats["black"]
    # The bulkhead behind the seat, and the floor.
    parts.append(box("bulkhead", (-0.72, -0.66, 0.14), (0.72, -0.63, 0.80), carbon))
    parts.append(box("cabin_floor", (-0.74, -0.64, 0.128), (0.74, 0.82, 0.142), carbon))
    # The window net on the driver's side: black webbing in a grid between the bar up the
    # pillar and the hoop, as M22_3544 shows, with a red strap to release it.
    webbing = []
    front = [rail(0.42), rail(0.30), rail(0.18)]
    back = [beside(HOOP_AT, 0.90), beside(HOOP_AT, 1.00), rail(-0.36)]
    low_front, low_back = beside(0.55, 0.86), beside(HOOP_AT, 0.86)
    rows = [(low_front, low_back)] + list(zip(front, back))
    for a, b in rows:
        webbing.append(tube("strap", [(-a[0], a[1], a[2]), (-b[0], b[1], b[2])], 0.011, mats["fabric"], 6))
    for t in (0.2, 0.4, 0.6, 0.8):
        column = [(-(a[0] + (b[0] - a[0]) * t), a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)
                  for a, b in rows]
        webbing.append(tube("strap", column, 0.011, mats["fabric"], 6))
    parts.append(join(webbing, "window_net"))
    a, b = rows[1]
    parts.append(tube("window_net_release", [(-a[0], a[1] - 0.02, a[2]), (-a[0] + 0.02, a[1] - 0.10, a[2] - 0.05)],
                      0.012, mats["belt"], 6))
    # The fire extinguisher's bottle, on the floor where a passenger's feet would be.
    def bottle(bm):
        lathe(bm, [(-0.17, 0.001), (-0.17, 0.055), (-0.15, 0.068), (0.13, 0.068), (0.16, 0.045),
                   (0.18, 0.02), (0.20, 0.02), (0.20, 0.001)], 24, axis="y", centre=(0.38, 0.15, 0.215))
    parts.append(smooth(new_object("fire_bottle", bottle, [mats["accent"]]), 40.0))
    # The pedals.
    pedals = [box("pedal", (DRIVER_X + dx - 0.03, 0.80, 0.19), (DRIVER_X + dx + 0.03, 0.815, 0.30),
                  mats["aluminium"], 0.004) for dx in (-0.09, 0.02, 0.12)]
    parts.append(join(pedals, "pedals"))
    _ = (bpy, prism, black, plate)
    return parts
