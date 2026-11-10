"""The 911 GT3 R's wheels: racing slicks on forged centre-lock rims, over steel discs.

Each wheel is one object with its origin at its hub and its outer face towards +X, turned
round for the left-hand side. Its caliper is an object of its own, with its origin 30 mm
inboard of the hub: it rides and steers with the wheel, and does not turn with it.
"""

import math

import bpy

import spec
from common import join, lathe, new_object, rounded, smooth

# How far inboard of the hub a caliper's origin is: what marks it as not turning.
UPRIGHT = 0.03


def tyre_profile(radius, width):
    """Half... the whole section of a slick, across the wheel: bead, sidewall, shoulder, tread.
    The sidewall bows out to its widest a third of the way up; the tread is all but flat."""
    half, rim = width / 2, spec.RIM_RADIUS
    wall = radius - rim
    right = [
        (half - 0.022, rim - 0.004),           # the bead, tucked behind the rim's lip
        (half - 0.010, rim + 0.012),
        (half + 0.002, rim + wall * 0.38),      # the widest
        (half - 0.004, rim + wall * 0.70),
        (half - 0.016, radius - 0.016),         # the shoulder
        (half - 0.040, radius - 0.0035),
        (half * 0.45, radius - 0.0008),
        (0.0, radius),
    ]
    whole = [(-x, r) for x, r in right] + list(reversed(right[:-1]))
    return rounded(whole, 6, closed=False)


def build_wheel(name, radius, width, front, mats, nut_colour="nut_red"):
    half, rim = width / 2, spec.RIM_RADIUS
    # Where the spokes' face is, across the wheel: the rears are dished deeper.
    face = half - (0.045 if front else 0.075)
    hub_face = face - 0.030
    disc_r = 0.195 if front else 0.185

    def tyre(bm):
        lathe(bm, tyre_profile(radius, width), 96)
        # A moulded band on each sidewall, where a maker's lettering stands proud.
        for side in (-1, 1):
            x = side * (half + 0.0035)
            wall = radius - rim
            lathe(bm, [(x - side * 0.004, rim + wall * 0.30), (x, rim + wall * 0.34),
                       (x, rim + wall * 0.62), (x - side * 0.004, rim + wall * 0.66)], 96)

    def barrel(bm):
        # From the inner lip, down into the well, and out to the outer lip and its face.
        lip = rim + 0.014
        profile = [(-half + 0.006, lip), (-half + 0.012, lip), (-half + 0.020, rim - 0.002),
                   (-half + 0.07, rim - 0.004), (-half + 0.12, rim - 0.007),
                   (face - 0.05, rim - 0.007), (face, rim - 0.006), (half - 0.022, rim - 0.004),
                   (half - 0.012, lip), (half - 0.004, lip), (half - 0.004, rim + 0.002),
                   (half - 0.014, rim - 0.014), (face + 0.004, rim - 0.026)]
        lathe(bm, profile, 96)
        # The inside of the barrel, seen through the spokes.
        lathe(bm, [(-half + 0.012, rim - 0.008), (-half + 0.12, rim - 0.013),
                   (face - 0.05, rim - 0.013)], 96)

    def spokes(bm):
        # Ten slender spokes, each forked where it meets the rim, sweeping in to the hub.
        count = 10
        inner_r, fork_r, outer_r = 0.060, 0.150, rim - 0.011

        def beam(points, width0, width1, depth0, depth1):
            """A tapered bar through (radius, angle, x) points."""
            rings = []
            n = len(points)
            for i, (r, a, x) in enumerate(points):
                t = i / (n - 1)
                w = width0 + (width1 - width0) * t
                d = depth0 + (depth1 - depth0) * t
                c, s = math.cos(a), math.sin(a)
                ring = []
                for dx, dw in ((0.0, -w / 2), (0.0, w / 2), (-d, w / 2 * 0.7), (-d, -w / 2 * 0.7)):
                    # dw is across the spoke: along the tangent.
                    ring.append(bm.verts.new((x + dx, r * c - dw * s, r * s + dw * c)))
                rings.append(ring)
            for a, b in zip(rings, rings[1:]):
                for k in range(4):
                    bm.faces.new((a[k], a[(k + 1) % 4], b[(k + 1) % 4], b[k]))
            bm.faces.new(list(reversed(rings[0])))
            bm.faces.new(rings[-1])

        for k in range(count):
            a = math.tau * k / count
            # The stem, from the hub out to the fork: set back at the hub, forward at the rim.
            stem = [(inner_r + (fork_r - inner_r) * t, a,
                     hub_face + (face - 0.006 - hub_face) * (t ** 0.8)) for t in (0, 0.33, 0.66, 1)]
            beam(stem, 0.026, 0.017, 0.040, 0.026)
            for turn in (-1, 1):
                prong = [(fork_r + (outer_r - fork_r) * t, a + turn * 0.085 * t,
                          face - 0.006 + 0.006 * t) for t in (0, 0.5, 1)]
                beam(prong, 0.014, 0.012, 0.026, 0.020)
        # The hub the spokes grow from.
        lathe(bm, [(hub_face - 0.045, 0.050), (hub_face - 0.040, 0.078), (hub_face - 0.004, 0.074),
                   (hub_face + 0.004, 0.062), (hub_face + 0.004, 0.040)], 48)

    def nut(bm):
        # The one nut that holds a racing wheel on, and the cap over the hub's end.
        lathe(bm, [(hub_face + 0.002, 0.044), (hub_face + 0.030, 0.042), (hub_face + 0.036, 0.036),
                   (hub_face + 0.036, 0.024)], 6)
        lathe(bm, [(hub_face + 0.030, 0.026), (hub_face + 0.046, 0.022), (hub_face + 0.050, 0.012),
                   (hub_face + 0.050, 0.0005)], 32)

    def disc(bm):
        # Two friction rings with the vanes' gap between them, on an aluminium bell.
        x = hub_face - 0.075
        for side in (-1, 1):
            a, b = x + side * 0.006, x + side * 0.017
            lathe(bm, [(a, disc_r - 0.060), (a, disc_r - 0.004), (b, disc_r - 0.001),
                       (b, disc_r - 0.062), (a, disc_r - 0.060)], 64)
        lathe(bm, [(x - 0.005, disc_r - 0.056), (x - 0.005, disc_r - 0.008),
                   (x + 0.005, disc_r - 0.008), (x + 0.005, disc_r - 0.056)], 64)

    def slots(bm):
        # Curved slots across the disc's face, which wipe the pads: shallow dark grooves.
        x = hub_face - 0.075 + 0.0175
        for k in range(12):
            a0 = math.tau * k / 12
            pts = []
            for t in range(7):
                r = disc_r - 0.056 + 0.050 * t / 6
                a = a0 + 0.45 * (t / 6) ** 1.3
                pts.append((r, a))
            for (r0, b0), (r1, b1) in zip(pts, pts[1:]):
                quad = []
                for r, b, w in ((r0, b0, -0.0025), (r1, b1, -0.0025), (r1, b1, 0.0025), (r0, b0, 0.0025)):
                    quad.append(bm.verts.new((x, (r + w) * math.cos(b), (r + w) * math.sin(b))))
                bm.faces.new(quad)

    def valve(bm):
        lathe(bm, [(0.0, 0.004), (0.035, 0.004), (0.038, 0.002), (0.038, 0.0003)], 8, axis="x",
              centre=(face - 0.01, 0.0, rim - 0.034))

    def bell(bm):
        x = hub_face - 0.075
        lathe(bm, [(x + 0.017, disc_r - 0.060), (x + 0.020, disc_r - 0.085),
                   (hub_face - 0.042, 0.088), (hub_face - 0.040, 0.050)], 48)

    pieces = [
        smooth(new_object("tyre", tyre, [mats["rubber"]]), 60.0),
        smooth(new_object("barrel", barrel, [mats["wheel"]]), 40.0),
        smooth(new_object("spokes", spokes, [mats["wheel"]]), 38.0),
        smooth(new_object("nut", nut, [mats[nut_colour]]), 25.0),
        smooth(new_object("slots", slots, [mats["liner"]]), 30.0),
        smooth(new_object("valve", valve, [mats["aluminium"]]), 30.0),
        smooth(new_object("disc", disc, [mats["disc"]]), 30.0),
        smooth(new_object("bell", bell, [mats["aluminium"]]), 40.0),
    ]
    return join(pieces, name)


def build_caliper(name, hub, front, mats):
    """A six-piston caliper clasping the disc: behind the axle at the front, ahead of it at
    the rear."""
    hx, hy, hz = hub
    side = 1 if hx > 0 else -1
    width = spec.FRONT_TYRE if front else spec.REAR_TYRE
    face = width / 2 - (0.045 if front else 0.075)
    x = face - 0.030 - 0.075          # the disc's middle, from the hub
    disc_r = 0.195 if front else 0.185
    middle = math.pi if front else 0.0
    sweep = math.radians(72 if front else 62)

    def body(bm):
        profile = [(x - 0.048, disc_r - 0.058), (x - 0.050, disc_r + 0.008),
                   (x - 0.030, disc_r + 0.017), (x + 0.030, disc_r + 0.017),
                   (x + 0.046, disc_r + 0.008), (x + 0.044, disc_r - 0.058),
                   (x + 0.024, disc_r - 0.060), (x + 0.024, disc_r + 0.003),
                   (x - 0.024, disc_r + 0.003), (x - 0.024, disc_r - 0.060)]
        profile.append(profile[0])
        lathe(bm, profile, 18, arc=sweep, start=middle - sweep / 2, capped=True)

    caliper = smooth(new_object(name, body, [mats["caliper"]]), 35.0)
    # Built about the hub with the wheel's outer face towards +X: turned round for the left.
    for v in caliper.data.vertices:
        v.co.x *= side
    if side < 0:
        caliper.data.flip_normals()
    caliper.location = (hx - side * UPRIGHT, hy, hz)
    for v in caliper.data.vertices:
        v.co.x += side * UPRIGHT
    return caliper


def build(mats):
    made = []
    for name, hub in spec.HUBS.items():
        front = name[-2] == "f"
        wheel = build_wheel(name, spec.FRONT_RADIUS if front else spec.REAR_RADIUS,
                            spec.FRONT_TYRE if front else spec.REAR_TYRE, front, mats,
                            "nut_red" if hub[0] < 0 else "nut_blue")
        wheel.location = hub
        if hub[0] < 0:
            wheel.rotation_euler = (0, 0, math.pi)
        made.append(wheel)
        made.append(build_caliper(name.replace("wheel", "caliper"), hub, front, mats))
    bpy.context.view_layer.update()
    return made
