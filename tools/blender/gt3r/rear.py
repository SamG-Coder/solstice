"""The 911 GT3 R's rear end: ducktail, light bar, bumper, exhausts, diffuser, and the wing on
its swan-neck mounts.

From the side view M22_3548 (measured), the rear view M22_3536 and the rear three-quarter
views M22_3537 and M22_3549.
"""

import math

import spec
from common import box, ellipse, join, line, new_object, prism, rounded, smooth, tube

# The wing, from the side view: its end plates reach from 1.992 to 2.412 m behind the middle of
# the wheelbase, and from 1.031 to 1.275 m up.
PLATE_FRONT, PLATE_REAR, PLATE_LOW, PLATE_TOP = -1.992, -2.412, 1.031, 1.275
WING_HALF_SPAN = 0.955
MOUNT_X = 0.48   # from the rear view M22_3536 and the top view
# The opening across the rear bumper, seen from behind: (x, z).
BUMPER = [(-0.60, 0.565), (-0.705, 0.48), (-0.68, 0.365), (-0.30, 0.345), (0.0, 0.34),
          (0.30, 0.345), (0.68, 0.365), (0.705, 0.48), (0.60, 0.565), (0.0, 0.585)]
# The grille in the engine cover, seen from above: (x, y).
GRILLE = [(-0.43, -1.555), (0.43, -1.555), (0.47, -1.78), (0.41, -1.97), (-0.41, -1.97),
          (-0.47, -1.78)]


def aerofoil(chord, thickness, camber, steps=18):
    """A wing's section, upside down, as racing wings are: (along, up) from the leading edge."""
    upper, lower = [], []
    for k in range(steps + 1):
        t = k / steps
        half = thickness * chord * (1.4845 * math.sqrt(t) - 0.63 * t - 1.758 * t * t
                                    + 1.4215 * t ** 3 - 0.5075 * t ** 4)
        bend = -camber * chord * math.sin(math.pi * t ** 0.85)
        upper.append((t * chord, bend + half))
        lower.append((t * chord, bend - half))
    return upper + list(reversed(lower[1:-1]))


def strip(centre, widths):
    """The outline of a band along a path of (y, z) points, `widths[i]` wide at each."""
    left, right = [], []
    n = len(centre)
    for i, (y, z) in enumerate(centre):
        (y0, z0), (y1, z1) = centre[max(i - 1, 0)], centre[min(i + 1, n - 1)]
        ty, tz = y1 - y0, z1 - z0
        length = math.hypot(ty, tz) or 1.0
        ny, nz = -tz / length, ty / length
        left.append((y + ny * widths[i] / 2, z + nz * widths[i] / 2))
        right.append((y - ny * widths[i] / 2, z - nz * widths[i] / 2))
    return left + list(reversed(right))


def build(shell, surface, mats):
    parts = []
    liner, black, carbon, accent, seal = (mats[k] for k in ("liner", "black", "carbon", "accent", "seal"))
    above = lambda x, y: ((x, y, 3.0), (0, 0, -1))  # noqa: E731
    behind = lambda x, z: ((x, -4.0, z), (0, 1, 0))  # noqa: E731

    def add(obj):
        if obj is not None:
            parts.append(obj)
        return obj

    # ---- The ducktail: the engine cover's edge, swept up and back into a lip.
    tip_y, tip_z = spec.DUCKTAIL_TIP
    section = [(-2.06, 0.889), (-2.20, 0.889), (tip_y, tip_z + 0.004), (tip_y + 0.004, tip_z - 0.004),
               (-2.24, 0.835), (-2.17, 0.79), (-2.06, 0.80)]

    def ducktail(bm):
        # The section, narrowing and dropping a little towards each end, where it meets the
        # wings.
        rings = []
        for k in range(-12, 13):
            t = k / 12
            x = 0.86 * math.sin(t * math.pi / 2) if abs(t) < 1 else 0.86 * t
            pull = 1.0 - 0.55 * abs(t) ** 3
            drop = 0.035 * abs(t) ** 2.5
            rings.append([bm.verts.new((x, -2.06 + (y + 2.06) * pull, z - drop)) for y, z in section])
        for a, b in zip(rings, rings[1:]):
            for i in range(len(section)):
                j = (i + 1) % len(section)
                bm.faces.new((a[i], a[j], b[j], b[i]))
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    add(smooth(new_object("ducktail", ducktail, [mats["paint"]]), 50.0))

    # ---- The light bar across the tail, and the rain light.
    for z, width, name in ((0.735, 0.018, "light_bar"),):
        rays = [behind(x, z - 0.02 * (x / 0.86) ** 4) for (x,) in line((-0.86,), (0.86,), 60)]
        add(surface.ribbon(name, rays, width, 0.002, mats["tail"]))
        add(surface.ribbon(f"{name}_surround", rays, width + 0.02, 0.0012, black))
    add(surface.patch("rain_light", [(-0.045, 0.635), (0.045, 0.635), (0.045, 0.665), (-0.045, 0.665)],
                      behind, mats["tail"], 0.002, 1))

    # ---- The bumper: one wide opening with mesh behind, picked out in the accent colour, and
    # the two tailpipes through its middle.
    opening = rounded(BUMPER, 8)
    surface.recess(shell, opening, behind, 0.11, liner)
    add(surface.patch("bumper_mesh", opening, behind, mats["mesh"], -0.075))
    add(surface.rim("bumper_surround", opening, behind, 0.012, 0.002, accent))
    for side in (-1, 1):
        add(tube("tailpipe", [(side * 0.058, -2.02, 0.44), (side * 0.058, -2.30, 0.44)], 0.040,
                 mats["steel"], 20))
        add(tube("tailpipe_bore", [(side * 0.058, -2.25, 0.44), (side * 0.058, -2.302, 0.44)], 0.033,
                 liner, 20))
    add(tube("tow_loop", [(-0.38, -2.20, 0.50), (-0.38, -2.31, 0.505), (-0.38, -2.32, 0.465),
                          (-0.38, -2.21, 0.46)], 0.007, accent, 8, closed=True))

    # ---- The diffuser: a tray rising from under the engine to the tail, with fences in it.
    tray = [(-1.42, 0.066), (-2.375, 0.305), (-2.375, 0.293), (-1.42, 0.054)]
    add(prism("diffuser", tray, "x", -0.83, 0.83, carbon))
    fence = [(-1.50, 0.07), (-2.385, 0.292), (-2.385, 0.105), (-2.12, 0.05), (-1.50, 0.045)]
    fences = [prism("fence", fence, "x", x - 0.005, x + 0.005, carbon)
              for x in (-0.83, -0.62, -0.41, -0.205, 0.0, 0.205, 0.41, 0.62, 0.83)]
    add(join(fences, "diffuser_fences"))

    # ---- The engine cover: its grille, and its edge.
    grille = rounded(GRILLE, 6)
    surface.recess(shell, grille, above, 0.03, liner)
    add(surface.patch("cover_mesh", grille, above, mats["mesh"], -0.018))
    for k in range(6):
        y = -1.60 - k * 0.068
        half = 0.42 + 0.03 * math.sin(math.pi * k / 5)
        add(surface.fin("cover_slat", [above(x, y) for (x,) in line((-half,), (half,), 24)],
                        lambda t: -0.012, 0.016, carbon, out=(0, -0.6, 1)))
    cover = [(-0.50, -1.50), (0.50, -1.50), (0.585, -1.80), (0.58, -2.07), (-0.58, -2.07),
             (-0.585, -1.80)]
    add(surface.rim("cover_gap", rounded(cover, 6), above, 0.005, 0.0012, seal))

    # ---- The wing: an aerofoil the width of the car, end plates, and two swan-neck mounts
    # that hold it from above, where they disturb its working surface least.
    chord = 0.37
    lead_y, lead_z = PLATE_FRONT - 0.03, 1.218
    pitch = math.radians(-9.0)  # trailing edge up
    section = []
    for along, up in aerofoil(chord, 0.11, 0.07):
        y = lead_y - (along * math.cos(pitch) - up * math.sin(pitch))
        z = lead_z + (-along * math.sin(pitch) + up * math.cos(pitch)) - 0.0
        section.append((y, z))
    wing = prism("wing", section, "x", -WING_HALF_SPAN, WING_HALF_SPAN, carbon)
    add(smooth(wing, 30.0))
    # The lip along its trailing edge.
    trail_y = lead_y - chord * math.cos(pitch)
    trail_z = lead_z - chord * math.sin(pitch)
    add(box("wing_gurney", (-WING_HALF_SPAN, trail_y - 0.004, trail_z - 0.002),
            (WING_HALF_SPAN, trail_y + 0.002, trail_z + 0.022), carbon))
    plate = rounded([(PLATE_FRONT, PLATE_TOP - 0.02), (PLATE_FRONT - 0.03, PLATE_TOP),
                     (PLATE_REAR + 0.03, PLATE_TOP), (PLATE_REAR, PLATE_TOP - 0.03),
                     (PLATE_REAR, PLATE_LOW + 0.03), (PLATE_REAR + 0.03, PLATE_LOW),
                     (PLATE_FRONT - 0.10, PLATE_LOW), (PLATE_FRONT, PLATE_LOW + 0.09)], 5)
    for side, tag in ((1, "right"), (-1, "left")):
        x = side * (WING_HALF_SPAN + 0.004)
        add(prism(f"wing_plate_{tag}", plate, "x", x - 0.005, x + 0.005, carbon))
        add(tube(f"wing_plate_edge_{tag}", [(x, y, z) for y, z in plate], 0.0055, accent, 6, closed=True))
        # The mount: up from the engine cover, over the wing's leading edge, and down onto it.
        path = [(-1.775, 0.86), (-1.80, 0.98), (-1.84, 1.11), (-1.89, 1.215), (-1.955, 1.268),
                (-2.04, 1.280), (-2.12, 1.270), (-2.165, 1.256)]
        path = rounded(path, 5, closed=False)
        widths = [0.10 - 0.070 * min(i / (len(path) * 0.5), 1.0) for i in range(len(path))]
        add(prism(f"wing_mount_{tag}", strip(path, widths), "x", side * MOUNT_X - 0.007,
                  side * MOUNT_X + 0.007, black, 0.003))
        # Its foot on the wing.
        add(box(f"wing_mount_foot_{tag}", (side * MOUNT_X - 0.03, -2.21, 1.243),
                (side * MOUNT_X + 0.03, -2.09, 1.258), black, 0.004))
    return parts
