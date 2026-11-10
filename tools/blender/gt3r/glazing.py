"""The cabin's hollow, the openings for its glass, and the glass and seals in them."""

import math

import bmesh  # noqa: F401

import body
import spec
from common import boolean, mirrored, new_object, prism, rounded

# The windscreen and the rear screen, seen from above: (x, y).
# The windscreen, seen from above: (x, y), its right half from the middle of its foot round to
# the middle of its top. From the top view M22_3546: the foot sweeps back 0.27 m from its middle
# to its corners, and the sides run back and in along the pillars.
SCREEN_HALF = [(0.0, 0.983), (0.353, 0.909), (0.606, 0.803), (0.675, 0.715), (0.70, 0.625),
               (0.672, 0.505), (0.61, 0.415), (0.54, 0.315), (0.485, 0.235), (0.42, 0.205),
               (0.25, 0.19), (0.0, 0.185)]
SCREEN = [(-x, y) for x, y in reversed(SCREEN_HALF[1:-1])] + SCREEN_HALF
SCREEN_TOP = 0.185


def side_glass_front(surface):
    """The front edge of the right-hand side glass, from the door's corner up to the roof, as
    points on the body: (x, y, z)."""
    edge = rounded([(0.565, 0.830), (0.455, 0.895), (0.303, 0.993), (0.150, 1.079), (0.074, 1.117)],
                   8, closed=False)
    points = []
    for y, z in edge:
        x = surface.side(y, z, 1)
        if x is not None:
            points.append((x, y, z))
    return points


def pillar_width(surface, screen):
    """The narrowest the pillar between the windscreen and the side glass is, on the body."""
    from mathutils import Vector
    glass = [Vector(p) for p in side_glass_front(surface)]
    edge = []
    for x, y in screen:
        if x > 0.3 and SCREEN_TOP + 0.02 < y < 0.80:
            z = surface.top(x, y)
            if z is not None:
                edge.append(Vector((x, y, z)))
    if not glass or not edge:
        return 0.0
    return min(min((g - e).length for e in edge) for g in glass[2:-2])
REAR_SCREEN = [(-0.40, -0.69), (0.0, -0.66), (0.40, -0.69), (0.465, -1.0), (0.435, -1.36),
               (0.0, -1.45), (-0.435, -1.36), (-0.465, -1.0)]
# How thick the shell is left round the cabin.
SHELL = 0.028
CABIN = (-1.30, 0.97)
CABIN_FLOOR = 0.13
# How thick the wheel housings' walls are, between a well and the cabin.
HOUSING = 0.04


def inner_ring(y, by=SHELL):
    """The cabin's cross-section at y: the body's, moved in by the shell's thickness, or by
    `by` for something that must stand clear of the cabin's wall."""
    half = body.ring(y)
    full = half + [(-x, z) for x, z in reversed(half[1:-1])]
    n = len(full)
    moved = []
    for i, (x, z) in enumerate(full):
        (x0, z0), (x1, z1) = full[i - 1], full[(i + 1) % n]
        # The outline runs anticlockwise seen from ahead, so inwards is to the left of travel.
        tx, tz = x1 - x0, z1 - z0
        length = (tx * tx + tz * tz) ** 0.5 or 1.0
        nx, nz = -tz / length, tx / length
        moved.append((x + nx * by, max(z + nz * by, CABIN_FLOOR)))
    return moved


def build_cavity(mat):
    ys = []
    y = CABIN[0]
    while y < CABIN[1]:
        ys.append(y)
        y += 0.06
    ys.append(CABIN[1])

    def housed(x, y, z):
        """The cabin's wall steps in round each wheel, as a wheel housing does: the wells cut
        for the wheels must not break through into it."""
        for axle, radius in ((spec.FRONT_AT, spec.FRONT_RADIUS), (-spec.REAR_AT, spec.REAR_RADIUS)):
            if math.hypot(y - axle, z - radius) < radius + 0.026 + HOUSING:
                return math.copysign(min(abs(x), body.WELL_INNER - HOUSING), x)
        # And round what is sunk into the body's sides: the wing's cut-away and vent behind
        # each front wheel, and the intake behind each door.
        if 0.68 <= y <= 0.98 and z < 0.76:
            return math.copysign(min(abs(x), 0.76), x)
        if -0.90 <= y <= -0.50 and 0.24 <= z <= 0.80:
            return math.copysign(min(abs(x), 0.79), x)
        return x

    def build(bm):
        rings = [[bm.verts.new((housed(x, y, z), y, z)) for x, z in inner_ring(y)] for y in ys]
        n = len(rings[0])
        for a, b in zip(rings, rings[1:]):
            for i in range(n):
                bm.faces.new((a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]))
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    return new_object("cavity", build, [mat])


def build(shell, surface, mats):
    """Hollows the cabin, opens the windows, and glazes them. Returns the glass and seals."""
    parts = []
    boolean(shell, build_cavity(mats["liner"]))

    side = rounded(spec.SIDE_GLASS, 8)
    screen, rear = rounded(SCREEN, 8), rounded(REAR_SCREEN, 8)
    print(f"glazing: the windscreen pillar is {pillar_width(surface, screen) * 1000:.0f} mm at its narrowest")
    # Against the side view M22_3548: where the windscreen's glass begins, seen from the side,
    # at three heights; and where the red seal of the door's opening is, at four places.
    edge = [(y, surface.top(x, y)) for x, y in screen if x > 0.3 and 0.2 < y < 0.75]
    edge = [(y, z) for y, z in edge if z is not None]
    for height, measured in ((0.955, 0.455), (0.909, 0.527), (0.864, 0.611)):
        near = min(edge, key=lambda p: abs(p[1] - height))
        print(f"glazing: windscreen edge at {height:.3f} m up is {near[0]:+.3f} m along, "
              f"photo {measured:+.3f} ({(near[0] - measured) * 1000:+.0f} mm)")
    for along, measured in ((-0.269, 1.170), (0.150, 1.087), (0.303, 1.001), (0.455, 0.902)):
        top = max(z for y, z in side if abs(y - along) < 0.02)
        print(f"glazing: side glass top at {along:+.3f} m along is {top:.3f} m up, "
              f"photo's seal {measured:.3f} ({(top - measured) * 1000:+.0f} mm)")
    cutter = prism("cut", side, "x", 0.47, 1.4, mats["seal"])
    other = mirrored(cutter)
    boolean(shell, cutter)
    boolean(shell, other)
    boolean(shell, prism("cut", screen, "z", 0.80, 1.6, mats["seal"]))
    boolean(shell, prism("cut", rear, "z", 0.86, 1.6, mats["seal"]))

    glass, seal = mats["glass"], mats["seal"]
    above = lambda x, y: ((x, y, 3.0), (0, 0, -1))  # noqa: E731

    def grown(outline, by):
        cx = sum(p[0] for p in outline) / len(outline)
        cy = sum(p[1] for p in outline) / len(outline)
        return [(cx + (x - cx) * by, cy + (y - cy) * by) for x, y in outline]

    # Each pane fills its opening, just under the skin, with a seal over the join.
    parts.append(surface.patch("glass_windscreen", screen, above, glass, -0.002))
    parts.append(surface.rim("seal_windscreen", screen, above, 0.020, 0.0015, seal))
    parts.append(surface.patch("glass_rear", rear, above, glass, -0.002))
    parts.append(surface.rim("seal_rear", rear, above, 0.020, 0.0015, seal))
    for sign, name in ((1, "right"), (-1, "left")):
        beside = lambda y, z, sign=sign: ((sign * 2.0, y, z), (-sign, 0, 0))  # noqa: E731
        parts.append(surface.patch(f"glass_side_{name}", side, beside, glass, -0.002))
        parts.append(surface.rim(f"seal_side_{name}", side, beside, 0.018, 0.0015, seal))
        # The pillar behind the door, across the glass.
        low = spec.ZV(spec.PILLAR_AT) + 0.012
        high = max(z for y, z in side if abs(y - spec.PILLAR_AT) < 0.03) - 0.004
        rays = [beside(spec.PILLAR_AT, low + (high - low) * k / 16) for k in range(17)]
        parts.append(surface.ribbon(f"pillar_{name}", rays, 0.055, 0.0018, seal))
    return parts
