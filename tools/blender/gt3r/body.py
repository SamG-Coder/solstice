"""The 911 GT3 R's body: the shell, its wheel arches, and the openings cut in it."""

import math

import bmesh
import bpy

import spec
from common import apply_modifiers, boolean, box, new_object, smoothstep

# How far out from the car's middle each wheel's well begins.
WELL_INNER = 0.50
RING = 17  # points in half a cross-section, from the floor's middle up and over to the top's


def ring(y):
    """Half the body's cross-section at y: (x, z) from the middle of the floor, out to the
    sill, up the side, over the wing's crown, across the valley or the foot of the glass, up
    to the roof's rail and in to the middle of the top."""
    zb, zt = spec.ZB(y), spec.ZT(y)
    xw, xc, zc = spec.XW(y), spec.XC(y), spec.ZC(y)
    xv, zv = spec.XV(y), spec.ZV(y)
    xr, zr = spec.XR(y), zt - spec.RAIL_DROP(y)
    xc = min(xc, xw - 0.02)
    xv = min(xv, xc - 0.03)
    xr = min(xr, xv - 0.03)
    zc, zv, zr = max(zc, zb + 0.02), max(zv, zb + 0.02), max(zr, zb + 0.02)
    xs = xw * 0.955
    side = zc - zb
    # Subdividing rounds a crown down: the front wings' are set that much higher, to end where
    # the photograph has them.
    zc += 0.012 * smoothstep(0.95, 1.2, y) * (1.0 - smoothstep(1.95, 2.1, y))
    front = smoothstep(1.0, 1.25, y) * (1.0 - smoothstep(1.9, 2.1, y))
    # The cabin's glass bows outwards between its foot and the roof.
    cabin = smoothstep(-1.7, -1.1, y) * (1.0 - smoothstep(0.55, 1.0, y))
    rise_x, rise_z = xr - xv, zr - zv
    length = math.hypot(rise_x, rise_z) or 1.0
    out_x, out_z = rise_z / length, -rise_x / length
    bow = lambda t, by: (xv + t * rise_x + out_x * by * cabin,  # noqa: E731
                         zv + t * rise_z + out_z * by * cabin)
    # Over each wheel the wing is boxed: its side stands upright to a corner, and its top runs
    # nearly flat in to the crown.
    boxed = max(1.0 - abs(y - spec.FRONT_AT) / 0.62, 1.0 - abs(y + spec.REAR_AT) / 0.70, 0.0)
    boxed = min(boxed * 1.5, 1.0)
    corner = (xc + 0.62 * (xw - xc), zc - 0.032)
    # The rear wing's shoulder is rounder than the front's: it falls away further.
    rear = 1.0 if y < 0.0 else 0.0
    square = (xw - 0.028 - 0.025 * rear, zc - 0.020 - 0.018 * rear)
    return [
        (0.0, zb),
        (xs * 0.55, zb),
        (xs - 0.035, zb),
        (xs - 0.012 * (1.0 - boxed), zb + min(0.05, side * 0.2)),
        (xw - 0.010 - 0.022 * (1.0 - boxed) * cabin, zb + side * 0.30),
        (xw, zb + side * 0.58),
        (xw - 0.006 + 0.004 * boxed, zc - side * (0.20 - 0.09 * boxed)),
        (corner[0] + (square[0] - corner[0]) * boxed, corner[1] + (square[1] - corner[1]) * boxed),
        (xc, zc),
        (xv + (0.45 - 0.17 * front) * (xc - xv), zv + (0.55 + 0.05 * front) * (zc - zv)),
        (xv, zv),
        bow(0.40, 0.030),
        bow(0.80, 0.028),
        (xr, zr),
        (xr * 0.62, zt - 0.30 * (zt - zr)),
        (xr * 0.25, zt - 0.04 * (zt - zr)),
        (0.0, zt),
    ]


def stations():
    """Where along the car its cross-sections are: closer together at the ends, where the
    surface turns down to the bumpers."""
    ys, y = [], spec.TAIL
    while y < spec.NOSE - 1e-6:
        ys.append(y)
        near_end = min(y - spec.TAIL, spec.NOSE - y)
        y += 0.02 if near_end < 0.10 else (0.035 if near_end < 0.30 else 0.06)
    ys.append(spec.NOSE)
    return ys


def build_shell(paint):
    """The body as one closed skin, before anything is cut in it."""
    ys = stations()
    full = 2 * RING - 2

    def build(bm):
        rings = []
        for y in ys:
            half = ring(y)
            verts = [bm.verts.new((x, y, z)) for x, z in half]
            verts += [bm.verts.new((-x, y, z)) for x, z in reversed(half[1:-1])]
            rings.append(verts)
        for j in range(len(ys) - 1):
            for i in range(full):
                bm.faces.new((rings[j][i], rings[j][(i + 1) % full],
                              rings[j + 1][(i + 1) % full], rings[j + 1][i]))
        # The sill is a crisp edge, and so is the corner where the boxed wing's side meets its
        # top: pressed lines that subdividing must keep.
        crease = bm.edges.layers.float.new("crease_edge")
        for j in range(len(ys) - 1):
            mid = 0.5 * (ys[j] + ys[j + 1])
            boxed = max(1.0 - abs(mid - spec.FRONT_AT) / 0.75, 1.0 - abs(mid + spec.REAR_AT) / 0.8, 0.0)
            for i in range(full):
                half_index = i if i <= RING - 1 else full - i
                sharp = {2: 0.75, 3: 0.4}.get(half_index, 0.0)
                if half_index == 7:
                    sharp = 0.55 * min(boxed * 1.6, 1.0)
                if half_index == 10 and 1.05 < mid < 2.0:
                    sharp = 0.45
                edge = bm.edges.get((rings[j][i], rings[j + 1][i]))
                if edge and sharp:
                    edge[crease] = sharp
        # The ends close on a point a little way out.
        for verts, y, sign in ((rings[0], spec.TAIL, -1), (rings[-1], spec.NOSE, 1)):
            zs = [v.co.z for v in verts]
            centre = bm.verts.new((0.0, y + sign * 0.02, 0.5 * (min(zs) + max(zs))))
            for i in range(full):
                a, b = verts[i], verts[(i + 1) % full]
                bm.faces.new((a, b, centre) if sign < 0 else (b, a, centre))

    body = new_object("body", build, [paint])
    sub = body.modifiers.new("subdivide", "SUBSURF")
    sub.levels = 2
    sub.render_levels = 2
    apply_modifiers(body)
    return body


def cut_arches(body, liner):
    """The wheel arches: round wells close about each tyre, lined in black; the wing cut away
    behind each front wheel, and the bumper behind each rear one."""
    for name, (x, y, z) in spec.HUBS.items():
        front = name[-2] == "f"
        radius = (spec.FRONT_RADIUS if front else spec.REAR_RADIUS) + 0.026
        side = 1 if x > 0 else -1
        inner, outer = WELL_INNER, 1.25
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=96, radius=radius, depth=outer - inner,
            location=(side * (inner + outer) / 2, y, z), rotation=(0, math.pi / 2, 0))
        cutter = bpy.context.object
        cutter.data.materials.append(liner)
        boolean(body, cutter)
        if front:
            # Behind the front wheel the wing stops short: open, down to the sill.
            cut = box("cut", (side * 0.80 if side > 0 else side * 1.25, 0.735, 0.085),
                      (side * 1.25 if side > 0 else side * 0.80, y - 0.30, 0.455), liner)
        else:
            # Behind the rear wheel the bumper's corner is cut back to show the tyre.
            cut = box("cut", (side * 0.83 if side > 0 else side * 1.25, y - 0.62, 0.10),
                      (side * 1.25 if side > 0 else side * 0.83, y - 0.30, 0.40), liner)
        boolean(body, cut)
    return body


def check(surface):
    """How far the body is from the photographs: its top profile against the measured one, and
    its widths against the top view's. Printed, to correct the lines by."""
    measured_top = [(-1.65, 0.948), (-1.345, 1.050), (-0.98, 1.150), (-0.61, 1.226),
                    (-0.25, 1.249), (0.12, 1.214), (0.49, 1.073), (0.85, 0.917),
                    (1.126, 0.837), (1.40, 0.810), (1.675, 0.730), (1.858, 0.635),
                    (2.04, 0.547)]
    worst = 0.0
    for y, z in measured_top:
        highest = max((surface.top(x * 0.05, y) or 0.0) for x in range(0, 21))
        worst = max(worst, abs(highest - z))
        print(f"  top at {y:+.2f}: {highest:.3f} against {z:.3f}  ({(highest - z) * 1000:+.0f} mm)")
    measured_wide = [(-1.25, 1.025), (-0.3, 0.915), (0.4, 0.908), (1.25, 1.0195), (1.944, 0.913)]
    for y, w in measured_wide:
        widest = max((surface.side(y, 0.2 + k * 0.03) or 0.0) for k in range(0, 22))
        worst = max(worst, abs(widest - w))
        print(f"  half-width at {y:+.2f}: {widest:.3f} against {w:.3f}  ({(widest - w) * 1000:+.0f} mm)")
    print(f"  worst: {worst * 1000:.0f} mm")
    return worst
