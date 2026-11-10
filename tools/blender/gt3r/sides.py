"""The 911 GT3 R's flanks and roof: intakes, vents, skirts, doors, mirrors, seams, fasteners.

From the side view M22_3548, the top view M22_3546 and the three-quarter views.
"""

import math

import bpy

import spec
from common import boolean, box, ellipse, join, line, prism, rounded, select_only, smooth, tube

# The vent at the back of each front wing, where the arch's air leaves: (y, z).
WING_VENT = [(0.742, 0.475), (0.865, 0.492), (0.878, 0.70), (0.752, 0.692)]
# The vent in the bumper behind each rear wheel.
REAR_VENT = [(-1.665, 0.42), (-1.665, 0.63), (-1.80, 0.615), (-1.815, 0.44)]


def rounded_rect(cx, cy, w, h, r=0.01):
    pts = []
    for (sx, sy), start in (((1, 1), 0), ((-1, 1), 90), ((-1, -1), 180), ((1, -1), 270)):
        for k in range(5):
            a = math.radians(start + 22.5 * k)
            pts.append((cx + sx * (w / 2 - r) + r * math.cos(a), cy + sy * (h / 2 - r) + r * math.sin(a)))
    return pts


def race_number(surface, text, side, at, size, mat):
    """A race number on the door: lettering laid onto the body's side."""
    data = bpy.data.curves.new("number", "FONT")
    data.body = text
    data.size = size
    data.align_x = "CENTER"
    data.align_y = "CENTER"
    obj = bpy.data.objects.new("number", data)
    bpy.context.collection.objects.link(obj)
    # Standing up, facing out of the car's side, reading from its left as it is looked at.
    obj.rotation_euler = (math.pi / 2, 0.0, side * math.pi / 2)
    obj.location = (side * 1.4, at[0], at[1])
    select_only(obj)
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    wrap = obj.modifiers.new("wrap", "SHRINKWRAP")
    wrap.target = surface.obj
    wrap.wrap_method = "PROJECT"
    wrap.use_project_x, wrap.use_project_y, wrap.use_project_z = True, False, False
    wrap.use_negative_direction = True
    wrap.use_positive_direction = True
    wrap.offset = 0.0016
    select_only(obj)
    bpy.ops.object.modifier_apply(modifier=wrap.name)
    obj.data.materials.append(mat)
    return obj


def build(shell, surface, mats):
    parts = []
    liner, black, carbon, accent, seal = (mats[k] for k in ("liner", "black", "carbon", "accent", "seal"))
    above = lambda x, y: ((x, y, 3.0), (0, 0, -1))  # noqa: E731

    def add(obj):
        if obj is not None:
            parts.append(obj)
        return obj

    for side, tag in ((1, "right"), (-1, "left")):
        beside = lambda y, z, side=side: ((side * 3.0, y, z), (-side, 0, 0))  # noqa: E731

        # ---- The intake in the rear wing, behind the door: it feeds the engine and coolers.
        intake = rounded(spec.SIDE_INTAKE, 8)
        surface.recess(shell, intake, beside, 0.11, liner)
        add(surface.patch(f"side_intake_mesh_{tag}", intake, beside, mats["mesh"], -0.07))
        add(surface.rim(f"side_intake_lip_{tag}", intake, beside, 0.016, 0.0015, carbon))

        # ---- The vent behind the front wheel, and the one behind the rear: louvred, in red.
        for name, outline, slats in (("wing_vent", WING_VENT, 5), ("rear_vent", REAR_VENT, 5)):
            shape = rounded(outline, 5)
            surface.recess(shell, shape, beside, 0.05, liner)
            ys = [p[0] for p in outline]
            zs = [p[1] for p in outline]
            for k in range(slats):
                t = (k + 0.5) / slats
                z = min(zs) + 0.02 + (max(zs) - min(zs) - 0.04) * t
                lean = 0.012 if name == "wing_vent" else -0.008
                rays = [beside(y, z + lean * (y - min(ys)) / (max(ys) - min(ys)))
                        for (y,) in line((min(ys) + 0.012,), (max(ys) - 0.012,), 10)]
                add(surface.fin(f"{name}_slat_{tag}_{k}", rays, lambda t: -0.03, 0.02, accent,
                                out=(side, 0, 0.35)))

        # ---- The skirt along the sill, widening towards the rear wheel, its edge picked out.
        plan = [(0.80, 0.915), (0.957, 0.915), (0.962, 0.2), (0.985, -0.55), (1.022, -0.875),
                (0.80, -0.875)]
        add(prism(f"skirt_{tag}", [(side * x, y) for x, y in plan], "z", 0.044, 0.078, carbon, 0.004))
        add(tube(f"skirt_edge_{tag}", [(side * (x + 0.002), y, 0.05) for x, y in
                                       ((0.957, 0.915), (0.962, 0.2), (0.985, -0.55), (1.022, -0.875))],
                 0.008, accent, 8))

        # ---- The door: its shut lines, and the pull let into it.
        front_line = [beside(0.627 + 0.073 * ((0.835 - z) / 0.665) ** 1.3, z)
                      for (z,) in line((0.835,), (0.17,), 40)]
        rear_line = [beside(y, z) for y, z in rounded(
            [(-0.525, 0.857), (-0.535, 0.60), (-0.552, 0.36), (-0.52, 0.22), (-0.42, 0.168)],
            12, closed=False)]
        foot = [beside(y, 0.168) for (y,) in line((-0.42,), (0.70,), 40)]
        for rays in (front_line, rear_line, foot):
            add(surface.ribbon(f"door_gap_{tag}", rays, 0.005, 0.0012, seal))
        pull = rounded_rect(-0.38, 0.64, 0.17, 0.034, 0.014)
        surface.recess(shell, pull, beside, 0.016, black, cuts=1)
        add(surface.fin(f"door_pull_{tag}", [beside(y, 0.642) for (y,) in line((-0.455,), (-0.305,), 8)],
                        lambda t: -0.004, 0.012, mats["aluminium"], out=(side, 0, 0)))

        # ---- Seams where the bumpers meet the wings, and the fasteners along them.
        for y0, y1, z0, z1 in ((1.70, 1.66, 0.62, 0.20), (-1.88, -1.86, 0.78, 0.42)):
            add(surface.ribbon(f"seam_{tag}", [beside(y0 + (y1 - y0) * t / 20, z0 + (z1 - z0) * t / 20)
                                               for t in range(21)], 0.004, 0.0012, seal))
            for t in (0.1, 0.5, 0.9):
                y, z = y0 + (y1 - y0) * t + 0.025, z0 + (z1 - z0) * t
                add(surface.patch(f"fastener_{tag}", ellipse(y, z, 0.007, 0.007, 10), beside,
                                  mats["aluminium"], 0.0015, 0))

        # ---- The edge of each arch: where the wing is cut, a dark lip.
        for hub_y, radius in ((spec.FRONT_AT, spec.FRONT_RADIUS), (-spec.REAR_AT, spec.REAR_RADIUS)):
            z0 = spec.FRONT_RADIUS if hub_y > 0 else spec.REAR_RADIUS
            rays = []
            for k in range(-2, 39):
                a = math.radians(k * 5.0)
                rays.append(beside(hub_y + (radius + 0.033) * math.cos(a), z0 + (radius + 0.033) * math.sin(a)))
            add(surface.ribbon(f"arch_lip_{tag}", rays, 0.012, 0.0015, seal))

        # ---- The mirror: a pod on a stalk from the door, flat at the back where the glass is.
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0,
                                             location=(side * 0.945, 0.47, 0.848))
        pod = bpy.context.object
        pod.scale = (0.072, 0.125, 0.050)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        pod.data.materials.append(carbon)
        boolean(pod, box("cut", (side * 0.945 - 0.2, 0.20, 0.7), (side * 0.945 + 0.2, 0.405, 1.0), mats["chrome"]))
        pod.name = f"mirror_{tag}"
        smooth(pod, 50.0)
        add(pod)
        stalk = [(side * 0.84, 0.50, 0.812), (side * 0.875, 0.495, 0.826), (side * 0.91, 0.485, 0.836)]
        add(tube(f"mirror_stalk_{tag}", stalk, 0.013, black, 8))
        # The cap of it, in the accent colour, as the works cars wear it.
        add(box(f"mirror_cap_{tag}", (side * 1.0 - 0.012, 0.42, 0.838), (side * 1.0 + 0.012, 0.50, 0.858),
                accent, 0.006))

        # ---- The race number, on the door.
        number = race_number(surface, "92", side, (0.12, 0.50), 0.36, mats["number"])
        number.name = f"number_{tag}"
        add(number)

        # ---- The fuel filler and its vent, in the quarter panel.
        for (y, z, r), mat in (((-0.80, 0.935, 0.034), mats["aluminium"]), ((-0.80, 0.935, 0.024), black),
                               ((-0.86, 1.005, 0.020), mats["aluminium"]), ((-0.86, 1.005, 0.013), black)):
            add(surface.patch(f"filler_{tag}", ellipse(y, z, r, r, 20), beside, mat,
                              0.001 + (0.034 - r) * 0.05, 1))

    # ---- The roof: the rescue hatch's outline, the banner across the top of the windscreen,
    # the wiper, and an aerial.
    hatch = rounded_rect(0.0, -0.20, 0.56, 0.40, 0.06)
    add(surface.rim("roof_hatch", hatch, above, 0.005, 0.0012, seal))
    # The banner: the top of the windscreen, down to a line across it.
    import glazing
    top = [p for p in rounded(glazing.SCREEN, 6) if p[1] < 0.30]
    top.sort(key=lambda p: p[0])
    if len(top) > 3:
        banner = top + [(top[-1][0] - 0.01, 0.30), (0.25, 0.285), (-0.25, 0.285), (top[0][0] + 0.01, 0.30)]
        add(surface.patch("screen_banner", banner, above, black, -0.004))
    for name, y0, y1 in (("stripe_bonnet", 1.02, 2.06), ("stripe_roof", -0.655, 0.168)):
        band = [(-0.045, y0), (0.045, y0), (0.045, y1), (-0.045, y1)]
        add(surface.patch(name, band, above, accent, 0.0009, 4))
    wiper = [(0.02, 0.985, 0.868), (0.03, 0.90, 0.905), (0.04, 0.72, 0.992), (0.05, 0.50, 1.085)]
    add(tube("wiper_arm", wiper, 0.006, black, 6))
    blade = [(0.045, 0.80, 0.955), (0.055, 0.42, 1.118)]
    add(tube("wiper_blade", blade, 0.008, seal, 6))
    add(tube("aerial", [(0.25, -0.55, 1.225), (0.25, -0.60, 1.33)], 0.004, black, 6))
    return parts
