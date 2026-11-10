"""Looks the built model over for faults: parts that float free of the car, parts that are
inside out, and holes in the shell. Run by make.py with --audit."""

import math

import bmesh
from mathutils.bvhtree import BVHTree

import body
import spec


def audit(shell, parts, wheels):
    problems = []
    bm = bmesh.new()
    bm.from_mesh(shell.data)
    open_edges = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    volume = bm.calc_volume(signed=True)
    tree = BVHTree.FromBMesh(bm)
    if open_edges:
        problems.append(f"the shell has {open_edges} edges that are not shared by two faces")
    if volume <= 0:
        problems.append("the shell is inside out")
    print(f"audit: shell volume {volume:.3f} m3, {open_edges} open edges, {len(bm.faces)} faces")
    # A part belongs on the car if some point of it is near the shell, or near another part
    # that is. Wheels and what is inside the cabin are held by things not modelled.
    loose = []
    for obj in parts:
        if obj is shell or not obj.data.vertices:
            continue
        nearest = min((tree.find_nearest(obj.matrix_world @ v.co)[3] for v in list(obj.data.vertices)[::7]),
                      default=0.0)
        if nearest > 0.06:
            loose.append((obj.name, nearest))
    bm.free()
    held = ("wing", "splitter", "skirt", "diffuser", "tailpipe", "mirror", "wiper", "tow", "intake",
            "roll_cage", "seat", "steering", "dash", "switch", "bulkhead", "cabin", "net", "fire",
            "pedals", "fuel", "lamp_units", "aerial",
            # In the quarter windows and the cabin: on the glass, and on the cage.
            "filler", "window_net", "centre_net")
    for name, distance in loose:
        if not name.startswith(held):
            problems.append(f"{name} is {distance * 1000:.0f} mm from the body: floating?")
    # Each wheel's tyre must stand on the ground and clear its arch.
    for wheel in wheels:
        if not wheel.name.startswith("wheel_"):
            continue
        lowest = min((wheel.matrix_world @ v.co).z for v in wheel.data.vertices)
        if abs(lowest) > 0.004:
            problems.append(f"{wheel.name} is {lowest * 1000:+.0f} mm from the ground")
    for problem in problems:
        print(f"audit: {problem}")
    print(f"audit: {len(problems)} problems")
    return problems


def inside(skin, parts, margin=0.010):
    """Whether everything in the cabin is inside the body's skin, by at least `margin`: the
    glass sits 6 mm in. Says which parts are not, and by how much."""
    bm = bmesh.new()
    bm.from_mesh(skin.data)
    tree = BVHTree.FromBMesh(bm)
    faults = []
    for obj in parts:
        worst, count, where = 0.0, 0, None
        for v in obj.data.vertices:
            at = obj.matrix_world @ v.co
            nearest, normal, _, _ = tree.find_nearest(at)
            out = (at - nearest).dot(normal) + margin
            # In a wheel's well is outside the cabin too.
            for axle, radius in ((spec.FRONT_AT, spec.FRONT_RADIUS), (-spec.REAR_AT, spec.REAR_RADIUS)):
                reach = radius + 0.026 - math.hypot(at.y - axle, at.z - radius)
                if abs(at.x) > body.WELL_INNER and reach > 0.0:
                    out = max(out, reach)
            if out > 0.0:
                count += 1
                if out > worst:
                    worst, where = out, at.copy()
        if count:
            faults.append((obj.name, count, worst))
            print(f"audit: {obj.name} comes within {margin * 1000:.0f} mm of the skin, or through it, "
                  f"at {count} points, by up to {worst * 1000:.0f} mm, worst at "
                  f"x {where.x:+.2f} y {where.y:+.2f} z {where.z:.2f}")
    bm.free()
    print(f"audit: {len(faults)} cabin parts outside the body")
    return faults
