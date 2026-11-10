"""What every part of the 911 GT3 R's model is built with: curves, meshes, materials, cuts."""

import math

import bmesh
import bpy
from mathutils import Matrix, Vector


def curve(points):
    """A smooth curve through (t, value) points, as a function of t: cubic between points,
    with slopes that never overshoot (Fritsch and Carlson)."""
    points = sorted(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    d = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    m = [d[0]] + [0.0] * (n - 2) + [d[-1]]
    for i in range(1, n - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 2 * d[i - 1] * d[i] / (d[i - 1] + d[i])

    def at(x):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        lo, hi = 0, n - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h
        return (
            (2 * t**3 - 3 * t**2 + 1) * ys[i]
            + (t**3 - 2 * t**2 + t) * h * m[i]
            + (-2 * t**3 + 3 * t**2) * ys[i + 1]
            + (t**3 - t**2) * h * m[i + 1]
        )

    return at


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def rounded(points, per=10, closed=True):
    """A smooth outline through these points (Catmull-Rom), `per` points a span."""
    n = len(points)
    dims = len(points[0])
    out = []
    spans = n if closed else n - 1
    for i in range(spans):
        if closed:
            p0, p1, p2, p3 = (points[(i + k - 1) % n] for k in range(4))
        else:
            p0 = points[max(i - 1, 0)]
            p1, p2 = points[i], points[i + 1]
            p3 = points[min(i + 2, n - 1)]
        for step in range(per):
            t = step / per
            out.append(tuple(
                0.5 * ((2 * p1[a]) + (-p0[a] + p2[a]) * t
                       + (2 * p0[a] - 5 * p1[a] + 4 * p2[a] - p3[a]) * t * t
                       + (-p0[a] + 3 * p1[a] - 3 * p2[a] + p3[a]) * t * t * t)
                for a in range(dims)))
    if not closed:
        out.append(tuple(points[-1]))
    return out


def ellipse(cx, cy, rx, ry, count=40):
    return [(cx + rx * math.cos(math.tau * k / count), cy + ry * math.sin(math.tau * k / count))
            for k in range(count)]


def line(start, end, steps=40):
    return [tuple(start[i] + (end[i] - start[i]) * k / steps for i in range(len(start)))
            for k in range(steps + 1)]


# ------------------------------------------------------------------------------- materials

MATERIALS = {}


def material(name, colour, roughness, metallic=0.0, emission=None, alpha=1.0, coat=0.0,
             transmission=0.0):
    """A physically based material, made once and shared by name."""
    if name in MATERIALS:
        return MATERIALS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    shader = m.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*colour, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if emission:
        shader.inputs["Emission Color"].default_value = (*emission[0], 1.0)
        shader.inputs["Emission Strength"].default_value = emission[1]
    if coat:
        shader.inputs["Coat Weight"].default_value = coat
        shader.inputs["Coat Roughness"].default_value = 0.03
    if transmission:
        shader.inputs["Transmission Weight"].default_value = transmission
    if alpha < 1.0:
        shader.inputs["Alpha"].default_value = alpha
        m.surface_render_method = "BLENDED"
    MATERIALS[name] = m
    return m


# ---------------------------------------------------------------------------------- meshes

def new_object(name, build, materials=(), collection=None):
    """An object from a bmesh that `build` fills in."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    build(bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    for m in materials:
        mesh.materials.append(m)
    obj = bpy.data.objects.new(name, mesh)
    (collection or bpy.context.collection).objects.link(obj)
    return obj


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def smooth(obj, angle=40.0):
    select_only(obj)
    bpy.ops.object.shade_auto_smooth(angle=math.radians(angle))
    return obj


def apply_modifiers(obj):
    select_only(obj)
    for modifier in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=modifier.name)


def join(objects, name):
    """One object of several, named. The first one's origin is kept."""
    objects = [o for o in objects if o is not None]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    result = bpy.context.object
    result.name = name
    return result


def boolean(obj, cutter, operation="DIFFERENCE", keep_cutter=False, solver="EXACT"):
    """Cuts `cutter` out of `obj` (or joins, or keeps what they share). The cutter's faces
    bring their own material with them."""
    for m in cutter.data.materials:
        if m.name not in [x.name for x in obj.data.materials if x]:
            obj.data.materials.append(m)
    mod = obj.modifiers.new("cut", "BOOLEAN")
    mod.operation = operation
    mod.object = cutter
    mod.solver = solver
    if hasattr(mod, "material_mode"):
        mod.material_mode = "TRANSFER"
    apply_modifiers(obj)
    if not keep_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return obj


def bevel(obj, width, segments=3, angle=30.0):
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(angle)
    apply_modifiers(obj)
    return obj


def box(name, low, high, mat, round_by=0.0):
    """A box between two corners."""
    def build(bm):
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            for axis in range(3):
                v.co[axis] = low[axis] + (v.co[axis] + 0.5) * (high[axis] - low[axis])
    obj = new_object(name, build, [mat])
    if round_by > 0.0:
        bevel(obj, round_by)
    return smooth(obj)


def prism(name, outline, axis, low, high, mat, round_by=0.0):
    """An outline, drawn in the plane across `axis` ("x", "y" or "z"), pulled along it from
    `low` to `high`. In the plane, the outline's two measures are the other two axes in order:
    (y, z) across x, (x, z) across y, (x, y) across z."""
    def place(u, v, w):
        return {"x": (w, u, v), "y": (u, w, v), "z": (u, v, w)}[axis]

    def build(bm):
        a = [bm.verts.new(place(u, v, low)) for u, v in outline]
        b = [bm.verts.new(place(u, v, high)) for u, v in outline]
        bm.faces.new(a)
        bm.faces.new(list(reversed(b)))
        for i in range(len(outline)):
            j = (i + 1) % len(outline)
            bm.faces.new((a[i], a[j], b[j], b[i]))
    obj = new_object(name, build, [mat])
    if round_by > 0.0:
        bevel(obj, round_by)
    return smooth(obj)


def lathe(bm, profile, segments=64, axis="x", centre=(0.0, 0.0, 0.0), arc=math.tau, start=0.0,
          capped=False):
    """Spins a profile of (along, radius) points about an axis through `centre`: all the way
    round, or through `arc` radians from `start`, with its ends closed if `capped`."""
    rings = []
    count = segments if arc >= math.tau - 1e-6 else segments + 1
    for k in range(count):
        angle = start + arc * k / segments
        c, s = math.cos(angle), math.sin(angle)
        ring = []
        for along, r in profile:
            if axis == "x":
                p = (along, r * c, r * s)
            elif axis == "y":
                p = (r * s, along, r * c)
            else:
                p = (r * c, r * s, along)
            ring.append(bm.verts.new((p[0] + centre[0], p[1] + centre[1], p[2] + centre[2])))
        rings.append(ring)
    spans = segments if arc >= math.tau - 1e-6 else segments
    for k in range(spans):
        a, b = rings[k], rings[(k + 1) % count]
        for i in range(len(profile) - 1):
            bm.faces.new((a[i], a[i + 1], b[i + 1], b[i]))
    if capped and count > segments:
        bm.faces.new(rings[0])
        bm.faces.new(list(reversed(rings[-1])))


def tube(name, path, radius, mat, sides=12, closed=False):
    """A round tube along a path of points: a roll cage's bar, a pipe, a stay."""
    path = [Vector(p) for p in path]

    def build(bm):
        rings = []
        n = len(path)
        last_normal = None
        for i, at in enumerate(path):
            before = path[i - 1] if (i > 0 or closed) else at
            after = path[(i + 1) % n] if (i < n - 1 or closed) else at
            along = (after - before)
            if along.length < 1e-9:
                along = Vector((0, 0, 1))
            along.normalize()
            if last_normal is None:
                helper = Vector((0, 0, 1)) if abs(along.z) < 0.9 else Vector((1, 0, 0))
                normal = along.cross(helper).normalized()
            else:
                normal = (last_normal - along * last_normal.dot(along))
                if normal.length < 1e-6:
                    normal = along.orthogonal()
                normal.normalize()
            last_normal = normal
            binormal = along.cross(normal)
            rings.append([bm.verts.new(at + (normal * math.cos(math.tau * k / sides)
                                             + binormal * math.sin(math.tau * k / sides)) * radius)
                          for k in range(sides)])
        spans = n if closed else n - 1
        for i in range(spans):
            a, b = rings[i], rings[(i + 1) % n]
            for k in range(sides):
                bm.faces.new((a[k], a[(k + 1) % sides], b[(k + 1) % sides], b[k]))
        if not closed:
            bm.faces.new(list(reversed(rings[0])))
            bm.faces.new(rings[-1])
    return smooth(new_object(name, build, [mat]), 60.0)


def plate(name, outline, axis, at, thickness, mat, round_by=0.0):
    """A flat plate of an outline, `thickness` thick, centred at `at` along `axis`."""
    return prism(name, outline, axis, at - thickness / 2, at + thickness / 2, mat, round_by)


def transform(obj, matrix):
    obj.data.transform(matrix)
    obj.data.update()
    return obj


def mirrored(obj, name=None):
    """A copy of an object, mirrored across the car's middle."""
    copy = obj.copy()
    copy.data = obj.data.copy()
    bpy.context.collection.objects.link(copy)
    copy.data.transform(Matrix.Scale(-1, 4, Vector((1, 0, 0))))
    copy.data.flip_normals()
    copy.name = name or obj.name
    return copy


def even_fill(outline):
    """Fills an outline of (u, v) points with triangles of an even size: its own points round
    the edge, a honeycomb of points inside, joined by a constrained Delaunay triangulation.
    Returns the points and the triangles, as indices into them."""
    from mathutils.geometry import delaunay_2d_cdt
    n = len(outline)
    us, vs = [p[0] for p in outline], [p[1] for p in outline]
    area = abs(sum(outline[i][0] * outline[(i + 1) % n][1] - outline[(i + 1) % n][0] * outline[i][1]
                   for i in range(n))) / 2
    spacing = min(max(math.sqrt(area) / 16, 0.006), 0.035)

    def inside(u, v):
        within = False
        for i in range(n):
            (u0, v0), (u1, v1) = outline[i], outline[(i + 1) % n]
            if (v0 > v) != (v1 > v) and u < u0 + (u1 - u0) * (v - v0) / (v1 - v0):
                within = not within
        return within

    def clear(u, v):
        # Not so near the edge as to make a sliver against it.
        least = spacing * 0.45
        for i in range(n):
            (u0, v0), (u1, v1) = outline[i], outline[(i + 1) % n]
            du, dv = u1 - u0, v1 - v0
            length = du * du + dv * dv or 1e-12
            t = min(max(((u - u0) * du + (v - v0) * dv) / length, 0.0), 1.0)
            if math.hypot(u - (u0 + du * t), v - (v0 + dv * t)) < least:
                return False
        return True

    points = [Vector((u, v)) for u, v in outline]
    row, v = 0, min(vs) + spacing * 0.5
    while v < max(vs):
        u = min(us) + (spacing * 0.5 if row % 2 else 0.0)
        while u < max(us):
            if inside(u, v) and clear(u, v):
                points.append(Vector((u, v)))
            u += spacing
        v += spacing * 0.866
        row += 1
    edges = [(i, (i + 1) % n) for i in range(n)]
    result = delaunay_2d_cdt(points, edges, [list(range(n))], 1, 1e-7)
    return [(p.x, p.y) for p in result[0]], [tuple(f) for f in result[2] if len(f) == 3]


# ---------------------------------------------------------------- fitting things to a surface

class Surface:
    """A built object, for laying other things on it where rays meet it."""

    def __init__(self, obj):
        self.obj = obj
        bpy.context.view_layer.update()

    def cast(self, origin, direction):
        hit, at, normal, _ = self.obj.ray_cast(Vector(origin), Vector(direction).normalized())
        return (at, normal) if hit else None

    def top(self, x, y):
        found = self.cast((x, y, 3.0), (0, 0, -1))
        return found[0].z if found else None

    def side(self, y, z, side=1):
        found = self.cast((side * 2.0, y, z), (-side, 0, 0))
        return found[0].x if found else None

    def ribbon(self, name, rays, width, lift, mat):
        """A strip lying on the surface along the points where `rays` meet it: a shut line, a
        seal, a stripe."""
        found = [f for f in (self.cast(o, d) for o, d in rays) if f]
        if len(found) < 2:
            return None

        def build(bm):
            edges = []
            for k, (at, normal) in enumerate(found):
                before = found[max(k - 1, 0)][0]
                after = found[min(k + 1, len(found) - 1)][0]
                across = (after - before).cross(normal)
                if across.length < 1e-6:
                    continue
                across.normalize()
                middle = at + normal * lift
                edges.append((bm.verts.new(middle - across * width / 2),
                              bm.verts.new(middle + across * width / 2)))
            for (a, b), (c, d) in zip(edges, edges[1:]):
                bm.faces.new((a, b, d, c))
        return smooth(new_object(name, build, [mat]), 60.0)

    def patch(self, name, outline, ray, mat, lift=0.0025, cuts=3):
        """The piece of the surface inside an outline. The outline is drawn flat, in whatever
        two measures `ray` takes, filled, and every point of it laid on the surface where its
        ray meets it."""
        def build(bm):
            if cuts >= 2:
                # An even mesh of well-shaped triangles. Cutting the outline into ears leaves
                # slivers round its edge, and slivers shade badly: a pane of glass showed
                # them as a ragged dark fringe.
                points, triangles = even_fill(outline)
                verts = [bm.verts.new((u, v, 0.0)) for u, v in points]
                for a, b, c in triangles:
                    bm.faces.new((verts[a], verts[b], verts[c]))
            else:
                face = bm.faces.new([bm.verts.new((u, v, 0.0)) for u, v in outline])
                bmesh.ops.triangulate(bm, faces=[face])
                for _ in range(cuts):
                    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
            missed = []
            for vert in bm.verts:
                found = self.cast(*ray(vert.co.x, vert.co.y))
                if found:
                    vert.co = found[0] + found[1] * lift
                    outward.append(found[1].copy())
                else:
                    missed.append(vert)
            if missed:
                bmesh.ops.delete(bm, geom=missed, context="VERTS")
        outward = []
        obj = new_object(name, build, [mat])
        # Facing out of the surface, whichever way the filling happened to wind.
        if obj.data.polygons and outward:
            faces = sum((p.normal * p.area for p in obj.data.polygons), Vector())
            if faces.dot(sum(outward, Vector())) < 0.0:
                obj.data.flip_normals()
        return smooth(obj, 80.0)

    def recess(self, target, outline, ray, depth, liner, cuts=2):
        """Sinks the piece of the surface inside an outline into `target` by `depth`: an
        opening, a duct, a lamp's bowl. Its walls and floor are lined with `liner`."""
        cutter = self.patch("recess", outline, ray, liner, lift=0.006, cuts=cuts)
        if not cutter.data.polygons:
            bpy.data.objects.remove(cutter, do_unlink=True)
            return
        solid = cutter.modifiers.new("solid", "SOLIDIFY")
        solid.thickness = depth + 0.006
        solid.offset = -1.0
        apply_modifiers(cutter)
        boolean(target, cutter)

    def fin(self, name, rays, reach, thickness, mat, out=None):
        """A thin plate standing out of the surface along the points where `rays` meet it:
        a dive plane, a strake, a lip. `reach(t)` is how far it stands out at `t` of the way
        along; `out`, if given, is the way it stands, instead of straight out of the surface."""
        found = [f for f in (self.cast(o, d) for o, d in rays) if f]
        if len(found) < 2:
            return None

        def build(bm):
            rows = []
            n = len(found)
            for k, (at, normal) in enumerate(found):
                way = Vector(out).normalized() if out else normal
                before = found[max(k - 1, 0)][0]
                after = found[min(k + 1, n - 1)][0]
                along = (after - before).normalized()
                up = way.cross(along).normalized() * (thickness / 2)
                root, tip = at - way * 0.01, at + way * reach(k / (n - 1))
                rows.append([bm.verts.new(p) for p in (root + up, tip + up * 0.5, tip - up * 0.5, root - up)])
            for a, b in zip(rows, rows[1:]):
                for i in range(4):
                    bm.faces.new((a[i], a[(i + 1) % 4], b[(i + 1) % 4], b[i]))
            bm.faces.new(list(reversed(rows[0])))
            bm.faces.new(rows[-1])
        return smooth(new_object(name, build, [mat]), 50.0)

    def rim(self, name, outline, ray, width, lift, mat):
        """A band round an outline: a window's seal, a lamp's surround."""
        return self.ribbon(name, [ray(u, v) for u, v in outline + outline[:2]], width, lift, mat)
