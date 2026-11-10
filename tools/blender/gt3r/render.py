"""Inspection views of the model: the seven the brief asks for, under studio light."""

import math
import os

import bpy
from mathutils import Vector

from common import material

# name: (where the camera is, what it looks at, orthographic width or None, lens)
VIEWS = {
    "front": ((0.0, 9.0, 0.62), (0.0, 0.0, 0.62), 3.5, None),
    "rear": ((0.0, -9.0, 0.66), (0.0, 0.0, 0.66), 3.5, None),
    "left": ((-9.0, -0.08, 0.66), (0.0, -0.08, 0.66), 5.3, None),
    "right": ((9.0, -0.08, 0.66), (0.0, -0.08, 0.66), 5.3, None),
    "top": ((0.0, -0.08, 9.0), (0.0, -0.08, 0.0), 5.3, None),
    "front_quarter": ((3.9, 5.6, 1.55), (0.0, 0.25, 0.55), None, 62),
    "rear_quarter": ((-3.9, -5.6, 1.75), (0.0, -0.25, 0.6), None, 62),
    # Close views, for looking at one part.
    "wheel_front": ((3.3, 2.6, 0.62), (0.85, 1.25, 0.36), None, 80),
    "wheel_rear": ((3.3, -2.6, 0.66), (0.86, -1.25, 0.38), None, 80),
    "nose": ((1.6, 4.6, 0.95), (0.25, 1.9, 0.45), None, 70),
    "tail": ((-1.7, -4.9, 1.25), (-0.2, -2.0, 0.65), None, 70),
    "cabin": ((-2.4, 1.5, 1.45), (-0.2, 0.0, 0.92), None, 75),
    "cabin_top": ((-0.6, 1.9, 2.3), (-0.25, 0.15, 0.8), None, 60),
    "side_detail": ((3.4, -0.6, 0.9), (0.9, -0.5, 0.55), None, 80),
    "wing": ((-2.2, -4.6, 1.9), (0.0, -2.1, 1.1), None, 75),
    "front_low": ((0.9, 5.2, 0.42), (0.0, 1.8, 0.42), None, 60),
    # The windscreen pillar, from outside and above, and the door's top from the side.
    "pillar": ((-1.9, 1.9, 1.75), (-0.55, 0.42, 1.0), None, 95),
    "pillar_right": ((2.2, 1.6, 1.5), (0.6, 0.35, 1.02), None, 95),
    "door_top": ((-3.0, 0.1, 1.25), (-0.7, 0.0, 0.98), None, 90),
    # From the driver's seat, looking at the far windscreen pillar from inside.
    "inside": ((-0.25, -0.30, 1.0), (0.66, 0.45, 1.0), None, 28),
    # The windscreen pillar square on from the side, as the photograph M22_3548 has it.
    "pillar_side": ((9.0, 0.42, 1.0), (0.0, 0.42, 1.0), 0.92, None),
}


def studio(outdoor=False):
    """A plain floor and even light: a neutral studio, or a clear sky."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
        scene.cycles.device = "GPU"
    except Exception as error:
        print(f"render: on the CPU ({error})")
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.0
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    if outdoor:
        sky = world.node_tree.nodes.new("ShaderNodeTexSky")
        try:
            sky.sun_elevation = math.radians(40)
            sky.sun_rotation = math.radians(150)
        except AttributeError:
            pass
        world.node_tree.links.new(sky.outputs[0], background.inputs["Color"])
        background.inputs["Strength"].default_value = 0.06
    else:
        background.inputs["Color"].default_value = (0.42, 0.43, 0.45, 1.0)
        background.inputs["Strength"].default_value = 0.55
    scene.world = world
    made = []
    bpy.ops.mesh.primitive_plane_add(size=400)
    floor = bpy.context.object
    floor.name = "studio_floor"
    floor.data.materials.append(material("studio_floor", (0.16, 0.165, 0.17), 0.55))
    made.append(floor)
    if not outdoor:
        for name, at, size, power in (("key", (2.5, 3.5, 6.0), 7.0, 1500.0),
                                      ("fill", (-6.0, -2.0, 4.0), 6.0, 700.0),
                                      ("back", (1.0, -6.5, 4.5), 6.0, 800.0)):
            data = bpy.data.lights.new(name, "AREA")
            data.size = size
            data.energy = power
            light = bpy.data.objects.new(name, data)
            bpy.context.collection.objects.link(light)
            light.location = at
            light.rotation_euler = (Vector((0, 0, 0.5)) - Vector(at)).to_track_quat("-Z", "Y").to_euler()
            made.append(light)
    return made


def probe(name, u, v, size=(800, 500)):
    """What a view sees at a point of its picture, `u` across and `v` down, each from 0 to 1:
    the part's name, its material, and where. For finding out what something in a render is."""
    eye, target, width, lens = VIEWS[name]
    direction = (Vector(target) - Vector(eye)).normalized()
    turn = direction.to_track_quat("-Z", "Y")
    aspect = size[1] / size[0]
    if width:
        origin = Vector(eye) + turn @ Vector(((u - 0.5) * width, (0.5 - v) * width * aspect, 0.0))
        ray = direction
    else:
        origin = Vector(eye)
        ray = (turn @ Vector(((u - 0.5) * 36.0 / lens, (0.5 - v) * 36.0 / lens * aspect, -1.0))).normalized()
    seen = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for _ in range(6):
        hit, at, normal, index, obj, _ = bpy.context.scene.ray_cast(depsgraph, origin, ray)
        if not hit:
            break
        slot = obj.data.polygons[index].material_index if obj.type == "MESH" else 0
        mat = obj.data.materials[slot].name if obj.type == "MESH" and obj.data.materials else "?"
        seen.append(f"{obj.name} ({mat}) at {at.x:+.2f} {at.y:+.2f} {at.z:.2f}")
        origin = at + ray * 0.002
    print(f"probe: {name} {u:.2f} {v:.2f}: " + "  then  ".join(seen))


def shoot(folder, names=None, size=(800, 500), outdoor=False, prefix="", samples=48):
    """Renders the named views into `folder`, one file each."""
    made = studio(outdoor)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.resolution_x, scene.render.resolution_y = size
    os.makedirs(folder, exist_ok=True)
    for name in names or VIEWS:
        eye, target, width, lens = VIEWS[name]
        data = bpy.data.cameras.new(name)
        if width:
            data.type = "ORTHO"
            data.ortho_scale = width
        else:
            data.lens = lens
        camera = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(camera)
        camera.location = eye
        up = "X" if name == "top" else "Y"
        camera.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", up).to_euler()
        if name == "top":
            # Nose to the right, as the side views have it.
            camera.rotation_euler = (0.0, 0.0, -math.pi / 2)
        scene.camera = camera
        scene.render.filepath = os.path.join(folder, f"{prefix}{name}.png")
        bpy.ops.render.render(write_still=True)
        made.append(camera)
    for obj in made:
        bpy.data.objects.remove(obj, do_unlink=True)
