"""A small rigged character with every fault the export tool fixes, built on purpose.

    blender --background --factory-startup --python tools/blender/tests/faulty_character.py -- OUT.blend

- a mask modifier hiding a strip of faces, by a vertex group;
- a vertex group that isn't a bone;
- a shape key, mixed in;
- vertices with six bone weights, two of them tiny;
- a leaf bone that deforms nothing, and one the role file names that deforms nothing either;
- a blended material, whose alpha is solid over most of it and see-through over a part.

Its faults are in Blender's terms, not MakeHuman's, so it needs no plugin.
"""

import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
for thing in list(bpy.data.objects):
    bpy.data.objects.remove(thing)

# A chain of bones up the Z axis, with three short leaves off the spine.
armature = bpy.data.armatures.new("rig")
rig = bpy.data.objects.new("rig", armature)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
bones = {}
for name, head, tail, parent in [
    ("Root", (0, 0, 0), (0, 0, 0.1), None),
    ("pelvis", (0, 0, 0.1), (0, 0, 0.5), "Root"),
    ("spine", (0, 0, 0.5), (0, 0, 1.0), "pelvis"),
    ("a", (0, 0, 1.0), (0.1, 0, 1.0), "spine"),
    ("b", (0, 0, 1.0), (-0.1, 0, 1.0), "spine"),
    ("c", (0, 0, 1.0), (0, 0.1, 1.0), "spine"),
    ("idle_tip", (0, 0, 1.0), (0, 0, 1.2), "spine"),
    ("named_tip", (0, 0, 1.0), (0, -0.1, 1.0), "spine"),
]:
    bone = armature.edit_bones.new(name)
    bone.head, bone.tail = head, tail
    if parent:
        bone.parent = bones[parent]
    bones[name] = bone
bpy.ops.object.mode_set(mode="OBJECT")

# A standing sheet, 1 m wide and tall, with UVs.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=1.0)
sheet = bpy.context.object
sheet.name = "sheet"
sheet.rotation_euler = (1.5707963, 0, 0)
sheet.location = (0, 0, 0.5)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
sheet.parent = rig
modifier = sheet.modifiers.new("Armature", "ARMATURE")
modifier.object = rig

groups = {name: sheet.vertex_groups.new(name=name) for name in ["pelvis", "spine", "a", "b", "c"]}
shown = sheet.vertex_groups.new(name="shown")       # what the mask keeps
extra = sheet.vertex_groups.new(name="helper-stuff")  # not a bone
for vert in sheet.data.vertices:
    z = vert.co.z
    groups["pelvis"].add([vert.index], 1.0 - z, "REPLACE")
    groups["spine"].add([vert.index], z, "REPLACE")
    # Six influences in all at the top: three more bones, two of them with tiny weights.
    if z > 0.7:
        groups["a"].add([vert.index], 0.3, "REPLACE")
        groups["b"].add([vert.index], 0.004, "REPLACE")
        groups["c"].add([vert.index], 0.003, "REPLACE")
        extra.add([vert.index], 1.0, "REPLACE")
    # The mask hides the strip of faces nearest x = 0.5.
    if vert.co.x < 0.4:
        shown.add([vert.index], 1.0, "REPLACE")
mask = sheet.modifiers.new("Hide strip", "MASK")
mask.vertex_group = "shown"

sheet.shape_key_add(name="Basis")
bulge = sheet.shape_key_add(name="bulge")
for point in bulge.data:
    point.co.y += 0.1
bulge.value = 0.5

# A blended material: solid on the left of its UVs, see-through on the right.
material = bpy.data.materials.new("cloth")
material.surface_render_method = "BLENDED"
tree = material.node_tree
principled = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
principled.inputs["Base Color"].default_value = (0.6, 0.3, 0.2, 1.0)
coords = tree.nodes.new("ShaderNodeTexCoord")
split = tree.nodes.new("ShaderNodeSeparateXYZ")
less = tree.nodes.new("ShaderNodeMath")
less.operation = "LESS_THAN"
less.inputs[1].default_value = 0.7
tree.links.new(coords.outputs["UV"], split.inputs["Vector"])
tree.links.new(split.outputs["X"], less.inputs[0])
tree.links.new(less.outputs["Value"], principled.inputs["Alpha"])
sheet.data.materials.append(material)

bpy.ops.wm.save_as_mainfile(filepath=out)
