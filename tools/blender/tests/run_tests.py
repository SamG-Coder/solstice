"""Tests of the export tool, run with Blender headless. Not in CI, which has no Blender.

    python tools/blender/tests/run_tests.py [--blender PATH] [--skip-keeper]

1. A character with every fault the tool fixes (faulty_character.py) comes out fixed, and passes
   the checks: the masked faces gone, the shape key applied, idle bones removed except one the
   role file names, at most four weights, cut out where it is see-through.
2. The same character exported as it is fails the checks.
3. The keeper (characters/keeper.py), with MakeHuman's plugin: two exports from scratch are the
   same to the byte, and pass the checks; exported as it is, it fails them.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
from make_character import blender_path  # noqa: E402

failures = []


def expect(condition, message):
    print(("  ok    " if condition else "  FAIL  ") + message, flush=True)
    if not condition:
        failures.append(message)


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def read_glb(path):
    raw = open(path, "rb").read()
    length = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20 + length])
    binary = raw[20 + length + 8:]
    return doc, binary


def faulty(blender, work):
    blend = os.path.join(work, "faulty.blend")
    code, log = run([blender, "--background", "--factory-startup", "--python-exit-code", "1",
                     "--python", os.path.join(HERE, "faulty_character.py"), "--", blend])
    expect(code == 0, "the faulty character is made")
    if code:
        print(log)
        return
    roles = os.path.join(HERE, "test.roles")
    export = [blender, "--background", "--factory-startup", "--python-exit-code", "1", blend,
              "--python", os.path.join(TOOLS, "solstice_export.py"), "--", None,
              "--roles", roles, "--texture-size", "64"]

    fixed = os.path.join(work, "fixed.glb")
    export[export.index(None)] = fixed
    code, log = run(export)
    expect(code == 0, "exported with its faults fixed, it passes the checks")
    if code:
        print(log)
        return
    expect("removed bones that deform nothing: ['b', 'c', 'idle_tip']" in log,
           "the idle leaf bones go, and those whose only weights were tiny")
    doc, _ = read_glb(fixed)
    joints = [doc["nodes"][j]["name"] for j in doc["skins"][0]["joints"]]
    expect(sorted(joints) == ["Root", "a", "named_tip", "pelvis", "spine"],
           f"the bones a role names stay, even idle: {joints}")
    primitive = doc["meshes"][0]["primitives"][0]
    positions = doc["accessors"][primitive["attributes"]["POSITION"]]
    # A 9 by 9 grid, less the column of nine the mask hid.
    expect(positions["count"] == 72, f"the masked faces are gone: {positions['count']} vertices")
    expect("targets" not in primitive, "no morph targets are left")
    # The shape key moved the sheet 0.05 along Blender's +Y, which is glTF's -Z.
    expect(abs(positions["min"][2] + 0.05) < 1e-4 and abs(positions["max"][2] + 0.05) < 1e-4,
           f"the shape key is applied as it was mixed: z from {positions['min'][2]} to {positions['max'][2]}")
    expect("JOINTS_1" not in primitive["attributes"], "at most four weights a vertex")
    material = doc["materials"][primitive["material"]]
    expect(material.get("alphaMode") == "MASK", "the see-through material is cut out, not blended")

    as_is = os.path.join(work, "as_is.glb")
    export[export.index(fixed)] = as_is
    export.append("--as-is")
    code, log = run(export)
    expect(code != 0 and "morph targets" in log and "is blended" in log,
           "exported as it is, it fails the checks")


def keeper(blender, work):
    maker = os.path.join(TOOLS, "make_character.py")
    character = os.path.join(REPO, "characters", "keeper.py")
    outputs = []
    for name in ("first", "second"):
        out = os.path.join(work, name, "keeper.glb")
        code, log = run([sys.executable, maker, character, out, "--blender", blender])
        expect(code == 0, f"the keeper's {name} export passes the checks")
        if code:
            print(log[-3000:])
            return
        outputs.append(out)
    same = open(outputs[0], "rb").read() == open(outputs[1], "rb").read()
    expect(same, "two exports of the keeper are the same to the byte")
    out = os.path.join(work, "as_is", "keeper.glb")
    code, log = run([sys.executable, maker, character, out, "--blender", blender, "--as-is"])
    expect(code != 0 and "not ready for the engine" in log,
           "the keeper exported as it is fails the checks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender")
    parser.add_argument("--skip-keeper", action="store_true",
                        help="skip the tests that need MakeHuman's plugin")
    args = parser.parse_args()
    blender = blender_path(args.blender)
    # Built once, so the checks the tool runs don't each wait on cargo.
    subprocess.run(["cargo", "build", "--release", "--quiet", "-p", "bake"], cwd=REPO, check=True)
    with tempfile.TemporaryDirectory(prefix="solstice-export-") as work:
        print("the faulty character:")
        faulty(blender, work)
        if not args.skip_keeper:
            print("the keeper:")
            keeper(blender, work)
    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
