#!/usr/bin/env python3
"""Publishes baked scenes as a GitHub release, for the site to serve.

Scenes take minutes to bake and hundreds of megabytes to store, so they are baked on a PC and
kept as release assets rather than baked by CI or committed. The site serves the release that
scenes/release.txt names: publishing a new one is a commit that changes that line.

    python scripts/publish-scenes.py TAG [--bake]

--bake bakes every scene first, as the site uses them. Otherwise the files already in assets/
are published. Needs the GitHub CLI, signed in with permission to create releases.
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Each scene the site offers, and the command that bakes it.
SCENES = {
    "web.solstice": ["cargo", "run", "--release", "-p", "bake", "--", "--quality", "web",
                     "--output", "assets/web.solstice"],
    "cove.solstice": ["cargo", "run", "--release", "-p", "bake", "--", "--scene", "scenes/cove.txt",
                      "--texture-size", "1024", "--output", "assets/cove.solstice"],
    "forest.solstice": ["cargo", "run", "--release", "-p", "bake", "--", "--forest",
                        "--output", "assets/forest.solstice"],
}


def run(command):
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def output(command):
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    tag = args[0]
    if "--bake" in sys.argv:
        run(["python3" if os.name != "nt" else "python", "scripts/fetch-scans.py", "scenes/cove.txt", "2k"])
        for command in SCENES.values():
            run(command)

    paths = [os.path.join(ROOT, "assets", name) for name in SCENES]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("not baked yet: " + ", ".join(missing) + " (run with --bake)")

    commit = output(["git", "rev-parse", "--short", "HEAD"])
    # Only changes to what bakes the scenes matter, not to this script or the release it names.
    baking = ["git", "status", "--porcelain", "--", "apps", "crates", "scenes", "Cargo.lock",
              ":!scenes/release.txt"]
    dirty = " (with uncommitted changes)" if output(baking) else ""
    lines = [f"Scenes baked from {commit}{dirty}.", "", "| Scene | Size | SHA-256 |", "|---|---|---|"]
    for path in paths:
        lines.append(f"| {os.path.basename(path)} | {os.path.getsize(path) / 1e6:.1f} MB | `{sha256(path)}` |")
    notes = "\n".join(lines)
    print(notes)

    run(["gh", "release", "create", tag, "--title", f"Scenes {tag}", "--notes", notes, *paths])
    with open(os.path.join(ROOT, "scenes", "release.txt"), "w", encoding="utf-8") as f:
        f.write(tag + "\n")
    print(f"published {tag}. Commit scenes/release.txt for the site to use it.")


if __name__ == "__main__":
    main()
