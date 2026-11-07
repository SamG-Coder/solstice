#!/usr/bin/env python3
"""Publishes baked scenes as a GitHub release, uploading only the scenes that changed.

Scenes take minutes to bake and hundreds of megabytes to store, so they are baked on a PC and
kept as release assets rather than baked by CI or committed. scenes/releases.txt records, for
each scene the site serves, the release that holds it and its SHA-256. Publishing compares
each scene in assets/ with that record, puts only the changed ones in a new release, and
updates their lines; unchanged scenes stay in whichever release already holds them. Commit
scenes/releases.txt for the site to serve the result.

    python scripts/publish-scenes.py [TAG] [--check] [--bake [SCENE...]]

TAG defaults to scenes-YYYY-MM-DD-HHMM. --check only reports what would be published. --bake
bakes scenes first: the ones named, or all. Needs the GitHub CLI, signed in with permission to
create releases.
"""
import datetime
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "scenes", "releases.txt")
HEADER = """\
# Which release holds each scene the site serves, and the scene's SHA-256. Written by
# scripts/publish-scenes.py, which uploads a scene again only when its content changes.
# The Pages workflow downloads each scene from its release and checks it against this.
"""
# Each scene the site offers, and the command that bakes it.
SCENES = {
    "web": ["cargo", "run", "--release", "-p", "bake", "--", "--quality", "web",
            "--output", "assets/web.solstice"],
    "cove": ["cargo", "run", "--release", "-p", "bake", "--", "--scene", "scenes/cove.txt",
             "--texture-size", "1024", "--output", "assets/cove.solstice"],
    "forest": ["cargo", "run", "--release", "-p", "bake", "--", "--forest",
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


def read_manifest():
    """Scene file name -> (release tag, SHA-256)."""
    entries = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                words = line.split("#")[0].split()
                if len(words) == 3:
                    entries[words[0]] = (words[1], words[2])
    return entries


def write_manifest(entries):
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER)
        for name in sorted(entries):
            tag, digest = entries[name]
            f.write(f"{name} {tag} {digest}\n")


def main():
    args = sys.argv[1:]
    to_bake = []
    if "--bake" in args:
        at = args.index("--bake")
        to_bake = [a for a in args[at + 1:] if not a.startswith("--")] or list(SCENES)
        args = args[:at] + [a for a in args[at + 1:] if a.startswith("--")]
    check_only = "--check" in args
    positional = [a for a in args if not a.startswith("--")]
    if len(positional) > 1 or any(a not in ("--check",) for a in args if a.startswith("--")):
        sys.exit(__doc__)
    tag = positional[0] if positional else datetime.datetime.now().strftime("scenes-%Y-%m-%d-%H%M")

    unknown = [s for s in to_bake if s not in SCENES]
    if unknown:
        sys.exit(f"unknown scene: {', '.join(unknown)} (scenes: {', '.join(SCENES)})")
    if "cove" in to_bake:
        run([sys.executable, "scripts/fetch-scans.py", "scenes/cove.txt", "2k"])
    for name in to_bake:
        run(SCENES[name])

    manifest = read_manifest()
    changed = []
    for name in SCENES:
        file = f"{name}.solstice"
        path = os.path.join(ROOT, "assets", file)
        if not os.path.exists(path):
            if file not in manifest:
                sys.exit(f"{file} is neither baked nor published yet (run with --bake {name})")
            print(f"{file}: not baked here, keeping {manifest[file][0]}")
            continue
        digest = sha256(path)
        if file in manifest and manifest[file][1] == digest:
            print(f"{file}: unchanged, in {manifest[file][0]}")
        else:
            changed.append((file, path, digest))
            print(f"{file}: changed")
    if not changed:
        print("nothing to publish")
        return
    if check_only:
        print(f"would publish {len(changed)} scene(s) as {tag}")
        return

    commit = output(["git", "rev-parse", "--short", "HEAD"])
    # Only changes to what bakes the scenes matter, not to this script or its record.
    baking = ["git", "status", "--porcelain", "--", "apps", "crates", "scenes", "Cargo.lock",
              ":!scenes/releases.txt"]
    dirty = " (with uncommitted changes)" if output(baking) else ""
    lines = [f"Scenes baked from {commit}{dirty}. Scenes not listed are unchanged, in earlier "
             "releases: see scenes/releases.txt.", "", "| Scene | Size | SHA-256 |", "|---|---|---|"]
    for file, path, digest in changed:
        lines.append(f"| {file} | {os.path.getsize(path) / 1e6:.1f} MB | `{digest}` |")
    notes = "\n".join(lines)
    print(notes)

    run(["gh", "release", "create", tag, "--title", f"Scenes {tag}", "--notes", notes,
         *[path for _, path, _ in changed]])
    for file, _, digest in changed:
        manifest[file] = (tag, digest)
    write_manifest(manifest)
    print(f"published {len(changed)} scene(s) as {tag}. "
          "Commit scenes/releases.txt for the site to use them.")


if __name__ == "__main__":
    main()
