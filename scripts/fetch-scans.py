#!/usr/bin/env python3
"""Downloads the Poly Haven models a scene file refers to, verifying each file.

Models are CC0, from https://polyhaven.com. They are not stored in this repository.

    python scripts/fetch-scans.py scenes/cove.txt [resolution]

Paths in the scene file must look like ../target/scan/NAME/NAME.gltf.
"""
import hashlib
import json
import os
import re
import sys
import urllib.request

AGENT = {"User-Agent": "solstice-fetch"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=AGENT)).read()


def fetch(entry, path):
    # By checksum, not size: the same model at another resolution can have the same size, as its
    # .gltf differs only in the "1k" or "2k" within its texture names.
    if os.path.exists(path) and os.path.getsize(path) == entry["size"]:
        with open(path, "rb") as f:
            if not entry.get("md5") or hashlib.md5(f.read()).hexdigest() == entry["md5"]:
                return
    data = get(entry["url"])
    if entry.get("md5") and hashlib.md5(data).hexdigest() != entry["md5"]:
        sys.exit(f"checksum mismatch: {entry['url']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def main():
    scene, resolution = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "1k"
    with open(scene, encoding="utf-8") as f:
        names = sorted(set(re.findall(r"^(?:model|part)\s+\S*/scan/([\w-]+)/", f.read(), re.M)))
    root = os.path.join(os.path.dirname(os.path.abspath(scene)), "..", "target", "scan")
    for name in names:
        entry = json.loads(get(f"https://api.polyhaven.com/files/{name}"))["gltf"][resolution]["gltf"]
        print(f"{name}...")
        # The scene refers to a fixed file name, whatever the resolution in the original.
        fetch(entry, os.path.join(root, name, f"{name}.gltf"))
        for relative, part in entry["include"].items():
            fetch(part, os.path.join(root, name, relative))


if __name__ == "__main__":
    main()
