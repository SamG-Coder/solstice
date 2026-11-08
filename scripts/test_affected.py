#!/usr/bin/env python3
"""Tests for scripts/affected.py's choice of crates, on a small made-up workspace.

    python scripts/test_affected.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from affected import affected  # noqa: E402

# asset <- builder <- render <- viewer; world <- script <- game <- viewer; game also needs asset.
FOLDERS = {
    "asset": "crates/asset/",
    "builder": "crates/builder/",
    "render": "crates/render/",
    "world": "crates/world/",
    "script": "crates/script/",
    "game": "crates/game/",
    "viewer": "apps/viewer/",
    # A crate nested in another's folder: the deepest folder owns the file.
    "inner": "crates/render/inner/",
}
DEPS = {
    "asset": set(),
    "builder": {"asset"},
    "render": {"asset", "builder"},
    "world": set(),
    "script": {"world"},
    "game": {"asset", "script", "world"},
    "viewer": {"asset", "game", "render"},
    "inner": set(),
}


def run(*files):
    return affected(list(files), FOLDERS, DEPS)


class Affected(unittest.TestCase):
    def test_a_crate_and_everything_that_depends_on_it(self):
        self.assertEqual(run("crates/render/src/lib.rs"), {"render", "viewer"})
        self.assertEqual(run("crates/world/src/clock.rs"), {"world", "script", "game", "viewer"})

    def test_dependencies_are_followed_however_indirectly(self):
        self.assertEqual(run("crates/asset/src/scene.rs"), {"asset", "builder", "render", "game", "viewer"})

    def test_a_leaf_affects_only_itself(self):
        self.assertEqual(run("apps/viewer/src/main.rs"), {"viewer"})

    def test_shaders_and_tests_belong_to_their_crate(self):
        self.assertEqual(run("crates/render/shaders/x.wgsl", "crates/render/tests/gpu.rs"), {"render", "viewer"})

    def test_the_deepest_folder_owns_a_file(self):
        self.assertEqual(run("crates/render/inner/src/lib.rs"), {"inner"})

    def test_documentation_affects_nothing(self):
        self.assertEqual(run("README.md", "docs/ocean.md", "crates/render/NOTES.md", "LICENSE"), set())

    def test_files_no_workspace_check_reads_affect_nothing(self):
        self.assertEqual(run("tools/gameplay-bench/b_world/src/lib.rs", "scenes/cove.txt"), set())

    def test_anything_outside_the_crates_affects_everything(self):
        for f in ("Cargo.lock", "Cargo.toml", ".github/workflows/ci.yml", "rust-toolchain.toml",
                  "scripts/build-web.sh", "scripts/affected.py", "something-new.rs"):
            with self.subTest(f=f):
                self.assertIsNone(run("crates/world/src/lib.rs", f))

    def test_a_crate_manifest_is_its_own_crate(self):
        self.assertEqual(run("crates/script/Cargo.toml"), {"script", "game", "viewer"})

    def test_no_changes_affect_nothing(self):
        self.assertEqual(run(), set())


if __name__ == "__main__":
    unittest.main()
