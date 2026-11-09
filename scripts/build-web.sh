#!/usr/bin/env bash
# Builds the browser viewer into dist/, a self-contained static site. Bake the scenes first:
#   cargo run --release -p bake -- --quality web --output assets/web.solstice
#
# The wasm-bindgen tool must match the wasm-bindgen library exactly, so the matching
# version is installed inside target/, leaving any global installation alone.
set -euo pipefail
cd "$(dirname "$0")/.."

version=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"$/\1/p')
tools=target/tools
bindgen="$tools/bin/wasm-bindgen"

if ! "$bindgen" --version 2>/dev/null | grep -q "$version"; then
  echo "installing wasm-bindgen-cli $version..."
  cargo install wasm-bindgen-cli --version "$version" --root "$tools" --locked
fi

# Two builds (docs/gameplay.md): the plain viewer, which every page loads, and the gameplay
# viewer, with scripts and physics, loaded only to play a scene's game. The gameplay build turns
# on wasm SIMD, which halves physics' cost and every browser with WebGPU has. Different compiler
# flags can't share compiled crates, so it has a target folder of its own.
rm -rf dist
mkdir -p dist/assets
cargo build --release --target wasm32-unknown-unknown -p viewer --lib --no-default-features
"$bindgen" --target web --no-typescript --out-dir dist/pkg   target/wasm32-unknown-unknown/release/solstice_viewer.wasm
RUSTFLAGS="-C target-feature=+simd128"   cargo build --release --target wasm32-unknown-unknown -p viewer --lib --target-dir target/web-play
"$bindgen" --target web --no-typescript --out-dir dist/play   target/web-play/wasm32-unknown-unknown/release/solstice_viewer.wasm
cp apps/viewer/web/index.html dist/
# Scene files are already compressed where it helps, part by part, so that the page can fetch
# any part alone. Hard links keep big scenes from taking disk space twice.
for scene in assets/*.solstice; do
  # None in CI: there, scenes are served by their own sites.
  [ -e "$scene" ] || continue
  ln -f "$scene" "dist/assets/$(basename "$scene")" 2>/dev/null || cp "$scene" "dist/assets/"
done
# Games kept as folders of text, to play on scenes baked without them (?game=NAME): each
# folder in scenes/ with a scene.game of its own.
for game in scenes/*/scene.game; do
  [ -e "$game" ] || continue
  mkdir -p "dist/games/$(basename "$(dirname "$game")")"
  cp -r "$(dirname "$game")"/* "dist/games/$(basename "$(dirname "$game")")/"
done

echo "built dist/. To try it: python scripts/serve.py"
