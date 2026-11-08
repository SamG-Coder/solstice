#!/usr/bin/env bash
# Runs the checks CI runs, for only the crates a change affects (scripts/affected.py): formatting,
# clippy, tests (the GPU tests too, on a machine with a GPU) and, when the viewer is affected, the
# browser builds. Uncommitted and untracked files count, so it covers work in progress.
#
#     scripts/check-changed.sh              # against where this branch left origin/main
#     scripts/check-changed.sh --base REV   # against another revision
#     scripts/check-changed.sh --all        # everything, as CI does for a change outside the crates
#
# It builds in release, as CI does. That also keeps debug builds' Vulkan validation layer out of
# the GPU tests, where it can crash when many devices are made at once.
set -euo pipefail
cd "$(dirname "$0")/.."

python=$(command -v python3 || command -v python)
if [[ "${1:-}" == "--all" ]]; then
  result=all
else
  result=$("$python" scripts/affected.py "$@")
fi
echo "affected: $result"

cargo fmt --all -- --check
"$python" scripts/test_affected.py

case "$result" in
  none)
    echo "No crate is affected: nothing else to check."
    exit 0
    ;;
  all)
    packages=(--workspace)
    web=true
    ;;
  *)
    packages=()
    web=false
    for name in $result; do
      packages+=(-p "$name")
      [[ "$name" == viewer ]] && web=true
    done
    ;;
esac

cargo clippy "${packages[@]}" --all-targets --release -- -D warnings
cargo test "${packages[@]}" --release

if [[ "$web" == true ]]; then
  cargo build --release --target wasm32-unknown-unknown -p viewer --lib --no-default-features
  cargo clippy --release --target wasm32-unknown-unknown -p viewer --lib --no-default-features -- -D warnings
  cargo build --release --target wasm32-unknown-unknown -p viewer --lib
  cargo clippy --release --target wasm32-unknown-unknown -p viewer --lib -- -D warnings
fi
echo "All checks passed for: $result"
