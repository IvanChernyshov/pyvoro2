#!/usr/bin/env bash
set -euo pipefail

# Local manylinux wheel build via cibuildwheel (WSL/Linux).
# Usage:
#   tools/build_wheels_wsl.sh              # outputs to ./wheelhouse
#   tools/build_wheels_wsl.sh dist_wheels  # outputs to ./dist_wheels
#
# Override defaults (examples):
#   CIBW_BUILD="cp312-manylinux_x86_64" tools/build_wheels_wsl.sh
#   CIBW_SKIP="*musllinux* pp*"         tools/build_wheels_wsl.sh

OUT_DIR="${1:-wheelhouse}"

# Defaults that match your usual one-liner.
export CIBW_BUILD="${CIBW_BUILD:-cp311-manylinux_x86_64}"
export CIBW_SKIP="${CIBW_SKIP:-*musllinux*}"

# Optional: uncomment to pin the manylinux image for consistency across machines.
# export CIBW_MANYLINUX_X86_64_IMAGE="${CIBW_MANYLINUX_X86_64_IMAGE:-manylinux_2_28}"

# Basic sanity checks (helps a lot in WSL).
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker / enable WSL integration." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon not running or not accessible from WSL." >&2
  exit 1
fi

echo "cibuildwheel build:"
echo "  CIBW_BUILD=${CIBW_BUILD}"
echo "  CIBW_SKIP=${CIBW_SKIP}"
echo "  output=${OUT_DIR}"

python -m cibuildwheel --output-dir "${OUT_DIR}"
