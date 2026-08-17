#!/usr/bin/env bash
# Configure, build, test and benchmark warpsmith on Linux.
#
# usage: scripts/build.sh [--test] [--bench] [--suite NAME] [--arch 86] [--clean]
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build"
CONFIG="Release"
ARCH=""
RUN_TEST=0
RUN_BENCH=0
SUITE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)   RUN_TEST=1; shift ;;
    --bench)  RUN_BENCH=1; shift ;;
    --suite)  SUITE="$2"; shift 2 ;;
    --arch)   ARCH="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --debug)  CONFIG="Debug"; shift ;;
    --clean)  rm -rf "$BUILD_DIR"; shift ;;
    -h|--help)
      sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v nvcc >/dev/null || { echo "nvcc not found on PATH; install the CUDA Toolkit" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake not found on PATH" >&2; exit 1; }

CMAKE_ARGS=(-S . -B "$BUILD_DIR" "-DCMAKE_BUILD_TYPE=$CONFIG")
[[ -n "$ARCH" ]] && CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$ARCH")

echo "==> cmake configure"
cmake "${CMAKE_ARGS[@]}"

echo "==> cmake build"
cmake --build "$BUILD_DIR" --parallel "$(nproc)"

if [[ "$RUN_TEST" == 1 ]]; then
  echo "==> ctest"
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

if [[ "$RUN_BENCH" == 1 ]]; then
  mkdir -p results docs/charts
  echo "==> benchmark ($SUITE)"
  "./$BUILD_DIR/warpsmith_bench" --suite "$SUITE" --json results/results.json
  echo "==> regenerating report and charts"
  python3 tools/report.py results/results.json --out results/REPORT.md
  python3 tools/plot.py results/results.json --outdir docs/charts
fi
