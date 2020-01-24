#!/usr/bin/env bash
#
# A wrapper script for llvm2graph which forces LD_PRELOAD for some LLVM libs.
#

# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null ||
  source "$0.runfiles/$f" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null ||
  {
    echo >&2 "ERROR: cannot find $f"
    exit 1
  }
f=
# --- end labm8 init ---

set -eu
LLVM_LIB_DIR="$(DataPath llvm_linux/lib)"
LD_PRELOAD="${LLVM_LIB_DIR}/libLTO.so:${LLVM_LIB_DIR}/libclang.so" \
  $(DataPath phd/deeplearning/ml4pl/graphs/llvm2graph/llvm2graph) "$@"
