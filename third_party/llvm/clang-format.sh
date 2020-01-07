#!/usr/bin/env bash
#
# A wrapper script for clang-format.
#
# Usage:
#
#    bazel run //third_party/llvm:clang-format -- <clang-format_args>
#

# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
   source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
   source "$0.runfiles/$f" 2>/dev/null || \
   source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
   source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
   { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=
# --- begin app init ---

set -e
if [[ -n $(DataPath llvm_mac/bin/clang-format) ]]; then
  $(DataPath llvm_mac/bin/clang-format) $@
elif [[ -n $(DataPath llvm_linux/bin/clang-format) ]]; then
  $(DataPath llvm_linux/bin/clang-format) $@
fi
