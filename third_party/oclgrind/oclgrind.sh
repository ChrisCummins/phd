#!/usr/bin/env bash
#
# A wrapper script for invoking a platform-specific oclgrind binary.
#
# Usage:
#
#    bazel run //third_party:oclgrind -- <oclgrind_args>
#

# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
   source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
   source "$0.runfiles/$f" 2>/dev/null || \
   source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
   source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
   { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=
# --- end app init ---

set -e
if [[ -n $(DataPath oclgrind_linux/bin/oclgrind) ]]; then
  $(DataPath oclgrind_linux/bin/oclgrind) $@
elif [[ -n $(DataPath oclgrind_mac/bin/oclgrind) ]]; then
  $(DataPath oclgrind_mac/bin/oclgrind) $@
else
  echo "oclgrind not found!" >&2
  exit 1
fi
