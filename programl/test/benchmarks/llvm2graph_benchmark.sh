#!/usr/bin/env bash
# --- begin labm8 init ---
f=phd/labm8/sh/app.sh
# shellcheck disable=SC1090
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
# --- end app init ---

LLVM2GRAPH="$(DataPath phd/programl/cmd/llvm2graph)"
LLVM_IR="$(DataPath phd/programl/test/data/llvm_ir)"

if [[ -f /usr/local/opt/gnu-time/bin/gtime ]]; then
  TIME=/usr/local/opt/gnu-time/bin/gtime
else
  TIME=time
fi

main() {
  for f in $(find "$LLVM_IR" -type f -o -type l); do
    local size="$(du -h $(readlink $f) | sed 's/^\s+//' | cut -f1)"
    "$TIME" -f "file_size=$size, time=%es, max_resident_size=%MK" "$LLVM2GRAPH" <$f >/dev/null
  done
}
main $@
