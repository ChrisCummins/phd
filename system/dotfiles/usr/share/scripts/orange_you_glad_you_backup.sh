#!/usr/bin/env bash
#
# Usage:
#     $ orange_you_glad_you_backup [rsync-options]
set -eu

# Paths must end with a trailing slash.
src="/Volumes/Orange/"
dst="/Volumes/Satsuma/"

backup() {
  local src="$1"
  local dst="$2"
  shift
  shift

  if [[ ! -d "$src" ]]; then
    echo "fatal: '$src' not found" >&2
    exit 1
  fi

  if [[ ! -d "$dst" ]]; then
    echo "fatal: '$dst' not found" >&2
    exit 1
  fi

  echo
  echo "================================================================="
  set -x
  rsync "$src" "$dst" \
    -avh --delete \
    --exclude "*.lrcat-journal" \
    --exclude "*.lrcat.lock" \
    --exclude "*.lrdata" \
    --exclude "._.DS_Store" \
    --exclude ".com.apple.timemachine.supported" \
    --exclude ".DS_Store" \
    --exclude ".sync.ffs_db" \
    --exclude "/.cache" \
    --exclude "/.DocumentRevisions-V100" \
    --exclude "/.fseventsd" \
    --exclude "/.Spotlight-V100" \
    --exclude "/.TemporaryItems" \
    --exclude "/.Trashes" \
    --exclude "/.VolumeIcon.icns" \
    --exclude "/.VolumeIcon.ico" \
    --exclude "/autorun.inf" \
    $@
  set +x
  echo "================================================================="
  echo
}

main() {
  backup "$src" "$dst" $@
}
main $@
