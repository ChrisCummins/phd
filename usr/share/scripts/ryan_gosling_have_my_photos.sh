#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0) <rsync-args>"
}
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 1
fi

set -eu

# Use this to optionally specify a subdirectory within the photo library for
# partial syncs. This must end in a trailing slash.
subpath=""

# Use the local area network if available.
remote_lan="192.168.0.203"
remote_wan="ryangosling"
remote_port=65335

if ping -c1 -W1 "$remote_lan" &>/dev/null; then
    remote="$remote_lan"
else
    remote="$remote_wan"
fi

if [[ -d "/Volumes/Data/$subpath" ]]; then
    src="/Volumes/Data/$subpath"
else
    src="/Volumes/Orange/$subpath"
fi
dst="$remote:img/photos/$subpath"

if [[ ! -d "$src" ]]; then
    echo "fatal: '$src' not found" >&2
    exit 1
fi

if find "$src" -name '*.lrcat.lock' 2>/dev/null | grep lrcat.lock ; then
    echo "fatal: $(find "$src" -name '*.lrcat.lock') found. Close Lightroom."
    exit 1
fi

echo "pushing from $src to $dst"

set -x

rsync -avh --delete "$src" "$dst" \
    -e 'ssh -p 65335' $@ \
    --exclude "*.lrcat-journal" \
    --exclude "*.lrcat.lock" \
    --exclude "._.DS_Store" \
    --exclude ".com.apple.timemachine.supported" \
    --exclude ".DS_Store" \
    --exclude ".sync.ffs_db" \
    --exclude "/.DocumentRevisions-V100" \
    --exclude "/.fseventsd" \
    --exclude "/.Spotlight-V100" \
    --exclude "/.TemporaryItems" \
    --exclude "/.Trashes" \
    --exclude "/.VolumeIcon.icns" \
    --exclude "/.VolumeIcon.ico" \
    --exclude "/autorun.inf"
