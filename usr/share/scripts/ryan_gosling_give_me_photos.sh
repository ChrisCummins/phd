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
subpath="photos/"

# Use the local area network if available.
remote_lan="192.168.0.203"
remote_wan="ryangosling"
remote_port=65335

if ping -c1 -W1 "$remote_lan" &>/dev/null; then
    remote="$remote_lan"
else
    remote="$remote_wan"
fi

src="$remote:img/photos/$subpath"
dst="/Volumes/Data/$subpath"

if [[ ! -d "$dst" ]]; then
    echo "fatal: '$dst' not found" >&2
    exit 1
fi

if find "$dst" -name '*.lrcat.lock' 2>/dev/null | grep lrcat.lock ; then
    echo "fatal: $(find "$dst" -name '*.lrcat.lock') found. Close Lightroom."
    exit 1
fi

echo "pulling from $src to $dst"

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
