#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0) <rsync-args>"
}
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 1
fi

set -eu

local_dst="/Volumes/Photo Library/"
remote_dst="ryangosling:img/photos/"

src="/Volumes/Orange/"

if [[ ! -d "$src" ]]; then
    echo "fatal: '$src' not found" >&2
    exit 1
fi

# Select between locally mounted network share or SSH:
if [[ -d "$local_dst" ]]; then
    dst="$local_dst"
else
    dst="$remote_dst"
fi
echo "pushing to $dst"

set -x

# copy from source to dest, excluding Lightroom Previews:
rsync -avh --delete "$src" "$dst" $@ \
    --exclude "*.lrcat-journal" \
    --exclude "*.lrcat.lock" \
    --exclude "*.lrdata" \
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
