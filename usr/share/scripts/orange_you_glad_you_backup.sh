#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0)"
}
test -z "$1" || { usage; exit 1; }

set -eu

src="/Volumes/Orange"
dst="/Volumes/Satsuma"

if [[ ! -d "$src" ]]; then
    echo "fatal: '$src' not found" >&2
    exit 1
fi

if [[ ! -d "$dst" ]]; then
    echo "fatal: '$dst' not found" >&2
    exit 1
fi

stats() {
    echo "Disk usage:"
    df -h | grep "$src"
    df -h | grep "$dst"

}

stats
echo
echo "================================================================="
rsync -avh --delete "$src/" "$dst/" \
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
echo "================================================================="
echo
stats
