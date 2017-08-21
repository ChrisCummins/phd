#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0)"
}
test -z "$1" || { usage; exit 1; }

set -eu

src="/Volumes/Orange/Photo Library/"
dst="/Volumes/Photo Library/"

if [[ ! -d "$src" ]]; then
    echo "fatal: '$src' not found" >&2
    exit 1
fi

if [[ ! -d "$dst" ]]; then
    echo "fatal: '$dst' not found" >&2
    exit 1
fi

set -x

rm-dsstore "$src"
rsync -avh --delete "$src" "$dst" \
    --exclude "/Lightroom/Mobile Downloads.lrdata" \
    --exclude "/Lightroom/Photo Library Previews.lrdata" \
    --exclude "/Lightroom/Photo Library Smart Previews.lrdata"
