#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0) <rsync-args>"
}
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 1
fi

set -eu

src="/Users/cec/Music/Music Library/"

local_dst="/Volumes/Music Library/"
remote_dst="ryangosling:audio/library/"

if [[ ! -d "$src" ]]; then
    echo "fatal: '/Volumes/Music Library' not mounted" >&2
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

rsync -avh --delete "$src" "$dst" $@ \
        --exclude "._.DS_Store" \
        --exclude ".DS_Store" \
        --exclude ".iTunes Preferences.plist" \
        --exclude "Automatically Add to iTunes.localized" \
        --exclude "Downloads" \
        --exclude "Mobile Applications"
