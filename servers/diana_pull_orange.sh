#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0)"
}
test -z "$1" || { usage; exit 1; }
set -eu


rsync_dir() {
    local src="$1"
    local dst="$2"

    if [[ ! -d "$src" ]]; then
        echo "fatal: '$src' not found" >&2
        exit 1
    fi

    if [[ ! -d "$dst" ]]; then
        echo "fatal: '$dst' not found" >&2
        exit 1
    fi

    set -x

    rsync -avh --delete "$src" "$dst"
}


main() {
    rsync_dir "/Volumes/Orange/Gallery/"             "/Volumes/Data/Gallery/"
    rsync_dir "/Volumes/Orange/Lightroom Catalogue/" "/Volumes/Data/Lightroom Catalogue/"
    rsync_dir "/Volumes/Orange/Naughty/"             "/Volumes/Data/Naughty/"
    rsync_dir "/Volumes/Orange/Photo Library/"       "/Volumes/Data/Photo Library/"
}
main $@
