#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0) {push,pull}"
}
set -e


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
    if [[ "$1" == "push" ]]; then
        # push to orange
        rsync_dir "/Volumes/Data/Gallery/"             "/Volumes/Orange/Gallery/"
        rsync_dir "/Volumes/Data/Lightroom Catalogue/" "/Volumes/Orange/Lightroom Catalogue/"
        rsync_dir "/Volumes/Data/Naughty/"             "/Volumes/Orange/Naughty/"
        rsync_dir "/Volumes/Data/Photo Library/"       "/Volumes/Orange/Photo Library/"
    elif [[ "$1" == "pull" ]]; then
        # pull from orange
        rsync_dir "/Volumes/Orange/Gallery/"             "/Volumes/Data/Gallery/"
        rsync_dir "/Volumes/Orange/Lightroom Catalogue/" "/Volumes/Data/Lightroom Catalogue/"
        rsync_dir "/Volumes/Orange/Naughty/"             "/Volumes/Data/Naughty/"
        rsync_dir "/Volumes/Orange/Photo Library/"       "/Volumes/Data/Photo Library/"
    else
        usage
        exit 1
    fi
}
main $@
