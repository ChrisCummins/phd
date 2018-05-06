#!/usr/bin/env bash

set -eu

main() {
    local language="$1"
    local extension="$2"

    local indir=~/dsmith/miner/repos/$language
    local outdir=~/dsmith/miner/files/$language
    mkdir -pv "$outdir"

    find "$indir" -type f -name '*'$extension | grep -v .git/ | while read f; do
        sha1sum=$(md5sum "$f" | awk '{print $1};')
        dst="$outdir/$sha1sum$extension"
        [ -f "$dst" ] || cp -v "$f" "$dst"
        chmod -x "$dst"
    done
}

main $@
