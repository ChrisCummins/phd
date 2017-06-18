#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0)"
}
test -z "$1" || { usage; exit 1; }

set -eu

if [[ ! -d /Volumes/Music\ Library/ ]]; then
    echo "fatal: '/Volumes/Music Library' not mounted" >&2
    exit 1
fi

set -x

rm-dsstore /Users/cec/Music/Music\ Library/
rsync -avh --delete \
        /Users/cec/Music/Music\ Library/ /Volumes/Music\ Library/ \
        --exclude ".iTunes Preferences.plist" \
        --exclude "Automatically Add to iTunes.localized" \
        --exclude "Downloads" \
        --exclude "Mobile Applications"
