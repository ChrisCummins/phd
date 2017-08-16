#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0)"
}
test -z "$1" || { usage; exit 1; }

set -eu

if [[ ! -d /Volumes/Photo\ Library/ ]]; then
    echo "fatal: '/Volumes/Photo Library' not mounted" >&2
    exit 1
fi

set -x

# transfer photo library
rm-dsstore /Volumes/Data/Photo\ Library/
rsync -avh --delete \
        /Volumes/Data/Photo\ Library/ /Volumes/Photo\ Library/ \
        --exclude "/Lightroom/*"

# transfer Lightroom catalogue
rm-dsstore /Users/cec/Dropbox/Pictures/Lightroom/
rsync -avh /Users/cec/Dropbox/Pictures/Lightroom/ /Volumes/Photo\ Library/Lightroom/ \
--exclude "/Backups" \
        --exclude "/Mobile Downloads.lrdata" \
        --exclude "/Photo Library Previews.lrdata" \
        --exclude "/Photo Library Smart Previews.lrdata"
