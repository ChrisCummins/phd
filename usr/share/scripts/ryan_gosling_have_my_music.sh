#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0) <rsync-args>"
}
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 1
fi

set -eu

# Use the local area network if available.
remote_lan="192.168.0.203"
remote_wan="ryangosling"
remote_port=65335

if ping -c1 -W1 "$remote_lan" &>/dev/null; then
    remote="$remote_lan"
else
    remote="$remote_wan"
fi

src="/Users/cec/Music/Music Library/"
dst="$remote:audio/third_party/"

if [[ ! -d "$src" ]]; then
    echo "fatal: '/Volumes/Music Library' not mounted" >&2
    exit 1
fi

echo "pushing to $dst"

set -x

rsync -avh --delete "$src" "$dst" \
        -e "ssh -p $remote_port" $@ \
        --exclude "._.DS_Store" \
        --exclude ".DS_Store" \
        --exclude ".iTunes Preferences.plist" \
        --exclude "Automatically Add to iTunes.localized" \
        --exclude "Downloads" \
        --exclude "Mobile Applications"
