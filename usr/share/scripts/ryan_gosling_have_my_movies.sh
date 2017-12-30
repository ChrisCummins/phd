#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0)"
}
test -z "$1" || { usage; exit 1; }

set -eu

if [[ ! -d /Volumes/Movies/ ]]; then
    echo "fatal: '/Volumes/Movies' not mounted" >&2
    exit 1
fi

if [[ ! -d /Volumes/TV\ Shows/ ]]; then
    echo "fatal: '/Volumes/TV Shows' not mounted" >&2
    exit 1
fi

set -x

rsync -avh \
    /Users/cec/Movies/ /Volumes/Movies/ \
    --exclude "._.DS_Store" \
    --exclude ".DS_Store" \
    --exclude ".localized"

rsync -avh \
    /Users/cec/TV Shows/ /Volumes/TV Shows/ \
    --exclude "._.DS_Store" \
    --exclude ".DS_Store" \
    --exclude ".localized"