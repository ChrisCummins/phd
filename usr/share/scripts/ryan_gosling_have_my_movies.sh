#!/usr/bin/env bash
usage() {
    echo "usage: $(basename $0) <rsync-args>"
}
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 1
fi

set -eu

movies_src="$HOME/Movies/"
local_movies_dst="/Volumes/Movies/"
remote_movies_dst="ryangosling:video/library/movies/"

tv_src="$HOME/TV Shows/"
local_tv_dst="/Volumes/TV Shows/"
remote_tv_dst="ryangosling:video/library/tv/"

# Select between locally mounted network share or SSH:
if [[ -d "$local_movies_dst" ]]; then
    movies_dst="$local_movies_dst"
else
    movies_dst="$remote_movies_dst"
fi
echo "pushing to $movies_dst"

# Select between locally mounted network share or SSH:
if [[ -d "$local_tv_dst" ]]; then
    tv_dst="$local_tv_dst"
else
    tv_dst="$remote_tv_dst"
fi
echo "pushing to $tv_dst"

set -x

rsync -avh "$movies_src" "$movies_dst" $@ \
    --exclude "._.DS_Store" \
    --exclude ".DS_Store" \
    --exclude ".localized"

rsync -avh "$tv_src" "$tv_dst" $@ \
    --exclude "._.DS_Store" \
    --exclude ".DS_Store" \
    --exclude ".localized"
