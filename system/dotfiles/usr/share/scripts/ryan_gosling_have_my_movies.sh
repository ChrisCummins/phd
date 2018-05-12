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
remote_lan="ryangosling"
remote_wan="ryangosling.wan"
remote_port=65335

if ping -c1 -W1 "$remote_lan" &>/dev/null; then
    remote="$remote_lan"
else
    remote="$remote_wan"
fi

movies_src="$HOME/Movies/"
movies_dst="$remote:video/third_party/movies/"

tv_src="$HOME/TV Shows/"
tv_dst="$remote:video/third_party/tv/"

echo "pushing to $movies_dst and $tv_dst"

set -x

rsync -avh "$movies_src" "$movies_dst" \
    -e "ssh -p $remote_port" $@ \
    --exclude "._.DS_Store" \
    --exclude ".DS_Store" \
    --exclude ".localized"

rsync -avh "$tv_src" "$tv_dst" \
    -e "ssh -p $remote_port" $@ \
    --exclude "._.DS_Store" \
    --exclude ".DS_Store" \
    --exclude ".localized"
