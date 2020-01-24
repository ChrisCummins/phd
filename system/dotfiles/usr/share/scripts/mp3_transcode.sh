#!/usr/bin/env bash
# mp3_transcode.sh - Transcode files to mp3 using VLC.

usage() {
  echo "usage: $(basename $0) <files-to-convert-to-mp3>"
}
if [ -z "$1" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 1
fi

set -eu

# Determine path to VLC binary.
if [[ "$(uname)" == "Darwin" ]]; then
  vlc="/Applications/VLC.app/Contents/MacOS/VLC"
else
  vlc="$(which vlc)"
fi

# Check that VLC exists.
if [ ! -e "$vlc" ]; then
  echo "fatal: VLC not found." >&2
  exit 1
fi

for file in "$@"; do
  echo "=> Transcoding '$file'... "
  dst=$(dirname "$file")
  new=$(basename "$file" | sed 's@\.[a-z][a-z][a-z]$@@').mp3
  $vlc -I dummy "$file" \
    ":sout=#transcode{acodec=mpga,ab=192}:std{dst=\"$dst/$new\",access=file}" \
    vlc://quit
  ls -lh "$file" "$dst/$new"
  echo
done
