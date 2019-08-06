#!/usr/bin/env bash
# Set the version file (//:version.txt) using today's date.
set -eux
test -f version.txt
date '+%Y.%m.%d' > version.txt
cat version.txt
