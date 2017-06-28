#!/usr/bin/env bash

set -eux

samples_dir="$(pwd)/../data/bootstrap/samples/"

cd ../difftest

./clgen-fetch.py "$samples_dir" --cl_launchable

set +x
echo "done. now, run manually:"
echo
echo "    $ cd ../difftest && ./run-programs.py <platform-id> <device-id>"
