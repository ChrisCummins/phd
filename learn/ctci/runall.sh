#!/usr/bin/env bash

set -u

for exe in $(find . -type f -executable -not -name '*.sh'); do
    echo "    $exe ..."
    echo
    $exe 2>&1
    echo
    echo
done
