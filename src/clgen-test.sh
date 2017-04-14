#!/usr/bin/env bash

cd src/clgen

virtualenv -p python3.6 env &>/dev/null
. env/bin/activate &>/dev/null

echo "PYTHON: $(which python)"
echo "DIR $PWD"

./configure -b
make all
make install
echo
echo
echo "FILES:"
ls tests
echo
echo
make test
