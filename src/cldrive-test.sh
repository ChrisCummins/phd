#!/usr/bin/env bash

cd src/cldrive

virtualenv -p python3.6 env &>/dev/null
. env/bin/activate &>/dev/null

echo "PYTHON: $(which python)"
echo "DIR $PWD"

make install &>/dev/null
echo
echo
echo "FILES:"
ls tests
echo
echo
pytest tests
