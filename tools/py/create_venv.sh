#!/usr/bin/env bash
# Create a python virtualenv and install the project dependencies.
#
# Usage:
#     $ ./tools/py/create_venv.sh
set -eux
DIR="$HOME/.cache/phd/tools/py/venv/phd"

rm -rf "$DIR"
mkdir -pv "$(dirname $DIR)"
python3 -m venv "$DIR"
set +u # The virtualenv activation script references potentially unbound variables.
source "$DIR/bin/activate"
python -m pip install -r requirements.txt
