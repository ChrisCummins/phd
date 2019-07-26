#!/usr/bin/env bash
# Create a python virtualenv parallel to this project and install the project
# dependencies.
#
# Usage:
#     $ ./tools/create_python_virtualenv.sh
set -eux
DIR="$PHD/../venv/phd"

rm -rf "$DIR"
mkdir -pv "$(dirname $DIR)"
python -m virtualenv "$DIR"
source "$DIR/bin/activate"
python -m pip install -r requirements.txt
