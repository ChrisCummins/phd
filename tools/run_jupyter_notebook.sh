#!/usr/bin/env bash

#
# run_jupyter_notebook.sh - Run Jupyter notebook server.
#
set -eu

# Directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main() {
    source "$DIR/jupyter.runfiles/phd/venv/phd/bin/activate"
    cd "$DIR/.."
    jupyter notebook
}

main $@
