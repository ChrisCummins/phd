#!/usr/bin/env bash

set -eu

main() {
        local testroot="$1"
        local python="$2"

        cd "$testroot"

        # virtualenv in tmpdir
        env="$(mktemp -d)"
        if [[ ! "$env" || ! -d "$env" ]]; then
        echo "failed to create temporary directory" >&2
        exit 2
        fi
        rm_env() {
                rm -rf "$env"
        }
        trap rm_env EXIT

        virtualenv -p "$python" "$env" >/dev/null
        set +u  # activate script tests for undefined vars
        . "$env/bin/activate" >/dev/null
        set -u

        echo
        echo "================================================="
        echo "RUNNER: $0"
        echo "CWD:    $PWD"
        echo "VENV:   $env"
        echo "PYTHON: python --version $(which python)"
        echo "PIP: pip --version"
        echo "================================================="
        echo

        set -x
        ./configure -b
        make all
        make test
}
main $@
