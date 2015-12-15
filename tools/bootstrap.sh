#!/usr/bin/env bash
#
# bootstrap.sh - Build the project toolchain
#
# Writes the date that it was last run to a "bootstrap file" upon
# completion. This can be checked using the --check argument.
#
# Usage:
#
#     ./boostrap.sh [--check]
#
set -eu

# Project sources root:
ROOT=~/phd

# Number of build threads:
NPROC=4

# Location of boostrapped file:
BOOTSTRAP_FILE=$ROOT/.bootstrapped


build_llvm() {
    mkdir -vp tools/llvm/build
    cd $ROOT/tools/llvm/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$NPROC
    cd $ROOT
}

build_libcxx() {
    mkdir -vp extern/libcxx/build
    cd $ROOT/extern/libcxx/build
    cmake .. -DLLVM_CONFIG=$ROOT/tools/llvm/build/bin/llvm-config
    make -j$NPROC
    cd $ROOT
}

write_bootstrap_file() {
    date > $BOOTSTRAP_FILE
}

print_boostrapped_info() {
    if [[ -f $BOOTSTRAP_FILE ]]; then
        echo "Boostrapped on: $(cat $BOOTSTRAP_FILE)"
    else
        echo "Not bootstrapped."
    fi
}

boostrap() {
    build_llvm
    build_libcxx
    write_bootstrap_file
}

main() {
    set +u
    arg=$1
    set -u

    if [[ -n "$arg" ]]; then
        case "$arg" in
            "--check")
                print_boostrapped_info
                exit
                ;;
            *)
                echo "boostrap: Unrecognised command '$arg'!" >&2
                exit 1
                ;;
        esac
    fi

    # No argument, run bootstrap:
    boostrap
}
main $@
