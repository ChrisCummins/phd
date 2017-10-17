#!/usr/bin/env bash
set -eux

checkout() {
    local dest=$1
    local release=$2

    cd $dest
    git checkout $release
    rm -rf .git
    cd tools/clang
    git checkout $release
    rm -rf .git
    cd ../../projects/compiler-rt
    git checkout $release
    rm -rf .git
}

checkout $@
