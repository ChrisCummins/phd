#!/usr/bin/env bash
set -eux

build() {
    local srcdir=$(readlink -f $1)
    local builddir=$2
    local cmake=${3:-cmake}
    local ninja=${4:-ninja}

    mkdir -pv $builddir
    cd $builddir
    $cmake $srcdir -G Ninja -DLLVM_TARGETS_TO_BUILD="X86"
    $ninja
}
build $@
