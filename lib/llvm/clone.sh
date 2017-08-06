#!/usr/bin/env bash
set -eux

clone() {
    local dest=$1
    git clone http://llvm.org/git/llvm.git $dest
    cd $dest/tools
    git clone http://llvm.org/git/clang.git
    cd ../projects
    git clone http://llvm.org/git/compiler-rt.git
}
clone $@
