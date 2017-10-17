#!/usr/bin/env bash
set -eux

clone() {
    local dest=$1
    test -d $dest || git clone http://llvm.org/git/llvm.git $dest
    cd $dest/tools
    test -d clang || git clone http://llvm.org/git/clang.git
    cd ../projects
    test -d compiler-rt || git clone http://llvm.org/git/compiler-rt.git
}
clone $@
