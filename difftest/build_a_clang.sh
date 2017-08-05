#!/usr/bin/env bash
build_a_clang() {
    local dest=$1
    git clone http://llvm.org/git/llvm.git $dest
    cd $dest/tools
    git clone http://llvm.org/git/clang.git
    cd ../projects
    git clone http://llvm.org/git/compiler-rt.git
}

build_a_clang $1
