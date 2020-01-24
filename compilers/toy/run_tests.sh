#!/usr/bin/env bash
set -eux
compile="$PWD/compilers/toy/compile.sh"
cd ./compilers/toy/test_data
./test_compiler.sh \
  "$compile" $@
