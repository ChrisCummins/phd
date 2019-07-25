#!/usr/bin/env bash
PREFIX=$HOME/.local
set -eux

mkdir -pv $PREFIX/bin
rm -f $PREFIX/bin/jasper
cp -v util/jasper/jasper.par $PREFIX/bin/jasper
