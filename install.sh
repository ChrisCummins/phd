#!/usr/bin/env bash
set -euv
make llvm
make
sudo make install
