#!/usr/bin/env bash
set -eux

wget https://github.com/ninja-build/ninja/releases/download/v1.7.1/ninja-linux.zip
unzip ninja-linux.zip
mv ninja-linux/ninja /usr/bin
