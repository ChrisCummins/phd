#!/usr/bin/env bash
set -eux

mkdir -p ninja
cd ninja
wget https://github.com/ninja-build/ninja/archive/v1.7.1.tar.gz -O ninja.tar.gz
tar -xf ninja.tar.gz --strip-components=1
./configure.py --bootstrap
mv ninja /usr/bin
