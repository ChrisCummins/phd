#!/usr/bin/env bash
set -eux

wget https://github.com/ninja-build/ninja/releases/download/v1.7.1/ninja-linux.zip
unzip ninja-linux.zip
mv ninja /usr/bin

ninja_path=$(which ninja)
echo "path to ninja: $ninja_path"
