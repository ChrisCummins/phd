#!/usr/bin/env bash
set -eux

wget http://www.cmake.org/files/v3.5/cmake-3.5.0-Linux-x86_64.sh -O cmake.sh
chmod +x cmake.sh
./cmake.sh --prefix=/usr/ --skip-license --exclude-subdir
