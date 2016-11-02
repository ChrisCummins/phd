#!/usr/bin/env bash
set -eu

ffi=(native/torch/*/build/share/lua/*/hdf5/ffi.lua)
sed "s,gcc -E \",gcc -E -D '_Nullable=' \"," -i "$ffi"
