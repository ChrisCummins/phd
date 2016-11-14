#!/usr/bin/env bash
set -eu

ffi=(clgen/data/torch/share/lua/*/hdf5/ffi.lua)
sed "s,gcc -E \",gcc -E -D '_Nullable=' \"," -i "$ffi"
