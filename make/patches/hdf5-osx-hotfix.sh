#!/usr/bin/env bash
set -eu

if [ -d /usr/local/Cellar/hdf5/*/include ]; then
	echo "patching torch hdf5 include path..."
	hdf5_config=(native/torch/*/build/share/lua/*/hdf5/config.lua)
	hdf5_include=(/usr/local/Cellar/hdf5/*/include)

	sed "s!HDF5_INCLUDE_PATH =.*!HDF5_INCLUDE_PATH = \"$hdf5_include\",!" -i "$hdf5_config"
fi
