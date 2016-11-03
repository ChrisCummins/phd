#!/usr/bin/env bash
set -eu

# determine whether 64 bit libdir is a thing
lib=lib
test -d /lib64 && lib=lib

if [ -f /usr/$lib/libOpenCL.so ] || [ -f /usr/local/$lib/libOpenCL.so ]; then
	exit
fi

if [ ! -d /usr/local/cuda/$lib/libOpenCL.so ]; then
	exit
fi

echo "patching CUDA OpenCL library into /usr/local/$lib ..."
echo "This requires sudo permissions, please enter your password if prompted."
set +x
sudo ln -s /usr/local/cuda/$lib/libOpenCL.so /usr/local/$lib/
