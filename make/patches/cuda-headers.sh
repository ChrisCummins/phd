#!/usr/bin/env bash
set -eu

if [ -d /usr/include/CL ] || [ -d /usr/local/include/CL ]; then
	exit
fi

if [ ! -d /usr/local/cuda/include/CL ]; then
	exit
fi

echo "patching CUDA headers into /usr/local/include ..."
echo "This requires sudo permissions, please enter your password if prompted."
set +x
sudo ln -s /usr/local/cuda/include/CL /usr/local/include/
