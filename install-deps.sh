#!/usr/bin/env bash
#
# One-liner to install dependencies for CLgen 0.1.1.
#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
#
# Adapted from Torch's install-deps script, see:
#     https://github.com/torch/distro
#     Copyright (c) 2016, Soumith Chintala, Ronan Collobert,
#     Koray Kavukcuoglu, Clement Farabet All rights reserved.
#
torch_version=3467b980c56942451ee242937dbe76d15fcfc5ab
torch_deps=https://raw.githubusercontent.com/ChrisCummins/distro/$torch_version/install-deps
curl -s "$torch_deps" | bash

set -e

# Based on Platform:
if [[ `uname` == 'Darwin' ]]; then

    brew tap homebrew/science
    brew install git hdf5 libffi wget

elif [[ "$(uname)" == 'Linux' ]]; then

    if [[ -r /etc/os-release ]]; then
        # this will get the required information without dirtying any env state
        DIST_VERS="$( ( . /etc/os-release &>/dev/null
                        echo "$ID $VERSION_ID") )"
        DISTRO="${DIST_VERS%% *}" # get our distro name
        VERSION="${DIST_VERS##* }" # get our version number
    elif [[ -r /etc/redhat-release ]]; then
        DIST_VERS=( $( cat /etc/redhat-release ) ) # make the file an array
        DISTRO="${DIST_VERS[0],,}" # get the first element and get lcase
        VERSION="${DIST_VERS[2]}" # get the third element (version)
    elif [[ -r /etc/lsb-release ]]; then
        DIST_VERS="$( ( . /etc/lsb-release &>/dev/null
                        echo "${DISTRIB_ID,,} $DISTRIB_RELEASE") )"
        DISTRO="${DIST_VERS%% *}" # get our distro name
        VERSION="${DIST_VERS##* }" # get our version number
    else # well, I'm out of ideas for now
        echo '==> Failed to determine distro and version.'
        exit 1
    fi

    if [[ "$DISTRO" == 'ubuntu' ]]; then

        sudo apt-get install -y build-essential python-dev python-virtualenv \
            python-pip git zlib1g-dev libhdf5-dev libffi-dev zlib1g-dev \
            curl wget

    fi

else
    # Unsupported
    echo '==> platform not supported, aborting'
    exit 1
fi

# Done.
echo "==> CLgen's dependencies have been installed"
