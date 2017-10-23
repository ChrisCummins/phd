#!/usr/bin/env bash
#
# One-liner to install dependencies for CLgen 0.4.0.dev0.
#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# Adapted from Torch's install-deps script, see:
#     https://github.com/torch/distro
#     Copyright (c) 2016, Soumith Chintala, Ronan Collobert,
#     Koray Kavukcuoglu, Clement Farabet All rights reserved.
#
set -e

# Based on Platform:
if [[ `uname` == 'Darwin' ]]; then

    brew tap homebrew/science
    brew install git libffi wget python3

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

        sudo add-apt-repository ppa:jonathonf/python-3.6
        sudo apt-get update
        sudo apt-get install -y build-essential git zlib1g-dev libffi-dev \
            zlib1g-dev curl wget python3-dev python3-pip python3-virtualenv \
            unzip libncurses5-dev libhdf5-dev python3.6 python3.6-venv \
            python3.6-dev clang

    fi

else
    # Unsupported
    echo '==> platform not supported, aborting'
    exit 1
fi

# Done.
echo "==> CLgen's dependencies have been installed"
