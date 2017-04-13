#!/usr/bin/env bash
#
# bootstrap.sh - Prepare the toolchain
#
# Usage:
#
#     ./boostrap.sh
#
set -eu


bootstrap_macos() {
    set -x
    brew list | grep '^gcc$' &>/dev/null || brew install gcc --without-multilib
    brew cask list | grep '^java$' &>/dev/null || brew cask install java
    brew list | grep '^bazel$' &>/dev/null || brew install bazel
}


bootstrap_ubuntu() {
    set -x

    # add bazel APT repositories
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

    sudo apt-get update
    dpkg -s 'bazel' &>/dev/null || sudo apt-get install -y bazel
}


main() {
    if [[ "$(uname)" != "Darwin" ]]; then
        bootstrap_macos
    else
        bootstrap_ubuntu
    fi
}
main $@
