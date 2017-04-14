#!/usr/bin/env bash
#
# bootstrap.sh - Prepare the toolchain
#
# Usage:
#
#     ./boostrap.sh
#
set -eu


main() {
    # Build system: Bazel
    if [[ "$(uname)" == "Darwin" ]]; then
        brew cask list | grep '^java$' &>/dev/null || brew cask install java
        brew list | grep '^bazel$' &>/dev/null || brew install bazel
    else
        # bazel APT repositories
        echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
        curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

        sudo apt-get update
        dpkg -s 'bazel' &>/dev/null || sudo apt-get install -y bazel
    fi

    # Compiler: Clang
    if [[ "$(uname)" != "Darwin" ]]; then
        sudo apt-get install -y clang
    fi

    # autoenv
    #   on linux, we need sudo to pip install.
    local use_sudo=""
    if [[ "$(uname)" != "Darwin" ]]; then
        use_sudo="sudo -H"
    fi
    pip freeze 2>/dev/null | grep "^autoenv" &>/dev/null \
        || $use_sudo pip install "autoenv" 2>/dev/null

    # LaTeX
    if [[ "$(uname)" == "Darwin" ]]; then
        brew cask list | grep mactex &>/dev/null || brew cask install mactex
    else
        sudo apt-get install -y texlive-full biber
    fi
}
main $@
