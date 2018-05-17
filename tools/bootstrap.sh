#!/usr/bin/env bash

# bootstrap.sh - A script to automatically install all project requirements.
#
# The aim of this script is to provide a "hands free" method to install all
# the relevant dependencies for building this project on an Ubuntu or macOS
# host.
#
# Usage:
#
#     ./boostrap.sh | bash
#
set -eu

# Directory of the root of this repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

main() {
    # header
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "# $0 on macOS"
    else
        echo "# $0 on Ubuntu Linux"
    fi
    echo "# $USER@$(hostname) $(date)"
    echo

    # If repository is cloned using https protocol, change the submodule
    # URLs to use https also.
    if  git -C "$ROOT" remote -v | grep -q 'git@' ; then
        echo '# git: cloned from SSH.'
    else
        echo '# git: change to HTTPS submodules'
        echo "perl -i -p -e 's|git@(.*?):|https://\1/|g' $ROOT/.gitmodules"
    fi

    # Ensure that submodules are checked out and set to correct versions.
    echo "git -C $ROOT submodule update --init --recursive"

    # On macOS: Homebrew & coreutils
    if [[ "$(uname)" == "Darwin" ]]; then
        if which brew &> /dev/null; then
            echo '# homebrew: installed'
        else
            echo '# homebrew:'
            echo '/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"'
            echo 'brew update'
            echo
        fi

        if brew list | grep '^coreutils$' &> /dev/null; then
            echo '# coreutils: installed'
        else
            echo '# coreutils:'
            echo 'brew install coreutils'
            echo
        fi
    fi

    # git hook
    if [[ -f "$ROOT/.git/hooks/pre-push" ]]; then
        echo '# git hook: installed'
    else
        echo '# git hook:'
        echo "cp -v $ROOT/tools/pre-push $ROOT/.git/hooks/pre-push"
        echo "chmod +x $ROOT/.git/hooks/pre-push"
        echo
    fi

    echo '# bazel:'
    echo "$ROOT/system/dotfiles/run -v Bazel"

    # Compiler: Clang
    if [[ "$(uname)" == "Darwin" ]]; then
        echo '# clang: installed (system)'
    else
        echo '# clang:'
        echo "$ROOT/system/dotfiles/run -v Clang"
    fi

    # mysql_config is required by Python MySQL client.
    if [[ "$(uname)" != "Darwin" ]]; then
        if dpkg -s libmysqlclient-dev &> /dev/null; then
            echo '# libmysql: installed'
        else
            echo '# libmysql:'
            echo 'sudo apt-get install -y libmysqlclient-dev'
        fi
    fi

    # Python 3.6
    echo '# python:'
    echo "$ROOT/system/dotfiles/run -v Python"
    PYTHON="python3"

    # Install Python packages.
    echo "$PYTHON -m pip install -r $ROOT/requirements.txt"

    # Jupyter kernel
    if [[ ! -f "$HOME/.ipython/kernels/phd/kernel.json" ]]; then
        echo "rm -rvf $HOME/.ipython/kernels/phd"
        echo "mkdir -vp ~/.ipython/kernels"
        echo "cp -vr $ROOT/tools/ipython/kernels/phd $HOME/.ipython/kernels/phd"
        echo "sed \"s,@PYTHON@,$(which $PYTHON),\" -i $HOME/.ipython/kernels/phd/kernel.json"
    fi

    # autoenv
    if pip freeze 2> /dev/null | grep '^autoenv' &> /dev/null; then
        echo '# autoenv: installed'
    else
        echo '# autoenv:'
        if [[ "$(uname)" == "Darwin" ]]; then
            echo 'pip install autoenv'
        else
            # we need sudo on linux
            echo 'sudo -H pip install autoenv'
        fi
        echo
    fi

    # LaTeX
    if [[ "$(uname)" == "Darwin" ]]; then
        if brew cask list | grep mactex &> /dev/null; then
            echo '# mactex: installed'
        else
            echo '# mactex:'
            echo 'brew cask install mactex'
            echo
        fi
    else
        if dpkg -s texlive-full &> /dev/null; then
            echo '# texlive-full: installed'
        else
            echo '# texlive-full:'
            echo 'sudo apt-get install -y texlive-full biber'
            echo
        fi
    fi

    # libexempi3 is required by //util/photolib/ and python package
    # python-xmp-toolkit to read XMP metadata from image files.
    if [[ "$(uname)" != "Darwin" ]]; then
        if dpkg -s texlive-full &> /dev/null; then
            echo '# libexempi3: installed'
        else
            echo '# libexempi3:'
            echo 'sudo apt-get install -y libexempi3'
            echo
        fi
    fi

    # Create autoenv environment file. This should be done last, since we can
    # use the presence of the .env to determine if the project has been
    # bootstrapped.
    echo "cp -v $ROOT/tools/env.sh $ROOT/.env"
    echo "perl -pi -e 's|__ROOT__|$ROOT|g' $ROOT/.env"
}

main $@
