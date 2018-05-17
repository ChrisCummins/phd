#!/usr/bin/env bash

# bootstrap.sh - A script to automatically install all project requirements.
#
# The aim of this script is to provide a "hands free" method to install all
# the relevant dependencies for building this project on an Ubuntu or macOS
# host.
#
# Usage:
#
#     ./boostrap.sh
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
        perl -i -p -e 's|git@(.*?):|https://\1/|g' $ROOT/.gitmodules
    fi

    # Ensure that submodules are checked out and set to correct versions.
    git -C $ROOT submodule update --init --recursive

    # mysql_config is required by Python MySQL client.
    if [[ "$(uname)" != "Darwin" ]]; then
        if dpkg -s libmysqlclient-dev &> /dev/null; then
            echo '# libmysql: installed'
        else
            echo '# libmysql:'
            sudo apt-get install -y --no-install-recommends libmysqlclient-dev
        fi
    fi

    # Python 3.6
    echo '# python:'
    $ROOT/system/dotfiles/run -v Python
    # Use the absolute path to Python, since the homebrew installed package
    # may not yet be in the $PATH.
    if [[ "$(uname)" == "Darwin" ]]; then
        PYTHON=/usr/local/opt/python/bin/python3
    else
        PYTHON=/home/linuxbrew/.linuxbrew/bin/python3
    fi
    test -f $PYTHON || { echo 'error: $PYTHON not found!' >&2; exit 1; }

    # Install Python packages.
    $PYTHON -m pip install -r $ROOT/requirements.txt

    # On macOS: Homebrew & coreutils
    if [[ "$(uname)" == "Darwin" ]]; then
        $ROOT/system/dotfiles/run -v GnuCoreutils
    fi

    echo '# bazel:'
    $ROOT/system/dotfiles/run -v Bazel

    # Compiler: Clang
    echo '# clang:'
    $ROOT/system/dotfiles/run -v Clang

    # Jupyter kernel
    if [[ ! -f "$HOME/.ipython/kernels/phd/kernel.json" ]]; then
        rm -rvf $HOME/.ipython/kernels/phd
        mkdir -vp ~/.ipython/kernels
        cp -vr $ROOT/tools/ipython/kernels/phd $HOME/.ipython/kernels/phd
        sed "s,@PYTHON@,$PYTHON," -i $HOME/.ipython/kernels/phd/kernel.json
    fi

    # git pre-commit hook
    if [[ -f "$ROOT/.git/hooks/pre-push" ]]; then
        echo '# git hook: installed'
    else
        echo '# git hook:'
        cp -v $ROOT/tools/pre-push $ROOT/.git/hooks/pre-push
        chmod +x $ROOT/.git/hooks/pre-push
        echo
    fi

    # autoenv
    echo '# autoenv:'
    $ROOT/system/dotfiles/run -v Autoenv

    # LaTeX
    echo '# mactex:'
    $ROOT/system/dotfiles/run -v LaTeX

    # libexempi3 is required by //util/photolib/ and python package
    # python-xmp-toolkit to read XMP metadata from image files.
    if [[ "$(uname)" != "Darwin" ]]; then
        if dpkg -s texlive-full &> /dev/null; then
            echo '# libexempi3: installed'
        else
            echo '# libexempi3:'
            sudo apt-get install -y --no-install-recommends libexempi3
            echo
        fi
    fi

    # Create autoenv environment file. This should be done last, since we can
    # use the presence of the .env to determine if the project has been
    # bootstrapped.
    cp -v $ROOT/tools/env.sh $ROOT/.env
    perl -pi -e 's|__ROOT__|$ROOT|g' $ROOT/.env
}

main $@
