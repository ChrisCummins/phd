#!/usr/bin/env bash
#
# bootstrap.sh - Prepare the toolchain
#
# Usage:
#
#     ./boostrap.sh | bash
#
set -eu

# Directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

main() {
    # header
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "# $0 on macOS"
    else
        echo "# $0 on Ubuntu Linux"
    fi
    echo "# $USER@$(hostname) $(date)"
    echo

    # On macOS: Homebrew & coreutils
    if [[ "$(uname)" == "Darwin" ]]; then
        if which brew &>/dev/null ; then
            echo '# homebrew: installed'
        else
            echo '# homebrew:'
            echo '/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"'
            echo 'brew update'
            echo
        fi

        if brew list | grep '^coreutils$' &>/dev/null ; then
            echo '# coreutils: installed'
        else
            echo '# coreutils:'
            echo 'brew install coreutils'
            echo
        fi
    fi

    # git hook
    if [[ -f "$DIR/../.git/hooks/pre-push" ]]; then
        echo '# git hook: installed'
    else
        echo '# git hook:'
        echo "cp -v $DIR/pre-push $DIR/../.git/hooks/pre-push"
        echo "chmod +x $DIR/../.git/hooks/pre-push"
        echo
    fi

    # Build system: Bazel
    if [[ "$(uname)" == "Darwin" ]]; then
        if brew list | grep '^bazel$' &>/dev/null ; then
            echo '# bazel: installed'
        else
            echo '# bazel:'
            echo "brew cask list | grep '^java$' &>/dev/null || brew cask install java"
            echo 'brew install bazel'
            echo
        fi
    else
        if dpkg -s bazel &>/dev/null ; then
            echo '# bazel: installed'
        else
            echo '# bazel:'
            echo 'echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list'
            echo 'curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -'
            echo 'sudo apt-get update'
            echo 'sudo apt-get install -y bazel'
            echo
        fi
    fi

    # Compiler: Clang
    if [[ "$(uname)" == "Darwin" ]]; then
        echo '# clang: installed (system)'
    else
        if dpkg -s clang &>/dev/null ; then
            echo '# clang: installed'
        else
            echo '# clang:'
            echo 'sudo apt-get install -y clang'
            echo
        fi
    fi

    # Python 3.6
    if [[ "$(uname)" == "Darwin" ]]; then
        if brew list | grep '^python$' &>/dev/null ; then
            echo '# python>=3.6: installed'
        else
            echo '# python>=3.6:'
            echo 'brew install python'
            echo
        fi
    else
        if dpkg -s python3.6 &>/dev/null ; then
            echo '# python3.6: installed'
        else
            echo '# python3.6:'
            echo 'sudo add-apt-repository ppa:jonathonf/python-3.6'
            echo 'sudo apt-get update'
            echo 'sudo apt-get install -y python3.6 python3.6-venv python3.6-dev'
            echo
        fi
    fi

    # Python 3.6 virtualenv
    if [[ ! -f "$DIR/../venv/phd/bin/activate" ]]; then
        echo "virtualenv -p python3.6 $DIR/../venv/phd"
        echo "source $DIR/../venv/phd/bin/activate && python setup.py install"
    fi

    if [[ ! -d "$DIR/../venv/phd/lib/python3.6/site-packages/numpy" ]]; then
        echo "source $DIR/../venv/phd/bin/activate && pip install -r $DIR/../requirements.txt"
    fi

    # Jupyter kernel
    if [[ ! -f "$HOME/.ipython/kernels/phd/kernel.json" ]]; then
        echo "rm -rvf $HOME/.ipython/kernels/phd"
        echo "mkdir -vp ~/.ipython/kernels"
        echo "cp -vr $DIR/ipython/kernels/phd $HOME/.ipython/kernels/phd"
        echo "sed \"s,@PYTHON@,$DIR/../venv/phd/bin/python,\" -i $HOME/.ipython/kernels/phd/kernel.json"
    fi

    # autoenv
    if pip freeze 2>/dev/null | grep '^autoenv' &>/dev/null ; then
        echo '# autoenv: installed'
    else
        echo '# autoenv:'
        if [[ "$(uname)" == "Darwin" ]]; then
            echo 'pip install autoenv'
        else  # we need sudo on linux
            echo 'sudo -H pip install autoenv'
        fi
        echo
    fi

    # LaTeX
    if [[ "$(uname)" == "Darwin" ]]; then
        if brew cask list | grep mactex &>/dev/null ; then
            echo '# mactex: installed'
        else
            echo '# mactex:'
            echo 'brew cask install mactex'
            echo
        fi
    else
        if dpkg -s texlive-full &>/dev/null ; then
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
        if dpkg -s texlive-full &>/dev/null ; then
            echo '# libexempi3: installed'
        else
            echo '# libexempi3:'
            echo 'sudo apt-get install -y libexempi3'
            echo
        fi
    fi
}
main $@
