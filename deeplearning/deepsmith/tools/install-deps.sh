#!/usr/bin/env bash
#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
set -eu

usage() {
    echo "usage: $0 [| bash]"
}

main() {
    set +u
    if [[ -n "$1" ]]; then
        usage >&2
        exit 1
    fi
    set -u

    if python -mplatform | grep -qi Ubuntu ; then
        echo '# guessed distribution: Ubuntu'

        local ubuntu_version="$(lsb_release -r | sed -r 's/^Release:\s+//')"
        if [[ "$ubuntu_version" != "16.04" ]]; then
            echo "# warning: Ubuntu $ubuntu_version is unsupported"
        fi

        if dpkg -s python3.6 &>/dev/null ; then
            echo '# python3.6: installed'
        else
            echo '# install python3.6:'
            echo 'sudo add-apt-repository ppa:jonathonf/python-3.6'
            echo 'sudo apt-get update'
            echo 'sudo apt-get install -y python3.6 python3.6-venv python3.6-dev'
        fi

        if dpkg -s libhdf5-dev &>/dev/null ; then
            echo '# libhdf5-dev: installed'
        else
            echo '# install libhdf5-dev:'
            echo 'sudo apt-get install -y libhdf5-dev'
        fi

        if dpkg -s clang &>/dev/null ; then
            echo '# clang: installed'
        else
            echo '#install clang:'
            echo 'sudo apt-get install -y clang'
        fi

    elif python -mplatform | grep -qi CentOS ; then
        echo '# guessed distribution: CentOS'

        echo '#install python3.6:'
        echo 'sudo yum -y update'
        echo 'sudo yum -y install yum-utils'
        echo 'sudo yum -y groupinstall development'
        echo 'sudo yum -y install https://centos7.iuscommunity.org/ius-release.rpm'
        echo 'sudo yum -y install python36u'

        echo '# install libhdf5'
        echo 'sudo yum -y install hdf5 hdf5-devel'

        echo '#install clang:'
        echo 'sudo yum -y clang'
    elif python -mplatform | grep -qi Arch ; then
        echo '# guessed distribution: Arch Linux'

        echo '# install python3.6:'
        echo 'sudo pacman -S python'

        echo '# install libhdf5'
        echo 'sudo pacman -S yaourt'
        echo 'sudo yaourt -Syy'
        echo 'sudo yaourt -S libhdf5'

        echo '# install clang:'
        echo 'sudo pacman -S clang'
    else
        echo '# warning: Unsupported OS detected!'
        echo '# Please ensure you have python3.6, libhdf5, and clang installed'
    fi
}
main $@
