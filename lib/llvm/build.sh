#!/usr/bin/env bash
#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
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
set -eux

build() {
    local srcdir=$(readlink -f $1)
    local builddir=$2
    local cmake=${3:-cmake}
    local ninja=${4:-ninja}

    mkdir -pv $builddir
    cd $builddir
    $cmake $srcdir -G Ninja -DLLVM_TARGETS_TO_BUILD="X86"
    $ninja
}
build $@
