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

clone() {
    local dest=$1
    test -d $dest || git clone http://llvm.org/git/llvm.git $dest
    cd $dest/tools
    test -d clang || git clone http://llvm.org/git/clang.git
    cd ../projects
    test -d compiler-rt || git clone http://llvm.org/git/compiler-rt.git
}
clone $@
