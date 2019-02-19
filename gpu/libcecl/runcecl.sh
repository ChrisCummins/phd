#!/usr/bin/env bash
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of libcecl.
#
# libcecl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libcecl.  If not, see <https://www.gnu.org/licenses/>.

# Execute an OpenCL application instrumented with libcecl. If no
# arguments provided, read from stdin.
#
# This file is part of cecl.
#
# cecl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cecl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cecl.  If not, see <http://www.gnu.org/licenses/>.
#
set -eu

main() {
    # Propagate return codes of pipeline.
    set -o pipefail

    if [[ $# > 0 ]]; then
        stdbuf -oL -eL 2>&1 > /dev/null | grep -E '^\[CECL\]' | sed 's/^\[CECL\] //'
    else
        stdbuf -oL -eL less <&0 | grep -E '^\[CECL\]' | sed 's/^\[CECL\] //'
    fi
}

main $@
