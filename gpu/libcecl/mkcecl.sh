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

# Rewrite OpenCL sources to use libcecl.
#
# This file is part of libcecl.
#
# libcecl is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cecl.  If not, see <http://www.gnu.org/licenses/>.
#
set -eu

usage() {
    echo "$0 <source-files ...>"
    echo
    echo "Rewrite OpenCL calls to use libcecl."
}

file_is_source() {
    local path="$1"

    if [[ "$path" == *.cpp ]]; then
        return 0
    elif [[ "$path" == *.cc ]]; then
        return 0
    elif [[ "$path" == *.cxx ]]; then
        return 0
    elif [[ "$path" == *.c ]]; then
        return 0
    elif [[ "$path" == *.h ]]; then
        return 0
    elif [[ "$path" == *.hpp ]]; then
        return 0
    else
        return 1
    fi
}

rewritefile() {
    local path="$1"

    local tmp="$path.tmp"
    local backup="$path.bkp"
    cp "$path" "$tmp"
    cp "$path" "$backup"

    local start_checksum="$(md5sum "$path" | awk '{print $1}')"

    sed -i 's/clBuildProgram/CECL_PROGRAM/g' "$tmp"
    sed -i 's/clCreateBuffer/CECL_BUFFER/g' "$tmp"
    sed -i 's/clCreateCommandQueue/CECL_CREATE_COMMAND_QUEUE/g' "$tmp"
    sed -i 's/clCreateKernel/CECL_KERNEL/g' "$tmp"
    sed -i 's/clCreateProgramWithSource/CECL_PROGRAM_WITH_SOURCE/g' "$tmp"
    sed -i 's/clEnqueueMapBuffer/CECL_MAP_BUFFER/g' "$tmp"
    sed -i 's/clEnqueueNDRangeKernel/CECL_ND_RANGE_KERNEL/g' "$tmp"
    sed -i 's/clEnqueueReadBuffer/CECL_READ_BUFFER/g' "$tmp"
    sed -i 's/clEnqueueTask/CECL_TASK/g' "$tmp"
    sed -i 's/clEnqueueWriteBuffer/CECL_WRITE_BUFFER/g' "$tmp"
    sed -i 's/clSetKernelArg/CECL_SET_KERNEL_ARG/g' "$tmp"
    sed -i 's/clCreateContextFromType/CECL_CREATE_CONTEXT_FROM_TYPE/g' "$tmp"
    sed -i 's/clCreateContext/CECL_CREATE_CONTEXT/g' "$tmp"
    sed -i 's/clGetKernelWorkGroupInfo/CECL_GET_KERNEL_WORK_GROUP_INFO/g' "$tmp"

    local end_checksum=$(md5sum "$tmp" | awk '{print $1}')

    if [[ "$start_checksum" == "$end_checksum" ]]; then
        rm "$tmp"
    else
        if ! grep '#include <libcecl.h>' "$tmp" &> /dev/null; then
            echo '#include <libcecl.h>' >"$path"
            cat "$tmp" >>"$path"
            rm "$tmp"
        else
            mv "$tmp" "$path"
        fi
    fi
}

main() {
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 1
    fi

    for arg in "$@"
    do
        if file_is_source "$arg" &> /dev/null; then
            echo "$arg"
            rewritefile "$arg"
        else
            echo "ignored: $arg" >&2
        fi
    done
}

main $@
