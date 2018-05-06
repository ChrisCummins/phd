#!/bin/sh

set -eux

tmp=$1
shift

"$@" <&0 | gcc -x c - -o $tmp -lOpenCL
$tmp
