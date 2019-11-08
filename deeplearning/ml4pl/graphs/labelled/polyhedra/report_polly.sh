#!/bin/sh

CC=clang-6.0
OPT=opt-6.0

P=$1
filename=$(basename -- "$P")
extension="${filename##*.}"
filename="${filename%.*}"

# polybench flag: -I ~/polybench-c-4.2/utilities
$CC -O1 -emit-llvm -S -g $*

if [ $? -ne 0 ]; then
    echo "FAILED TO COMPILE $1"
    exit 1
fi


run() {
    echo "==== $* ===="
    rm -f scops*.dot
    
    $OPT -polly-process-unprofitable -polly-optimized-scops -polly-dot $* ${filename}.ll -S -o ${filename}.polly.ll

    grep "style = filled" scops*.dot
}

runopt() {
    #run -O1 $*
    #run -O2 $*
    run -O3 $*
}

runopt 
runopt -polly-optimizer=none
runopt -polly-tiling
runopt -polly-parallel
