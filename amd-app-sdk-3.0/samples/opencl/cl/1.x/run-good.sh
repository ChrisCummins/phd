#!/usr/bin/env bash
set -u

ROOT=$(pwd)

benchmarks=(
     AdvancedConvolution
     BinomialOption
     BitonicSort
     BlackScholes
     FastWalshTransform
     FloydWarshall
     Histogram
     MatrixMultiplication
     MatrixTranspose
     MonteCarloAsian
     NBody
     PrefixSum
     Reduction
     ScanLargeArrays
     SimpleConvolution
     SobelFilter
)

for f in ${benchmarks[@]}; do
     echo "$(tput bold)amd-app-sdk-3.0 $f$(tput sgr0) ... "

     cd $f/bin/x86_64/Release/
     ./$f
     cd "$ROOT"
done
