#!/usr/bin/env bash
set -eu

ROOT=$(pwd)

benchmarks=(
     AtomicCounters
     BasicDebug
     BinomialOption
     BinomialOptionMultiGPU
     BitonicSort
     BlackScholesDP
     BlackScholes
     DCT
     DeviceFission11Ext
     DeviceFission
     DwtHaar1D
     DynamicOpenCLDetection
     FastWalshTransform
     FloydWarshall
     # FluidSimulation2D # requires display
     HelloWorld
     HistogramAtomics
     Histogram
     ImageOverlap
     KernelLaunch
     MatrixMulImage
     MatrixMultiplication
     MatrixTranspose
     MemoryModel
     MonteCarloAsian
     MonteCarloAsianMultiGPU
     NBody
     PrefixSum
     QuasiRandomSequence
     RecursiveGaussian
     Reduction
     ScanLargeArrays
     SimpleConvolution
     SimpleImage
     SimpleMultiDevice
     SobelFilter
     StringSearch
     Template
     TransferOverlap
     URNG
)

for f in ${benchmarks[@]}; do
     echo "$(tput bold)amd-app-sdk-3.0 $f$(tput sgr0) ... "

     cd $f/bin/x86_64/Release/
     ./$f
     cd "$ROOT"
done
