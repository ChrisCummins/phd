# GPGPU Benchmarks

This package contains seven benchmark suites.

#### Instrumentating Benchmarks to un with libcecl

1. Modify Makefile to add `cec-profile.o` to links.
1. Add `#include <cec-profile.h>` to source files.
1. Replace `clCreateCommandQueue()` with `CEC_COMMAND_QUEUE()`.
1. Replace `clEnqueueNDRangeKernel()` with `CEC_ND_KERNEL()`, and remove last 
   argument (event*).
1. Replace `clEnqueueReadBuffer()` with `CEC_READ_BUFFER()`.
