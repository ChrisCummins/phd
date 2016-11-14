Title:      Autotuning OpenCL Workgroup Sizes
Authors:    Chris Cummins, Pavlos Petoumenos, Michel Steuwer, Hugh Leather
Presented:  ACACES 2016 - Twelfth International Summer School on
            Advanced Computer Architecture and Compilation for
            High-Performance and Embedded Systems. Fiuggi, Italy,
            Wednesday, July 13th 2016.

Abstract:

Selecting appropriate workgroup sizes for OpenCL is critical for
program performance, and requires knowledge of the underlying
hardware, the data being operated on, and the kernel
implementation. We propose the use of machine learning-enabled
autotuning to automatically predict workgroup sizes for stencil
patterns on CPUs and multi-GPUs, using the Algorithmic Skeleton
library SkelCL. In an evaluation across 429 combinations of
architecture, kernel, and dataset, we find that static tuning of
workgroup size achieves only 26% of the optimal performance. Using
machine learning and synthetically generated stencil programs, we
achieve 92% of this maximum, demonstrating a median 3.79x speedup over
the best possible fixed workgroup size.
