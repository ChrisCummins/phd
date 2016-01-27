Title:      Autotuning Stencils Codes with Algorithmic Skeletons
Authors:    Chris Cummins
Thesis:     Master of Science by Research, Institute of Computing
            Systems Architecture, School of Informatics,
            University of Edinburgh. 2015.

Abstract:

The physical limitations of microprocessor design have forced the
industry towards increasingly heterogeneous architectures to extract
performance. This trend has not been matched with software tools to
cope with such parallelism, leading to a growing disparity between the
levels of available performance and the ability for application
developers to exploit it.

Algorithmic skeletons simplify parallel programming by providing
high-level, reusable patterns of computation. Achieving performant
skeleton implementations is a difficult task; developers must attempt
to anticipate and tune for a wide range of architectures and use
cases. This results in implementations that target the general case
and cannot provide the performance advantages that are gained from
tuning low level optimisation parameters.

To address this, I present OmniTune --- an extensible and distributed
framework for runtime autotuning of optimisation parameters. Targeting
the workgroup size of OpenCL kernels, I demonstrate an implementation
of OmniTune for stencil codes on CPUs and multi-GPU systems. I show in
a comprehensive evaluation of 2.7 x 10^5 test cases that simple
heuristics cannot provide portable performance across the range of
architectures, kernels, and datasets which algorithmic skeletons must
target.

OmniTune uses procedurally generated synthetic benchmarks and machine
learning to predict workgroup sizes for unseen programs. In an
evaluation of 429 combinations of programs, architectures, and
datasets, with up to 7.3 x 10^3 parameter values for each, OmniTune is
able to achieve a median 94% of the available performance, providing a
1.33x speedup over the values selected by human experts, without
requiring any user intervention. This adaptive tuning provides a
median speedup of 3.79x (max 74.0x) over the best possible performance
which can be achieved without autotuning.
