# cldrive - Run arbitrary OpenCL kernels

<a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
  <img src="https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat">
</a>


## Prerequisites

Install [Bazel](https://docs.bazel.build/versions/master/install.html).

Build using:

```sh
$ bazel build //gpu/cldrive
```

## Usage

```sh
$ bazel run //gpu/cldrive -- --srcs=<opencl_sources> --envs=<opencl_devices>
```

Where `<opencl_sources>` if a comma separated list of absolute paths to OpenCL
source files, and `<opencl_devices>` is a comma separated list of
fully-qualified OpenCL device names. To list the available device names use
`--clinfo`. Use `--help` to see the full list of options.

### Example

For example, given a file:

```sh
$ cat kernel.cl
kernel void my_kernel(global int* a, global int* b) {
    int tid = get_global_id(0);
    a[tid] += 1;
    b[tid] = a[tid] * 2;
}
```

and available OpenCL devices:

```sh
$ bazel run //gpu/cldrive -- --clinfo
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0
```

To run the kernel 5 times on both devices using 4096 work items divided into
work groups of size 1024:

```sh
$ bazel run //gpu/cldrive -- --srcs=$PWD/kernel.cl --num_runs=5 \
    --gsize=4096 --lsize=1024 \
    --envs='GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2','CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0'
OpenCL Device, Kernel Name, Global Size, Local Size, Transferred Bytes, Runtime (ns)
I 2019-02-26 09:54:10 [gpu/cldrive/libcldrive.cc:59] clBuildProgram() with options '-cl-kernel-arg-info' completed in 1851 ms
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 113344
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 57984
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 64096
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 73696
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 73632
I 2019-02-26 09:54:11 [gpu/cldrive/libcldrive.cc:59] clBuildProgram() with options '-cl-kernel-arg-info' completed in 76 ms
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 105440
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 55936
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 63296
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 56192
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 55680
```

By default, cldrive prints a CSV summary of kernel stats and runtimes to
stdout, and logging information to stderr. The raw information produced by
cldrive is described in a set of protocol buffers
[//gpu/cldrive/proto:cldrive.proto](/gpu/cldrive/proto/cldrive.proto). To print
`cldrive.Instances` protos to stdout, use argumet `--output_format=pbtxt`
to print text format protos, or `--output_format=pb` for binary format.


## License

Copyright 2016, 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See
[LICENSE](/gpu/cldrive/LICENSE) for details.
