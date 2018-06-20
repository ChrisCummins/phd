# OpenCL fuzzing through DeepSmith


## Usage

Build the binary:

```sh
$ bazel build //experimental/deeplearning/deepsmith/opencl_fuzz
```

Test a device using:

```sh
$ bazel-bin/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz \
    --batch_size 10 --max_testing_time_seconds 60 \
    --generator $PWD/experimental/deeplearning/deepsmith/opencl_fuzz/generator.txt \
    --dut 'CPU|ComputeAorta|Codeplay_Software_Ltd._-_host_CPU|1.14|1.2'
```

## Usage: Docker Image

Build the docker image:

```sh
$ ./experimental/deeplearning/deepsmith/opencl_fuzz/make_image.sh
```

Test a device using:

```sh
$ docker run \
    -v/etc/OpenCL/vendors:/etc/OpenCL/vendors \
    -v/opt/ComputeAorta:/opt/ComputeAorta \
    -v$PWD/opencl_fuzz:/interesting_results opencl_fuzz \
    /app/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image.binary \
    --generator /datasets/generator.pbtxt \
    --interesting_results_dir=/interesting_results \
    --dut 'CPU|ComputeAorta|Codeplay_Software_Ltd._-_host_CPU|1.14|1.2'
```

Note the use of `-v` arguments to map various directories  interesting results
are exported to the host.
