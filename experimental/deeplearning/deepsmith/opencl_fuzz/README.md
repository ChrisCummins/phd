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
$ docker run opencl_fuzz \
    /app/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image.binary \
    --generator /datasets/generator.pbtxt
```
