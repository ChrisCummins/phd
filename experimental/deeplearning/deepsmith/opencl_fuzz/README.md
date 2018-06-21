# OpenCL fuzzing through DeepSmith


## Usage

Build the binary:

```sh
$ bazel build //experimental/deeplearning/deepsmith/opencl_fuzz
```

Identify the name of an OpenCL device to test:

```sh
$ bazel run //gpu/cldrive -- --ls_env
CPU|Intel(R)_OpenCL|Intel(R)_Xeon(R)_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0
    Platform:     Intel(R) OpenCL
    Device:       Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
    Driver:       1.2.0.25
    Device Type:  CPU
    OpenCL:       2.0
...
```

Export the name for convenience:

```sh
$ export DEVICE_NAME="CPU|Intel(R)_OpenCL|Intel(R)_Xeon(R)_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0"
```

Test the device using:

```sh
$ bazel-bin/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz \
    --batch_size 10 --max_testing_time_seconds 60 \
    --generator $PWD/experimental/deeplearning/deepsmith/opencl_fuzz/generator.txt \
    --dut "$DEVICE_NAME"
```

## Usage: Docker Image

Build the docker image:

```sh
$ ./experimental/deeplearning/deepsmith/opencl_fuzz/make_image.sh
```

You can test an OpenCL device on the host by mapping the host's ICD files and
the device installation directory to the image. E.g., assuming you have
set `$DEVICE_NAME` and `$ICD` to the name of the device and the path to the ICD
file respectively:

```sh
$ docker run \
    -v/etc/OpenCL/vendors:/etc/OpenCL/vendors \
    -v$(dirname $(cat /etc/OpenCL/vendors/$ICD)):$(dirname $(cat /etc/OpenCL/vendors/$ICD)) \
    -v$PWD/opencl_fuzz:/interesting_results opencl_fuzz \
    /app/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image.binary \
    --generator /datasets/generator.pbtxt \
    --interesting_results_dir=/interesting_results \
    --dut "$DEVICE_NAME"
```
