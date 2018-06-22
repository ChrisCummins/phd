# OpenCL fuzzing through DeepSmith


## Installation

Pull the docker image using:

```sh
$ docker pull chriscummins/opencl_fuzz
```

Alternatively you can build the docker image from source using:

```sh
$ ./experimental/deeplearning/deepsmith/opencl_fuzz/make_image.sh
```

## Usage

To test an OpenCL device on the host, identify it's name using `//gpu/cldrive`.
Let's assume we want to test this Intel OpenCL driver:

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

Copy the ICD file for the driver into a new directory which we will map into the
docker image:

```sh
$ mkdir vendors
$ cp /etc/OpenCL/vendors/intel64.icd vendors
```

Find out where the driver is installed, so that we can map this directory to the
docker image too:

```
$ cat vendors/intel64.icd
/opt/intel/opencl-1.2-6.4.0.25/lib64/libintelocl.so
```

The docker image creates
[Result protos](/deeplearning/deepsmith/proto/deepsmith.proto) for testcases
it finds interesting in `/out`. We should map that directory to the host
machine so that we can get the results out.

Finally, run the docker image, mapping the three directories to the host, and
using the `--dut` flag to pass the name of the driver we identified from
`//gpu/cldrive`:

```sh
$ docker run \
    -v$PWD:/out \
    -v$PWD/vendors:/etc/OpenCL/vendors \
    -v/opt/intel:/opt/intel \
    chriscummins/opencl_fuzz \
    --dut 'CPU|Intel(R)_OpenCL|Intel(R)_Xeon(R)_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0'
```

By default, the image runs until `--min_interesting_results` interesting results
have been found, or until `--max_testing_time_seconds` have elapsed. Pass values
for these flags to override the default values.
