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


## Building the pre-trained model

The docker image uses a pre-trained CLgen model. To build a model, modify the
"working_dir" and "local_directory" fields of the file
[//experimental/deeplearning/deepsmith/opencl_fuzz/clgen.pbtxt](experimental/deeplearning/deepsmith/opencl_fuzz/clgen.pbtxt)
to point to your local corpus, then train and export the model using:

```
$ bazel run //deeplearning/clgen -- \
    --config ~/phd/experimental/deeplearning/deepsmith/opencl_fuzz/clgen.pbtxt \
    --stop_after train
$ bazel run //deeplearning/clgen -- \
    --config experimental/deeplearning/deepsmith/opencl_fuzz/clgen.pbtxt \
    --export_model ~/phd/experimental/deeplearning/deepsmith/opencl_fuzz/model
```

The script `./make_image.sh` will now use the newly exported model.


## Updating the docker image

```sh
$ cd ~/phd/experimental/deeplearning/deepsmith/opencl_fuzz
$ ./make_image.sh
# Make a note of the last line which prints the image id, something like:
# "Successfully built da07586b1fa5"
$ docker tag $ID chriscummins/opencl_fuzz
$ docker push chriscummins/opencl_fuzz
```
