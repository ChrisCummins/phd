# DeepSmith: Compiler Fuzzing through Deep Learning

DeepSmith is a novel approach to automate and accelerate compiler validation
which takes advantage of state-of-the-art deep learning techniques. It
constructs a learned model of the structure of real world code based on a large
corpus of example programs. Then, it uses the model to automatically generate
tens of thousands of realistic programs. Finally, it applies established
differential testing methodologies on them to expose bugs in compilers.

## Implementation Design

DeepSmith uses a service architecture. There are three types of services:

 * **DataStore Service** servers as the backend, storing generated testcases,
   results from executing testcases, and analysis of testcase results.
 * **Generator Service** generates testcases.
 * **Harness Service** coordinates the execution of testcases on one or more
   testbeds.

## Usage

### Installation

First checkout this repo and install the build requirements:

```sh
$ git clone https://github.com/ChrisCummins/phd.git
$ cd phd
$ ./tools/bootstrap.sh
```

Then run the unit tests using:

```sh

$ bazel test //deeplearning/deepsmith/...
```

### Example: Testing OpenCL compilers using CLgen

#### Launching Services

First launch a datastore service using:

```sh
$ bazel build //deeplearning/deepsmith/services:datastore && \
    ./bazel-phd/bazel-out/*/bin/deeplearning/deepsmith/services/datastore \
    --datastore_config=$PHD/deeplearning/deepsmith/proto/datastore_sqlite.pbtxt
```

Launch a generator for OpenCL testcases using CLgen with:

```sh
$ bazel build //deeplearning/deepsmith/services:clgen && \
    ./bazel-phd/bazel-out/*/bin/deeplearning/deepsmith/services/clgen \
    --generator_config=$PHD/deeplearning/deepsmith/proto/generator_opencl_clgen.pbtxt
```

Launch an OpenCL harness service for executing OpenCL testcases using cldrive 
with:

```sh
$ bazel build //deeplearning/deepsmith/services:cldrive && \
    ./bazel-phd/bazel-out/*/bin/deeplearning/deepsmith/services/cldrive \
    --harness_config=$PHD/deeplearning/deepsmith/proto/harness_opencl_cldrive.pbtxt
```

#### Running Experiments

Generate 1000 testcases using the CLgen generator service with:

```sh
$ bazel run //deeplearning/deepsmith/cli:generate_testcases -- \
    --datastore_config=$PHD/deeplearning/deepsmith/proto/datastore_sqlite.pbtxt \
    --generator_config=$PHD/deeplearning/deepsmith/proto/generator_opencl_clgen.pbtxt \
    --target_total_testcases=1000
```

Execute the OpenCL testcases with:

```sh
$ bazel run //deeplearning/deepsmith/cli:run_testcases -- \
      --datastore_config=$PHD/deeplearning/deepsmith/proto/datastore_sqlite.pbtxt \
      --harness_config=$PHD/deeplearning/deepsmith/proto/harness_opencl_cldrive.pbtxt
```


## License

Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE](/LICENSE) for
details.
