# Compiler Fuzzing through Deep Learning: Artifact

This directory contains the supporting artifact for our paper on compiler
fuzzing through deep learning. It contains reduced-size datasets for evaluting
our testcase generator, testcase harness, and differential tester. The full
dataset is quite large (>200 GB uncompressed), and we are working on finding a
method for sharing it with the community. The idea is that this directory
contains minimal working examples which can be evaluated in a reasonable amount
of time. All of our code is open source and has been developed with
extensibility as a primary goal. Please see Section 4 of this document for
more details.


## 1. Artifact Contents

 * `01_evaluate_generator/` A demonstration of training a DeepSmith model to
   generate programs.

 * `02_evaluate_harness/` A demonstration which executes test cases on a local
   OpenCL test bed which we found to cause crashes/bugs in at least one device.

 * `03_evaluate_results/` A demonstration of our differential testing approach.
   The results from the prior demonstration are combined with our own data from
   the paper to perform differential testing.

 * `diagnose_me.sh` A script which reports various diagnostic information about
 your system.


## 2. Installation

There are two options for evaluating our artifact: either using a
self-contained Docker image or by building from source. The installation
process for both is described below.


### 2.1. Docker container

Our [Docker](https://docs.docker.com/install/) image must be run on a host with
an Intel CPU.

Download our Dockerfile:

```sh
$ wget -ODockerfile "https://raw.githubusercontent.com/ChrisCummins/phd/919a535e1c4ddc0a9aea3fa610550e14423fea74/docs/2018_07_issta/artifact_evaluation/Dockerfile"
```

Build the docker container using:

```sh
$ sudo docker build -t deepsmith .
```

Launch an interactive shell in the docker container using:

```sh
$ sudo docker run -it deepsmith /bin/zsh
```

The docker image contains everything you need to evaluate our artifact
pre-installed, including an Intel OpenCL driver. Should you wish, you may
install drivers for additional OpenCL devices you have available on your system.


### 2.2. Build from source

Building from source is supported only on Ubuntu Linux or macOS, since
unfortunately Windows is not supported by many of our dependencies. Other Linux
distributions may work, but we have not tested them, and our install.sh script
uses the apt package manager to automate the installation of depdencies. If you
have requirements for a specific Linux distribution that is not Ubuntu >=
16.04, please contact us.

We use OpenCL as the programming language to demonstrate our approach. We
require an OpenCL compiler to test. The installation process depends on what
hardware you have available on your system. On macOS, OpenCL support is built
in. OpenCL drivers on Linux are typically distributed as part of GPU driver
packages or as standalone SDKs. Please refer to your manufacturers
documentation (e.g. [Intel](https://software.intel.com/en-us/intel-opencl),
[NVIDIA](http://www.nvidia.com/drivers),
[AMD](https://www.amd.com/en-us/solutions/professional/hpc/opencl)).

If you have not already, please clone this git repository and change to this
directory:

```sh
$ git clone https://github.com/ChrisCummins/phd.git
$ cd phd/docs/2018_07_issta/artifact_evaluation
```

Install local and system-wide dependencies using the command:

```sh
$ ./install.sh
```

The entire installation process is automated. Please note that it may take
some time. In case of a problem at this stage, please run the ./diagnose_me.sh
command and send the output to us.


## 3. Evaluating the artifact


### 3.1. Evaluate the generator
*(approximate runtime: 2.5 hours)*

Evaluate the DeepSmith generator by running the following script:

```
$ ./01_evaluate_generator/run.sh
```

The script downloads a small OpenCL corpus of 1000 kernels randomly selected
from our GitHub corpus, pre-processes the corpus, trains a model on the
preprocessed corpus, and generates 1000 new OpenCL programs.

We have reduced the size of the corpus and network so that it takes around 2
hours to train on a CPU. The model we used to generate the programs used in the
review copy of our paper is much larger, is trained on more data, and is trained
for longer. It takes around 2 days to train on a 16-core CPU.

Once the script has completed, the generated programs are written to the
directory `./01_evaluate_generator/output/generated_kernels`. Testcases are
generated for each of the kernels, found in
`./01_evaluate_generator/output/generated_testcases`.

#### 3.1.1. Extended Evaluation (optional)

By changing the parameters of the model defined in the
`./01_evaluate_generator/run.sh` script, you could evaluate the output of models
with different architectures, trained for different amounts of time, etc.

You could also change the corpus by adding additional OpenCL files, removing
files and training on only a subset, etc.


### 3.2. Evaluate the harness
*(approximate runtime: 30 minutes)*


Evaluate the DeepSmith harness by running a set of test cases on your local
OpenCL driver. Use our self-diagnostic script `./diagnose_me.sh` to print a list
of the available OpenCL devices on this system. For example:

```sh
$ ./diagnose_me.sh
# ...
Available OpenCL devices:
  1:  Device: Intel(R) Core(TM) i5-3570K CPU @ 3.40GHz, Platform: Apple
  2:  Device: GeForce GTX 970, Platform: Apple
```

Each item in the list is prefixed by a number. Pass this number, and either `+`
or `-` (for OpenCL optimizations on or off, respectively), to the
experiment script:

```sh
$ ./02_evaluate_harness/run.sh  <number±>
```

For example, to run the experiments on the GeForce GPU device listed above with
optimizations enabled, you would run `$ ./02_evaluate_harness/run.sh 2+`.

The script runs 45 test cases taken from the experimental data we used in the
paper, located in the `data/testcases` directory. You are free to run it on as
many different OpenCL devices as you have (per-device results will be stored).

The results of execution are written to the directories
`./02_evaluate_harnesses/output/results/<number±>/`. Each file in these
directories stores the result of a single test case execution.


#### 3.2.1. Extended Evaluation (optional)

In addition to running the test cases on multiple OpenCL devices, you could add
more test cases to execute. For example, you could run the test cases generated
by your model from the previous step:

```sh
$ rsync -avh \
    ./01_evaluate_generator/output/generated_testcases/ \
    ./02_evaluate_harness/data/testcases/
$ ./02_evaluate_harness/run.sh
```

### 3.3. Evaluate the results (approximate runtime: 5 minutes)

Evaluate the DeepSmith difftester by running the following script:

```sh
$ ./03_evaluate_results/run.sh
```

The script evaluates the results generated from your local system in the
previous experiment, and differential tests the outputs against the data we
used in the paper. At the end of execution, the script prints a table of
classifications, using the same format as Table 2 of the paper. Individually
classified results are available in
`./03_evaluate_results/output/classifications/<class>/<number>/<±>/`.


#### 3.3.1. Extended Evaluation (optional)

The evaluation script difftests all results files from these directories:

```sh
./02_evaluate_harness/output/results  # Results from your system
./03_evaluate_harness/data/results    # Results from our machines
```

You could add new results to this directory by running the previous step on 
multiple OpenCL devices. Once a test case has results from three or more
devices, it will be difftested. Alternatively you could modify individual
results files, such as by changing the returncode, to simulate different
test case outcomes, and observe how that influences the classification of
results.


## 4. Further Reading and References

DeepSmith is currently undergoing a ground-up rewrite to support more languages
and add new features, such as a distributing tests across multiple machines.
The work-in-progress implementation can be found in:

    https://github.com/ChrisCummins/phd/tree/master/deeplearning/deepsmith

The code we used for generating the data in the review copy of our paper is
available at:

    https://github.com/ChrisCummins/phd/tree/master/experimental/dsmith

## 5. License

Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See
[LICENSE](/docs/2018_07_issta/artifact_evaluation/LICENSE) for details.
