================================================
Compiler Fuzzing through Deep Learning: Artifact
================================================

This directory contains the supporting artifact for our paper on compiler
fuzzing through deep learning. It contains reduced-size datasets for evaluting
our testcase generator, testcase harness, and differential tester. The full
dataset is quite large (>200 GB uncompressed), and we are working on finding a
method for sharing it with the community. The idea is that this directory
contains minimal working examples which can be evaluated in a reasonable amount
of time. All of our code is open source and has been developed with
extensibility as a primary goal. Please see Section 4 of this document for
more details.


1. Artifact Contents
====================

  install.sh
    Our automated installation process.

  01_evaluate_generator/
    A demonstration of training a DeepSmith model to generate programs.

  02_evaluate_harness/
    A demonstration which executes test cases on a local OpenCL test bed which
    we found to cause crashes/bugs in at least one device.

  03_evaluate_results/
    A demonstration of our differential testing approach. The results from the
    prior demonstration are combined with our own data from the paper to
    perform differential testing.

  diagnose_me.sh
    A script which reports various diagnostic information about your system.


2. Requirements
===============

2.1 Operating system
--------------------

Ubuntu Linux or macOS. Unfortunately Windows is not supported by many of our
dependencies. Other Linux distributions may work, but we have not tested them,
and our install.sh script uses the apt package manager to automate the
installation of depdencies. If you have requirements for a specific Linux
distribution that is not Ubuntu >= 16.04, please contact us.


2.2 Disk space
--------------

We have not optimized our artifact for space, requiring approximately 6GB of
disk space in total. Please contact us if this is an issue.


2.3 OpenCL
----------

We use OpenCL as the programming language to demonstrate our approach. We
require an OpenCL compiler to test. The installation process depends on what
hardware you have available on your system. OpenCL drivers are typically
distributed as part of GPU driver packages or as standalone SDKs. Please refer
to your manufactures documentation:

  Intel   https://software.intel.com/en-us/intel-opencl
  NVIDIA  http://www.nvidia.com/drivers
  AMD     https://www.amd.com/en-us/solutions/professional/hpc/opencl

On macOS, OpenCL support is built in.


3. Instructions
===============

3.1. Installation (approximate runtime: 1 hour).
-----------------

If you have not already, please clone this git repository and change to this
directory:

    $ git clone https://github.com/ChrisCummins/phd.git
    $ cd phd/docs/2018_07_issta/artifact_evaluation

Install local and system-wide dependencies using the command:

    $ ./install.sh

The entire installation process is automated. Please note that it may take
some time. In case of a problem at this stage, please run the ./diagnose_me.sh
command and send the output to us.


3.2. Evaluate the generator (approximate runtime: 2.5 hours)
---------------------------

Evaluate the DeepSmith generator by running the following script:

    $ ./01_evaluate_generator/run.sh

The script downloads a small OpenCL corpus of 1000 kernels randomly selected
from our GitHub corpus, pre-processes the corpus, trains a model on the
preprocessed corpus, and generates 1000 new OpenCL programs.

We have reduced the size of the corpus and network so that it takes around 2
hours to train on a CPU. The model we used to generate the programs used in the
review copy of our paper is much larger, is trained on more data, and is trained
for longer. It takes around 2 days to train on a CPU.

Once the script has completed, the generated programs are written to the
directory ./01_evaluate_generator/output/generated_kernels. Testcases are
generated for the kernels, found in
./01_evaluate_generator/output/generated_testcases.

3.2.1. Extended Evaluation (optional)

By changing the parameters of the model defined in the
./01_evaluate_generator/run.sh script, you could evaluate the output of models
with different architectures, trained for different amounts of time, etc.

You could also change the corpus by adding additional OpenCL files, removing
files and training on only a subset, etc.


3.3. Evaluate the harness (approximate runtime: 30 minutes)
-------------------------

Evaluate the DeepSmith harness by running a set of testcases on your local
OpenCL driver. First, run our self-diagnostic script:

    $ ./diagnose_me.sh

At the end of the script output is a list of the available OpenCL devices on
this system. For example:

    Available OpenCL devices:
        1:  Device: Intel(R) Core(TM) i5-3570K CPU @ 3.40GHz, Platform: Apple
        2:  Device: GeForce GTX 970, Platform: Apple

Each item in the list is prefixed by a number. Pass this number, and either +
or - (for OpenCL optimizations on or off, respectively), to the
experiment script:

    $ ./02_evaluate_harness/run.sh  <DEVICE_ID>

For example, to run the experiments on the Intel device listed above with
optimizations enabled, you would run $ ./02_evaluate_harness/run.sh 2+.

The script runs 45 testcases taken from the experimental data we used in the
paper, located in the data/testcases directory. You are free to run it on as
many different OpenCL devices as you have (per-device results will be stored).

The results of execution are written to the directories
./02_evaluate_harnesses/output/results/<device_id>/. Each file in these
directories stores the result of a single testcase execution. Open the files

3.3.3. Extended Evaluation (optional)

In addition to running the testcases on multiple OpenCL devices, you could add
more testcases to execute. For example, you could run the testcases generated
by your model from the previous step:

    $ rsync -avh ./01_evaluate_generator/output/generated_testcases/ \
        ./02_evaluate_harness/data/testcases/
    $ ./02_evaluate_harness/run.sh


3.4. Evaluate the results (approximate runtime: 5 minutes)
-------------------------

Evaluate the DeepSmith difftester by running the following script:

    $ ./03_evaluate_results/run.sh

The script evaluates the results generated from your local system in the
previous experiment, and differential tests the outputs against the data we
used in the paper. At the end of execution, the script prints a table of
classifications, using the same format as Table 2 of the paper. Individua
classified results are available in
./03_evaluate_results/output/classifications/<class>/<device>/.


4. Further Reading and References
=================================

DeepSmith is currently undergoing a ground-up rewrite to support more languages
and add new features, such as a distributing tests across multiple machines.
The work-in-progress implementation can be found in:

    https://github.com/ChrisCummins/phd/tree/master/deeplearning/deepsmith

The code we used for generating the data in the review copy of our paper is
available at:

    https://github.com/ChrisCummins/phd/tree/master/experimental/dsmith

We extended our neural network program generator CLgen for generating the
tests used in the review copy of our paper. The documentation for CLgen is
available at:

    http://chriscummins.cc/clgen
