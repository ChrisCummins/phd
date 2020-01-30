# Compiler Fuzzing through Deep Learning: Artifact

This directory contains the supporting artifact for our paper on compiler
fuzzing through deep learning. It contains reduced-size data sets for evaluating
our testcase generator, testcase harness, and differential tester. The full
dataset is quite large (>200 GB uncompressed), and we are working on finding a
method for sharing it with the community. The idea is that this directory
contains minimal working examples which can be evaluated in a reasonable amount
of time. All of our code is open source and has been developed with
extensibility as a primary goal. Please see Section 4 of this document for
more details.


## 1. Artifact Contents

 * [01_evaluate_generator](01_evaluate_generator.py): A demonstration of
   training a neural network model to generate programs.

 * [02_evaluate_harness](02_evaluate_harness.py): A demonstration which executes
   test cases on an OpenCL test bed.

 * [03_evaluate_results](03_evaluate_results.py): A demonstration of our
   differential testing approach. The results from the prior demonstration are
   combined with our own data from the paper to perform differential testing.


## 2. Installation

Please see the top level [README.md](/README.md) file for instructions on how
to prepare the bazel build environment.


## 3. Evaluating the artifact


### 3.1. Evaluate the generator
*(approximate runtime: 2 hours)*

Evaluate the DeepSmith generator by running the following program:

```sh
$ bazel run //docs/2018_07_issta/artifact_evaluation:01_evaluate_generator
```

The program uses a small OpenCL corpus of 1000 kernels randomly selected
from our GitHub corpus, pre-processes the corpus, trains a model on the
preprocessed corpus, and generates 1024 new OpenCL testcases.

We have reduced the size of the corpus and network so that it takes around 2
hours to train on a CPU. The model we used to generate the programs used in the
review copy of our paper is much larger (512 nodes per layer rather than 256),
is trained on more data (30 million tokens instead of 4.5 million), and is
trained for longer (50 epochs rather than 20). As such, the quality of output
of this small model is much lower, with very few syntactically correct programs
being generated.

Training the model can be interrupted and resumed at any time. Once trained, the
model does not need to be re-trained. The trained model is stored in
`/tmp/phd/docs/2018_07_issta/artifact_evaluation/clgen`.

Generated programs are written to the
directory `/tmp/phd/docs/2018_07_issta/artifact_evaluation/generated_kernels`.
Two Testcases are generated for each kernels, one single threaded, one
multi-threaded. Generated testcases are written to
`/tmp/phd/docs/2018_07_issta/artifact_evaluation/generated_testcases`.

#### 3.1.1. Extended Evaluation (optional)

By changing the parameters of the model defined in the
`//docs/2018_07_issta/artifact_evaluation/data/clgen.pbtxt`, you can evaluate
the output of models with different architectures, trained for different amounts
of time, etc. The schema for the file is defined in
`//deeplearning/deepsmith/proto/generator.proto`.

You could also change the corpus by adding additional OpenCL files, removing
files and training on only a subset, etc.


### 3.2. Evaluate the harness

*(approximate runtime: 30 minutes)*


Evaluate the DeepSmith harness by running a set of test cases using an
Oclgrind testbed:

```sh
$ bazel run //docs/2018_07_issta/artifact_evaluation/02_evaluate_harness
```

The program runs the 1024 generated testcases from the previous program, plus
a nuber of test cases taken from the experimental data we used in the paper,
located in the directory
`//docs/2018_07_issta/artifact_evaluation/data/testcases`. We include
pre-generated test cases so that we can differential test the results in the
next stage of the evaluation.

The results of execution are written to the directory
`/tmp/phd/docs/2018_07_issta/artifact_evaluation/results`. Each file in this
directory stores the result of a single test case execution. The schema for
these files is defined in `//deeplearning/deepsmith/proto/deepsmith.proto`.


#### 3.2.1. Extended Evaluation (optional)

You could add more test cases to execute, or manually change the contents of
testcases, for example, by changing the `src` field for a testcase to change
the OpenCL kernel which is evaluated.


### 3.3. Evaluate the results

*(approximate runtime: 5 minutes)*

Evaluate the DeepSmith difftester using:

```sh
$ bazel run //docs/2018_07_issta/artifact_evaluation/03_evaluate_results
```

The program evaluates the results generated from your local system in the
previous experiment, and differential tests the outputs against the data we
used in the paper. At the end of execution, the program prints a table of
classifications, using the same format as Table 2 of the paper. Individually
classified results are written to
`/tmp/phd/docs/2018_07_issta/artifact_evaluation/difftest_classifications/<class>/<device>/<±>/`.


#### 3.3.1. Extended Evaluation (optional)

The evaluation program difftests all results files from these directories:

```sh
/tmp/phd/docs/2018_07_issta/artifact_evaluation/difftest_classifications  # Results from your system
//docs/2018_07_issta/artifact_evaluation/data/our_results  # Results from our machines
```

You could add new results to these directories by repeatedly running the first
two programs. Alternatively you could modify individual results files, such as
by changing the returncode to simulate different test case outcomes, and
observe how that influences the classification of results.


## 4. Further Reading and References

DeepSmith is currently undergoing a ground-up rewrite to support more languages
and add new features, such as a distributing tests across multiple machines.
The work-in-progress implementation can be found in:

    https://github.com/ChrisCummins/phd/tree/master/deeplearning/deepsmith

The code we used for generating the data in the review copy of our paper is
available at:

    https://github.com/ChrisCummins/phd/tree/master/experimental/dsmith


## 5. License

Copyright 2017-2020 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See
[LICENSE](/docs/2018_07_issta/artifact_evaluation/LICENSE) for details.
