===========================================================
Compiler Fuzzing through Deep Learning: Artifact Evaluation
===========================================================

The full dataset is quite large (>200 GB), and we are working on finding a
method for sharing it with the community.


Contents
========

  install.sh
    Our automated installation process.

  01_evaluate_generator/
    A demonstration of training a DeepSmith model to generate programs.

  02_evaluate_harness/
    A demonstration which executes test cases on a local OpenCL test bed which
    we found to cause crashes/bugs in at least one device.

  03_evaluate_results/
    A script which differential tests the output of the prior script and our
    data, to show the classification of results.


Requirements
============

Operating system
----------------

Ubuntu Linux or macOS. Unfortunately Windows is not supported by many of our
dependencies. Other Linux distributions may work, but we have not tested them,
and our install.sh script uses the apt package manager to automate the
installation of depdencies. If you have requirements for a specific Linux
distribution that is not Ubuntu >= 16.04, please contact us.

OpenCL
------

macOS has OpenCL support built in.

Disk space
----------

Due to time constraints we have not optimized our artifact for space, requiring
approximately 10GB of disk space in total. Please contact us if this is an
issue.


Instructions
============

Installation
------------

Install local and system-wide dependencies using the command:

    $ ./install.sh

The entire installation process is automated. Please note that it may take
some time. In case of a problem at this stage please contact us.


Evaluate the generator
----------------------

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


Evaluate the harness
--------------------

Evaluate the DeepSmith harness by running the following script:

    $ ./02_evaluate_harness/run.sh

The script executes the 1000 programs generated in the last experiment on a
local OpenCL device.


Evaluate the results
--------------------

Evaluate the DeepSmith difftester by running the following script:

    $ ./03_evaluate_results/run.sh


Clean up
--------

Once you are done evaluating our artifact, you can remove the files generated
during installation and experiment execution using the command:

  $ ./cleanup.sh

Please note that system-wide packages which are installed (for example
through apt-get) are not removed, since we do not know if they were installed
by our artifact or by you.


Further Reading
===============

DeepSmith is currently undergoing a ground-up rewrite to support more languages
and add new features, such as a distributing tests across multiple machines.
The work-in-progress implementation can be found in:

    https://github.com/ChrisCummins/phd/tree/master/deeplearning/deepsmith

The implementation used for the review copy of our paper is available at:

    https://github.com/ChrisCummins/phd/tree/master/experimental/dsmith

We extended our neural network program generator CLgen for generating the
tests used in the review copy of our paper. The documentation for CLgen is
available at:

    http://chriscummins.cc/clgen
