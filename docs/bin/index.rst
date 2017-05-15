Command Line Interface
======================

The CLgen command line interface consists of a suite of related programs.

clgen
------

::

    usage: clgen [-h] [--version] [-v] [--debug] [--profile] [--ls-files]
                 [--corpus-dir] [--model-dir] [--sampler-dir]
                 <model> <sampler>
    
    Generate OpenCL programs using Deep Learning.
    
    This is a five-step process:
       1. Input files are collected from the model specification file.
       2. The input files are preprocessed into an OpenCL kernel database.
       3. A training corpus is generated from the input files.
       4. A model is instantiated and trained on the corpus.
       5. The trained model is sampled for new kernels.
    
    This program automates the execution of all five stages of the pipeline.
    The pipeline can be interrupted and resumed at any time. Results are cached
    across runs.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      <model>        path to model dist or specification file
      <sampler>      path to sampler specification file
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling
      --ls-files     print cached corpus, model, and sampler, files
      --corpus-dir   print path to corpus cache
      --model-dir    print path to model cache
      --sampler-dir  print path to sampler cache

clgen-atomize
--------------

::

    usage: clgen-atomize [-h] [--version] [-v] [--debug] [--profile] [-t TYPE]
                         [-s]
                         input
    
    Extract and print corpus vocabulary.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input                 path to input text file
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show version information and exit
      -v, --verbose         increase output verbosity
      --debug               in case of error, print debugging information
      --profile             enable internal API profiling
      -t TYPE, --type TYPE  vocabulary type
      -s, --size            print vocabulary size

clgen-create-db
----------------

::

    usage: clgen-create-db [-h] [--version] [-v] [--debug] [--profile] [-g] input
    
    Create an empty OpenCL kernel database.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL input dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling
      -g, --github   generate dataset with GitHub metadata

clgen-dump
-----------

::

    usage: clgen-dump [-h] [--version] [-v] [--debug] [--profile] [-d] [-i]
                      [--input-samples] [--eof] [-r] [-s STATUS]
                      input output
    
    Dump kernel dataset to file(s).
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input                 path to kernels database
      output                path to output file or directory
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show version information and exit
      -v, --verbose         increase output verbosity
      --debug               in case of error, print debugging information
      --profile             enable internal API profiling
      -d                    output to directory (overrides -i, --eof, -r)
      -i                    include file separators
      --input-samples       use input contents, not preprocessed
      --eof                 print end of file
      -r                    use reverse order
      -s STATUS, --status STATUS
                            status code to use

clgen-explore
--------------

::

    usage: clgen-explore [-h] [--version] [-v] [--debug] [--profile] input
    
    Exploratory analysis of preprocessed dataset.
    
    Provides an overview of the contents of an OpenCL kernel database.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL input dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling

clgen-features
---------------

::

    usage: clgen-features [-h] [--version] [-v] [--debug] [--profile] [-d] [-s]
                          [-e] [--shim] [-q] [-H]
                          inputs [inputs ...]
    
    Extract static OpenCL kernel features.
    
    This extracts the static compile-time features of the paper:
    
        Grewe, D., Wang, Z., & O'Boyle, M. F. P. M. (2013). Portable Mapping of
        Data Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      inputs              input path(s)
    
    optional arguments:
      -h, --help          show this help message and exit
      --version           show version information and exit
      -v, --verbose       increase output verbosity
      --debug             in case of error, print debugging information
      --profile           enable internal API profiling
      -d, --dir-mode      treat inputs as directories
      -s, --stats         summarize a features files
      -e, --fatal-errors  quit on compiler error
      --shim              include shim header
      -q, --quiet         minimal error output
      -H, --no-header     no features header

clgen-fetch
------------

::

    usage: clgen-fetch [-h] [--version] [-v] [--debug] [--profile]
                       input paths [paths ...]
    
    Import OpenCL files into kernel datbase.
    
    The kernel database is used as a staging ground for input files, which are
    then preprocessed and assembled into corpuses. This program acts as the front
    end, assembling files from the file system into a database for preprocessing.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL dataset
      paths          path to OpenCL files or directories
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling

clgen-fetch-github
-------------------

::

    usage: clgen-fetch-github [-h] [--version] [-v] [--debug] [--profile] input
    
    Mines OpenCL kernels from Github. Requires the following environment
    variables to be set:
    
         GITHUB_USERNAME   github username
         GITHUB_PW         github password
         GITHUB_TOKEN      github api token
    
    For instructions to generate an API token, see:
    
      <https://help.github.com/articles/creating-an-access-token-for-command-line-use/>
    
    This process issues thousands of GitHub API requests per minute. Please
    exercise restrained in minimizing your use of this program -- we don't
    want to upset the nice folks at GH :-)
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL input dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling

clgen-grid
-----------

::

    usage: clgen-grid [-h] [--version] [-v] [--debug] [--profile]
    
    Print model stats.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling

clgen-merge
------------

::

    usage: clgen-merge [-h] [--version] [-v] [--debug] [--profile]
                       dataset [inputs [inputs ...]]
    
    Merge kernel datasets.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      dataset        path to output dataset
      inputs         path to input datasets
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling

clgen-preprocess
-----------------

::

    usage: clgen-preprocess [-h] [--version] [-v] [--debug] [--profile] [-f] [-i]
                            [-G] [--remove-bad-preprocessed]
                            [--remove-preprocessed]
                            inputs [inputs ...]
    
    Process OpenCL files for machine learning.
    
    This is a three step process. First, the OpenCL kernels are compiled to
    bytecode, then the source files are preprocessed, before being rewritten.
    
    Preprocessing is computationally demanding and highly paralellised.
    Expect high resource contention during preprocessing.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      inputs                path to input
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show version information and exit
      -v, --verbose         increase output verbosity
      --debug               in case of error, print debugging information
      --profile             enable internal API profiling
      -f, --file            treat input as file
      -i, --inplace         inplace file rewrite
      -G, --gpuverify       run GPUVerify on kernels
      --remove-bad-preprocessed
                            delete the contents of all bad or ugly preprocessed files,
                            but keep the entries in the table
      --remove-preprocessed
                            remove all preprocessed files from database

clgen-refresh-cache
--------------------

::

    usage: clgen-refresh-cache [-h] [--version] [-v] [--debug] [--profile]
    
    Refresh the cached model, corpus, and sampler IDs.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling

clgen-test
-----------

::

    usage: clgen-test [-h] [--version] [-v] [--debug] [--profile]
                      [--coveragerc-path] [--coverage-path]
    
    Run the CLgen self-test suite.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    optional arguments:
      -h, --help         show this help message and exit
      --version          show version information and exit
      -v, --verbose      increase output verbosity
      --debug            in case of error, print debugging information
      --profile          enable internal API profiling
      --coveragerc-path  print path to coveragerc file
      --coverage-path    print path to coverage file

clgen-train
------------

::

    usage: clgen-train [-h] [--version] [-v] [--debug] [--profile] <model>
    
    Train a CLgen model.
    
    Copyright (C) 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      <model>        path to model dist or specification file
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --debug        in case of error, print debugging information
      --profile      enable internal API profiling

