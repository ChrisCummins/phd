Command Line Interface
======================

The CLgen command line interface consists of a suite of related programs.

cldrive
--------

::

    usage: cldrive [-h] [--version] [-v] [-s] [--cpu] [--gpu] [--fatal-errors]
                   <input> [<input> ...]
    
    Drive OpenCL kernels.
    
    For each kernel, generate a randomly sized payload and execute.
    Use environment variable 'DSIZE' to override random payload size.
    
    Program output is in CSV format, with the following scheme:
    
        <path>,<dsize>,<kernel>,<transfer>,<mean>,<ci>
    
    where each value corresponds to:
    
       <path>      path to input file
       <dsize>     payload size
       <kernel>    kernel name
       <transfer>  transfer size, in bytes
       <mean>      mean execution time
       <ci>        95% confidence interval of execution time
    
    In case of an error, "-" is output for values which cannot be determined,
    and the kernel name field is substituted for an error name.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      <input>         input file(s) or directories
    
    optional arguments:
      -h, --help      show this help message and exit
      --version       show version information and exit
      -v, --verbose   increase output verbosity
      -s, --strict    reject any kernels which do not validate
      --cpu           execute on CPU (default: no)
      --gpu           execute on GPU (default: yes)
      --fatal-errors  exit on error

clgen
------

::

    usage: clgen [-h] [--version] [-v] [--corpus-dir] [--model-dir]
                 [--sampler-dir]
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
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      <model>        path to model dist or specification file
      <sampler>      path to sampler specification file
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      --corpus-dir   print path to corpus cache
      --model-dir    print path to model cache
      --sampler-dir  print path to sampler cache

clgen-atomize
--------------

::

    usage: clgen-atomize [-h] [--version] [-v] [-t TYPE] [-s] input
    
    Extract and print corpus vocabulary.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input                 path to input text file
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show version information and exit
      -v, --verbose         increase output verbosity
      -t TYPE, --type TYPE  vocabulary type
      -s, --size            print vocabulary size

clgen-create-db
----------------

::

    usage: clgen-create-db [-h] [--version] [-v] [-g] input
    
    Create an empty OpenCL kernel database.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL input dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      -g, --github   generate dataset with GitHub metadata

clgen-dist
-----------

::

    usage: clgen-dist [-h] [--version] [-v] [--author AUTHOR] <model> <distname>
    
    Package CLgen models for distribution.
    
    Once a CLgen model has been trained, it can be distribute to other devices
    for sampling. This program provides the mechanism for doing so. So called
    "dist files" contain trained Neural Network models and metadata describing
    the method by which they were trained.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      <model>          path to model specification file
      <distname>       name of dist file
    
    optional arguments:
      -h, --help       show this help message and exit
      --version        show version information and exit
      -v, --verbose    increase output verbosity
      --author AUTHOR  Name of author (default: $USER@$HOSTNAME)

clgen-explore
--------------

::

    usage: clgen-explore [-h] [--version] [-v] input
    
    Exploratory analysis of preprocessed dataset.
    
    Provides an overview of the contents of an OpenCL kernel database.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL input dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity

clgen-features
---------------

::

    usage: clgen-features [-h] [--version] [-v] [-d] [-s] [-e] [--shim] [-q] [-H]
                          inputs [inputs ...]
    
    Extract static OpenCL kernel features.
    
    This extracts a subset of the features required for the paper:
    
        Grewe, D., Wang, Z., & O'Boyle, M. F. P. M. (2013). Portable Mapping of
        Data Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
    
    Note that dynamic features are extracted using the cldrive program for CLgen
    kernels, or by using libcecl for ad-hoc programs.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      inputs              input path(s)
    
    optional arguments:
      -h, --help          show this help message and exit
      --version           show version information and exit
      -v, --verbose       increase output verbosity
      -d, --dir-mode      treat inputs as directories
      -s, --stats         summarize a features files
      -e, --fatal-errors  quit on compiler error
      --shim              include shim header
      -q, --quiet         minimal error output
      -H, --no-header     no features header

clgen-fetch
------------

::

    usage: clgen-fetch [-h] [--version] [-v] input paths [paths ...]
    
    Import OpenCL files into kernel datbase.
    
    The kernel database is used as a staging ground for input files, which are
    then preprocessed and assembled into corpuses. This program acts as the front
    end, assembling files from the file system into a database for preprocessing.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL dataset
      paths          path to OpenCL files or directories
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity

clgen-fetch-clgen
------------------

::

    usage: clgen-fetch-clgen [-h] [--version] [-v] [-d D] [-f F] [--first] input
    
    Generate OpenCL kernels from CLgen samples.
    
    This splits the continuous output of CLgen into discrete OpenCL kernels for
    preprocessing.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      -d D           path to samples directory
      -f F           path to sample file
      --first        extract only first kernel from sample file(s)

clgen-fetch-clsmith
--------------------

::

    usage: clgen-fetch-clsmith [-h] [--version] [-v] [-n N] input
    
    Generate OpenCL programs using CLSmith.
    
    CLSmith is a random program generator designed for fuzz testing OpenCL
    compilers and implementations.
    
    Install CLSmith into your system path from here:
    
       <https://github.com/ChrisLidbury/CLSmith>
    
    Note CLSmith is *not* developed by us. It is the efforts of the fine folks
    at Imperial College London: Christopher Lidbury, Andrei Lascu, Nathan Chong,
    Alastair F. Donaldson.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity
      -n N           number of OpenCL kernels to generate

clgen-fetch-db
---------------

::

    usage: clgen-fetch-db [-h] [--version] [-v] output input
    
    Copies OpenCL kernels from an existing SQL database into a new one.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      output         path to output SQL dataset
      input          path to input SQL dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity

clgen-fetch-github
-------------------

::

    usage: clgen-fetch-github [-h] [--version] [-v] input
    
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
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL input dataset
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity

clgen-merge
------------

::

    usage: clgen-merge [-h] [--version] [-v] dataset [inputs [inputs ...]]
    
    Merge kernel datasets.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      dataset        path to output dataset
      inputs         path to input datasets
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity

clgen-preprocess
-----------------

::

    usage: clgen-preprocess [-h] [--version] [-v] [-f] [-i]
                            [--remove-bad-preprocessed] [--remove-preprocessed]
                            inputs [inputs ...]
    
    Process OpenCL files for machine learning.
    
    This is a three step process. First, the OpenCL kernels are compiled to
    bytecode, then the source files are preprocessed, before being rewritten.
    
    Preprocessing is computationally demanding and highly paralellised.
    Expect high resource contention during preprocessing.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      inputs                path to input
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show version information and exit
      -v, --verbose         increase output verbosity
      -f, --file            treat input as file
      -i, --inplace         inplace file rewrite
      --remove-bad-preprocessed
                            delete the contents of all bad or ugly preprocessed files,
                            but keep the entries in the table
      --remove-preprocessed
                            remove all preprocessed files from database

clgen-train
------------

::

    usage: clgen-train [-h] [--version] [-v] [-d] [-i] [--input-samples] [--eof]
                       [-r] [-s STATUS]
                       input output
    
    Create training datasets.
    
    Provides a front-end for utilities for turning kernel databases into corpuses
    for training CLgen models on.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input                 path to SQL input dataset
      output                path to output file or directory
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show version information and exit
      -v, --verbose         increase output verbosity
      -d                    output to directory (overrides -i, --eof, -r)
      -i                    include file separators
      --input-samples       use input contents, not preprocessed
      --eof                 print end of file
      -r                    use reverse order
      -s STATUS, --status STATUS
                            status code to use

