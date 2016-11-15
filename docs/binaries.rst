Programs
========

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

    usage: clgen [-h] [--version] [-v] <model> <sampler>
    
    Generate OpenCL programs using Deep Learning.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      <model>        path to model dist or specification file
      <sampler>      path to sampler specification file
    
    optional arguments:
      -h, --help     show this help message and exit
      --version      show version information and exit
      -v, --verbose  increase output verbosity

clgen-create-db
----------------

::

    usage: clgen-create-db [-h] [--version] [-v] [-g] input
    
    Create an empty SQL database.
    
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
    
    Package CLgen model for distribution.
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      <model>          path to model specification file
      <distname>       name of dist file
    
    optional arguments:
      -h, --help       show this help message and exit
      --version        show version information and exit
      -v, --verbose    increase output verbosity
      --author AUTHOR  Name of author (default: cec@florence)

clgen-explore
--------------

::

    usage: clgen-explore [-h] [--version] [-v] input
    
    Exploratory analysis of preprocessed dataset.
    
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
    
    Extract OpenCL kernel features.
    
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
    
    Import OpenCL files into datbase.
    
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
    
    Exploratory analysis of preprocessed dataset.
    
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
    
    Import kernels from an existing database.
    
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
    
    Fetch OpenCL kernels from Github. Reads github authentication
    from environmental variables:
    
         GITHUB_USERNAME   github username
         GITHUB_PW         github password
         GITHUB_TOKEN      github api token
    
    Copyright (C) 2016 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>
    
    positional arguments:
      input          path to SQL input dataset
    
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

