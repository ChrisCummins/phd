Command Line Interface
======================

The CLgen command line interface is accessible through the `clgen` command.

clgen
------

::

    usage: clgen [-h] [-v] [--version] [--debug] [--profile]
                 [--corpus-dir <corpus>] [--model-dir <model>]
                 [--sampler-dir <model> <sampler>]
                 {test,train,t,tr,sample,s,sa,db,fetch,f,fe,ls,preprocess,p,pp,features,atomize,cache}
                 ...
    
    A deep learning program generator for the OpenCL programming language.
    
    The core operations of CLgen are:
    
       1. OpenCL files are collected from a model specification file.
       2. These files are preprocessed into an OpenCL kernel database.
       3. A training corpus is generated from the input files.
       4. A machine learning model is trained on the corpus of files.
       5. The trained model is sampled for new kernels.
       6. The samples are tested for compilability.
    
    This program automates the execution of all six stages of the pipeline.
    The pipeline can be interrupted and resumed at any time. Results are cached
    across runs. If installed with CUDA support, NVIDIA GPUs will be used to
    improve performance where possible.
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase output verbosity
      --version             show version information and exit
      --debug               in case of error, print debugging information
      --profile             enable internal API profiling. When combined with
                            --verbose, prints a complete profiling trace
      --corpus-dir <corpus>
                            print path to corpus cache
      --model-dir <model>   print path to model cache
      --sampler-dir <model> <sampler>
                            print path to sampler cache
    
    available commands:
      {test,train,t,tr,sample,s,sa,db,fetch,f,fe,ls,preprocess,p,pp,features,atomize,cache}
        test                run the testsuite
        train (t, tr)       train models
        sample (s, sa)      train and sample models
        db                  manage databases
        fetch (f, fe)       gather training data
        ls                  list files
        preprocess (p, pp)  preprocess files for training
        features            extract OpenCL kernel features
        atomize             atomize files
        cache               manage filesystem cache
    
    For information about a specific command, run `clgen <command> --help`.
    
    Copyright (C) 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
    <http://chriscummins.cc/clgen>

