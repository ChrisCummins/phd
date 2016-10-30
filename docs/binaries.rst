Executables
===========

clgen
------

::

    usage: clgen [-h] [-v] [--version] <model-json> <sampler-json>
    
    Generate OpenCL programs using Deep Learning.
    
    positional arguments:
      <model-json>    path to model specification file
      <sampler-json>  path to sampler specification file
    
    optional arguments:
      -h, --help      show this help message and exit
      -v, --verbose   increase output verbosity
      --version       show version information and exit

clgen-create-db
----------------

::

    usage: clgen-create-db [-h] [-g] input
    
    positional arguments:
      input       path to SQL input dataset
    
    optional arguments:
      -h, --help  show this help message and exit
      -g          generate dataset with GitHub metadata

clgen-drive
------------

::

    usage: clgen-drive [-h] [-f] [-s] [--cpu] [--gpu] [--fatal-errors] input
    
    positional arguments:
      input           path to input
    
    optional arguments:
      -h, --help      show this help message and exit
      -f              treat input as file
      -s, --strict    reject any kernels which do not validate
      --cpu           execute on CPU (default: no)
      --gpu           execute on GPU (default: yes)
      --fatal-errors  exit on failure

clgen-explore
--------------

::

    usage: clgen-explore [-h] input
    
    positional arguments:
      input       path to SQL input dataset
    
    optional arguments:
      -h, --help  show this help message and exit

clgen-features
---------------

::

    usage: clgen-features [-h] [-d] [-s] [-e] [--shim] [-q] [-H]
                          inputs [inputs ...]
    
    positional arguments:
      inputs              input path(s)
    
    optional arguments:
      -h, --help          show this help message and exit
      -d, --dir-mode      treat inputs as directories
      -s, --stats         summarize a features files
      -e, --fatal-errors  quit on compiler error
      --shim              include shim header
      -q, --quiet         minimal error output
      -H, --no-header     no features header

clgen-fetch
------------

::

    usage: clgen-fetch [-h] input paths [paths ...]
    
    positional arguments:
      input       path to SQL dataset
      paths       path to OpenCL files or directories
    
    optional arguments:
      -h, --help  show this help message and exit

clgen-fetch-clgen
------------------

::

    usage: clgen-fetch-clgen [-h] [-d D] [-f F] [--first] input
    
    positional arguments:
      input       path to SQL dataset
    
    optional arguments:
      -h, --help  show this help message and exit
      -d D        path to samples directory
      -f F        path to sample file
      --first     extract only first kernel from sample file(s)

clgen-fetch-clsmith
--------------------

::

    usage: clgen-fetch-clsmith [-h] [-n N] input
    
    positional arguments:
      input       path to SQL dataset
    
    optional arguments:
      -h, --help  show this help message and exit
      -n N        number of OpenCL kernels to generate

clgen-fetch-db
---------------

::

    usage: clgen-fetch-db [-h] output input
    
    positional arguments:
      output      path to output SQL dataset
      input       path to input SQL dataset
    
    optional arguments:
      -h, --help  show this help message and exit

clgen-fetch-github
-------------------

::

    usage: clgen-fetch-github [-h] input
    
    positional arguments:
      input       path to SQL input dataset
    
    optional arguments:
      -h, --help  show this help message and exit

clgen-preprocess
-----------------

::

    usage: clgen-preprocess [-h] [-f] [-i] [--remove-bad-preprocessed]
                            inputs [inputs ...]
    
    positional arguments:
      inputs                path to input
    
    optional arguments:
      -h, --help            show this help message and exit
      -f, --file            treat input as file
      -i, --inplace         inplace file rewrite
      --remove-bad-preprocessed
                            delete the contents of all bad or ugly preprocessed
                            files, but keep the entries in the table

clgen-train
------------

::

    usage: clgen-train [-h] [-d] [-i] [--input-samples] [--eof] [-r] [-s STATUS]
                       input output
    
    positional arguments:
      input                 path to SQL input dataset
      output                path to output file or directory
    
    optional arguments:
      -h, --help            show this help message and exit
      -d                    output to directory (overrides -i, --eof, -r)
      -i                    include file separators
      --input-samples       use input contents, not preprocessed
      --eof                 print end of file
      -r                    use reverse order
      -s STATUS, --status STATUS
                            status code to use

