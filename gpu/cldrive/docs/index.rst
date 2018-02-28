cldrive
=======

.. toctree::
   :maxdepth: 2
   :caption: Contents:


cldrive
-------
.. automodule:: cldrive
   :members:

cldrive.args
------------

.. automodule:: cldrive.args
   :members:

cldrive.cgen
------------

.. automodule:: cldrive.cgen
   :members:

cldrive.env
-----------

.. automodule:: cldrive.env
   :members:

cldrive.data
------------

.. automodule:: cldrive.data
   :members:

cldrive.driver
--------------

.. automodule:: cldrive.driver
   :members:


Command Line Interface
======================

::

  usage: cldrive [-h] [--version] [--clinfo] [-p <platform name>]
                 [-d <device name>] [--devtype <devtype>] [-s <size>]
                 [-i <{rand,arange,zeros,ones}>] [--scalar-val <float>]
                 [-g <global size>] [-l <local size>] [-t <timeout>] [--no-opts]
                 [--profiling] [--debug] [-b]

  Reads an OpenCL kernel from stdin, generates data for it, executes it on a
  suitable device, and prints the outputs. Copyright (C) 2017 Chris Cummins
  <chrisc.101@gmail.com>. <https://github.com/ChrisCummins/cldrive>

  optional arguments:
    -h, --help            show this help message and exit
    --version             show version information and exit
    --clinfo              list available OpenCL devices and exit
    -p <platform name>, --platform <platform name>
                          use OpenCL platform with name, e.g. 'NVIDIA CUDA'
    -d <device name>, --device <device name>
                          use OpenCL device with name, e.g. 'GeForce GTX 1080'
    --devtype <devtype>   use any OpenCL device of type {all,cpu,gpu} (default:
                          all)
    -s <size>, --size <size>
                          size of input arrays to generate (default: 64)
    -i <{rand,arange,zeros,ones}>, --generator <{rand,arange,zeros,ones}>
                          input generator to use, one of:
                          {rand,arange,zeros,ones} (default: arange)
    --scalar-val <float>  values to assign to scalar inputs (default: <size>
                          argumnent)
    -g <global size>, --gsize <global size>
                          comma separated NDRange for global size (default:
                          64,1,1)
    -l <local size>, --lsize <local size>
                          comma separated NDRange for local (workgroup) size
                          (default: 32,1,1)
    -t <timeout>, --timeout <timeout>
                          error if execution has not completed after this many
                          seconds(default: off)
    --no-opts             disable OpenCL optimizations (on by default)
    --profiling           enable kernel and transfer profiling
    --debug               enable more verbose OpenCL copmilation and execution
    -b, --binary          Print outputs as a pickled binary numpy array


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
