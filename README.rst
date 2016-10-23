clgen - Deep Learning Program Generator
=======================================

|Build Status|

CLgen is the first application for generating executable programs using
deep learning. It *learns* to program using neural networks which model
the semantics and usage from large volumes of program fragments,
generating many-core OpenCL programs that are representative of, but
*distinct* from, the programs it learns from.

Requirements
------------

-  Linux (x64) or OS X.
-  An NVIDIA GPU and
   `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__ >= 6.5.
-  `GNU Make <http://savannah.gnu.org/projects/make>`__ >= 3.79.
-  `CMake <https://cmake.org/>`__ >= 3.4.3.
-  `Ninja <https://ninja-build.org/>`__ >= 1.7.
-  `GCC <https://gcc.gnu.org/>`__ >= 4.7 or
   `clang <http://llvm.org/releases/download.html>`__ >= 3.1.
-  `Python <https://www.python.org/>`__ 2.7 or >= 3.4.
-  `zlib <http://zlib.net/>`__ >= 1.2.3.4.
-  `libhdf5 <https://support.hdfgroup.org/HDF5/>`__ >= 1.8.11.

Getting started
---------------

::

    $ sudo make install

Build and install clgen.

::

    $ clgen model.json arguments.json

Train and sample a clgen model using a small included training set.

License
-------

Copyright 2016 Chris Cummins chrisc.101@gmail.com.

Released under the terms of the GPLv3 license. See
`LICENSE.txt </LICENSE.txt>`__ for details.

.. |Build Status| image:: https://travis-ci.com/ChrisCummins/clgen.svg?token=RpzWC2nNxou66YeqVQYw&branch=master
   :target: https://travis-ci.com/ChrisCummins/clgen
