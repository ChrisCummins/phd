.. figure:: assets/logo.svg
   :alt: CLgen: Deep Learning Program Generator.
   :width: 420 px
   :align: center

------

.. centered:: |Build Status| |Documentation Status| |Python Version| |License Badge|

**CLgen** is an open source application for generating runnable programs using
deep learning. CLgen *learns* to program using neural networks which model
the semantics and usage from large volumes of program fragments, generating
many-core OpenCL programs that are representative of, but *distinct* from,
the programs it learns from.

.. figure:: assets/pipeline.png
   :alt: labm8

Requirements
------------

-  Linux (x64) or OS X.
-  `GCC <https://gcc.gnu.org/>`__ >= 4.7 or
   `clang <http://llvm.org/releases/download.html>`__ >= 3.1.
-  `GNU Make <http://savannah.gnu.org/projects/make>`__ >= 3.79.
-  `Python <https://www.python.org/>`__ 2.7 or >= 3.4.
-  `OpenCL <https://www.khronos.org/opencl/>`__ == 1.2.
-  `zlib <http://zlib.net/>`__ >= 1.2.3.4.
-  `libhdf5 <https://support.hdfgroup.org/HDF5/release/obtainsrc.html>`__ >= 1.8.11.

Optional, but highly recommended:

-  An NVIDIA GPU and
   `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__ >= 6.5.

Getting started
---------------

Checkout CLgen locally:

::

    git clone --recursive https://github.com/ChrisCummins/clgen.git

If CUDA support is not available on your system (see optional requirements),
run:

::

    export CLGEN_GPU=0

Build and install clgen:

::

    ./install.sh

Run the test suite. Everything should pass:

::

    ./test.sh

Train and sample a very small clgen model using the included training set:

::

    clgen model.json sampler.json

License
-------

Copyright 2016 Chris Cummins chrisc.101@gmail.com.

Released under the terms of the GPLv3 license. See
`LICENSE.txt </LICENSE.txt>`__ for details.

.. |Build Status| image:: https://travis-ci.org/ChrisCummins/clgen.svg?branch=master
   :target: https://travis-ci.org/ChrisCummins/clgen

.. |Documentation Status| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg

.. |Python Version| image:: https://img.shields.io/badge/python-2%20%26%203-blue.svg
   :target: https://www.python.org/

.. |License Badge| image:: https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
