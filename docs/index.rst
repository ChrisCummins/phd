.. figure:: https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/logo.png
   :alt: CLgen: Deep Learning Program Generator.
   :target: http://chriscummins.cc/clgen/
   :width: 420 px
   :align: center
------

.. centered:: |Build Status| |Coverage Status| |Documentation Status| |Python Version| |License Badge|

**CLgen** is an open source application for generating runnable programs using
deep learning. CLgen *learns* to program using neural networks which model the
semantics and usage from large volumes of program fragments, generating many-
core OpenCL programs that are representative of, but *distinct* from, the
programs it learns from.

.. figure:: https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/pipeline.png
   :alt: CLgen synthesis pipeline.
   :width: 500 px

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

Contents
--------

.. toctree::

   build_system
   binaries
   api
   license


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Build Status| image:: https://img.shields.io/travis/ChrisCummins/clgen/master.svg?style=flat
   :target: https://travis-ci.org/ChrisCummins/clgen

.. |Coverage Status| image:: https://img.shields.io/coveralls/ChrisCummins/clgen/master.svg?style=flat
   :target: https://coveralls.io/github/ChrisCummins/clgen?branch=master

.. |Documentation Status| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: http://chriscummins.cc/clgen/

.. |Python Version| image:: https://img.shields.io/badge/python-2%20%26%203-blue.svg?style=flat
   :target: https://www.python.org/

.. |License Badge| image:: https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
