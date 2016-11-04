.. figure:: https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/logo.png
   :alt: CLgen: Deep Learning Program Generator.
   :target: http://chriscummins.cc/clgen/
   :width: 420 px
   :align: center
------

.. centered:: |Build Status| |Coverage Status| |Documentation Status| |Release Version| |License Badge|

**CLgen** is an open source application for generating runnable programs using
deep learning. CLgen *learns* to program using neural networks which model the
semantics and usage from large volumes of program fragments, generating  many-
core OpenCL programs that are representative of, but *distinct* from, the
programs it learns from.

Requirements
------------

-  Linux (x64) or OS X.
-  `GCC <https://gcc.gnu.org/>`__ > 4.8.2 or
   `clang <http://llvm.org/releases/download.html>`__ >= 3.1.
-  `GNU Make <http://savannah.gnu.org/projects/make>`__ > 3.79.
-  `Python <https://www.python.org/>`__ 2.7.12 or >= 3.4.
-  `git <https://git-scm.com/>`__ >= 1.8.1.4.
-  `libhdf5 <https://support.hdfgroup.org/HDF5/release/obtainsrc.html>`__
   >= 1.8.11.
-  `libffi <https://sourceware.org/libffi/>`__ >= 3.0.13.
-  `zlib <http://zlib.net/>`__ >= 1.2.3.4.
-  `curl <https://curl.haxx.se/>`__ and `wget <https://www.gnu.org/software/wget/>`__.

On Ubuntu, these can be installed using:

::

    $ sudo apt-get update
    $ sudo apt-get install build-essential python-dev git zlib1g-dev libhdf5-dev libffi-dev zlib1g-dev curl wget

On OS X, install these requirements using `Homebrew <http://brew.sh/>`_:

::

    $ brew tap homebrew/science
    $ brew install python git hdf5 libffi wget


Installation
------------

Install the latest release of CLgen using one of the following configurations:

**CPU-only:** *slow performance, some features disabled.*

::

    $ bash -c "$(curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.28/install-cpu.sh)"

**OpenCL enabled:** *slow performance, all features enabled.*
Requires `OpenCL <https://www.khronos.org/opencl/>`__.

::

    $ bash -c "$(curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.28/install-opencl.sh)"

**CUDA enabled:** *fast performance, all features enabled.*
Requires `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__ >= 6.5.

::

    $ bash -c "$(curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.28/install-cuda.sh)"

If you encounter any problems, please consider opening a `bug report
<https://github.com/ChrisCummins/clgen/issues>`_.


Contents
--------

.. toctree::

   installation
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

.. |Documentation Status| image:: https://img.shields.io/badge/docs-0.0.28-brightgreen.svg?style=flat
   :target: http://chriscummins.cc/clgen/

.. |Release Version| image:: https://img.shields.io/badge/release-0.0.28-blue.svg?style=flat
   :target: https://github.com/ChrisCummins/clgen/releases

.. |License Badge| image:: https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
