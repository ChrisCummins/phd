.. figure:: https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/logo.png
   :alt: CLgen: Deep Learning Program Generator.
   :target: http://chriscummins.cc/clgen/
   :width: 420 px
   :align: center
------

.. centered:: |Build Status| |Coverage Status| |Documentation Status| |Release Version| |License Badge|

Welcome to the CLgen documentation.

Requirements
------------

-  Linux (x64) or OS X.
-  `GCC <https://gcc.gnu.org/>`__ > 4.7 or
   `clang <http://llvm.org/releases/download.html>`__ >= 3.1.
-  `GNU Make <http://savannah.gnu.org/projects/make>`__ > 3.79.
-  `Python <https://www.python.org/>`__ 2.7 or >= 3.4.
-  `git <https://git-scm.com/>`__ >= 1.8.1.4.
-  `zlib <http://zlib.net/>`__ >= 1.2.3.4.
-  `libhdf5 <https://support.hdfgroup.org/HDF5/release/obtainsrc.html>`__
   >= 1.8.11.

Optional, but highly recommended:

-  `OpenCL <https://www.khronos.org/opencl/>`__ == 1.2.
-  An NVIDIA GPU and
   `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__ >= 6.5.


Installation
------------

Download and unpack the latest CLgen release:

::

    $ wget https://github.com/ChrisCummins/clgen/archive/0.0.13.tar.gz -O clgen-0.0.13.tar.gz
    $ tar xf clgen-0.0.13.tar.gz && cd cglen-0.0.13

Configure and build CLgen using:

::

    $ ./configure
    $ make
    $ sudo make install

This may take some time (over an hour). If you encounter any problems,
please read the `deailed build instructions <installation.html>`_ and consider
opening a `bug report <https://github.com/ChrisCummins/clgen/issues>`_.

(Optional) Run the test suite using:

::

    $ make test


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

.. |Documentation Status| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: http://chriscummins.cc/clgen/

.. |Release Version| image:: https://img.shields.io/badge/release-0.0.13-blue.svg?style=flat
   :target: https://github.com/ChrisCummins/clgen/releases

.. |License Badge| image:: https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
