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
-  `libhdf5 <https://support.hdfgroup.org/HDF5/release/obtainsrc.html>`__ >=
   1.8.11.
-  `libffi <https://sourceware.org/libffi/>`__ >= 3.0.13.
-  `zlib <http://zlib.net/>`__ >= 1.2.3.4.
-  `git <https://git-scm.com/>`__, `curl <https://curl.haxx.se/>`__ and `wget
   <https://www.gnu.org/software/wget/>`__.

On **Ubuntu 16.04** or **OS X**, the following command will install the
requirements automatically:

::

    $ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.30/install-deps.sh | bash

Note that on Ubuntu, sudo privileges are required to install these requirements
- you may be prompted for your password.

For **other Linux distributions** and older versions of Ubuntu, please check
each of these requirements yourself.


Installation - Virtualenv
-------------------------

We recommend using a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_
environment to install CLgen. This installs CLgen in its own directory, not
impacting any existing programs on the machine. Unlike a system-wide install,
this does not require sudo privileges.

Create a virtualenv environment in the directory `~/clgen`:

::

    $ virtualenv --system-site-packages ~/clgen

Activate this environment:

::

    $ source ~/clgen/bin/activate

Install the latest release of CLgen using one of the following configurations:

**1. CPU-only:** *slow performance, some features disabled.*

::

    (clgen)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.30/install-cpu.sh | bash

**2. OpenCL enabled:** *slow performance, all features enabled.*
Requires `OpenCL <https://www.khronos.org/opencl/>`__.

::

    (clgen)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.30/install-opencl.sh | bash

**3. CUDA enabled:** *fast performance, all features enabled.* Requires NVIDIA
GPU, `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__ >= 6.5.

::

    (clgen)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.30/install-cuda.sh | bash

When you are done using CLgen, deactivate the virtualenv environment:

::

    (clgen)$ deactivate

To use CLgen later you will need to activate the virtualenv environment again:

::

    $ source ~/clgen/bin/activate


Installation - System-wide
--------------------------

A system-wide install allows any user on the machine to run CLgen without
activating a virtualenv environment. This may update some of the previously
installed Python packages. System-wide installation requires sudo priveledges -
you may be prompted for your password.

Install the latest release of CLgen system-wide using one of the following
configurations:

**1. CPU-only:** *slow performance, some features disabled.*

::

    $ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.30/install-cpu.sh | bash

**2. OpenCL enabled:** *slow performance, all features enabled.*
Requires `OpenCL <https://www.khronos.org/opencl/>`__.

::

    $ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.30/install-opencl.sh | bash

**3. CUDA enabled:** *fast performance, all features enabled.*
Requires `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`__ >= 6.5.

::

    $ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.0.30/install-cuda.sh | bash


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

.. |Documentation Status| image:: https://img.shields.io/badge/docs-0.0.30-brightgreen.svg?style=flat
   :target: http://chriscummins.cc/clgen/

.. |Release Version| image:: https://img.shields.io/badge/release-0.0.30-blue.svg?style=flat
   :target: https://github.com/ChrisCummins/clgen/releases

.. |License Badge| image:: https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
