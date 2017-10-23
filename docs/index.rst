.. figure:: https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/logo.png
   :alt: CLgen: Deep Learning Program Generator.
   :target: http://chriscummins.cc/clgen/
   :width: 420 px
   :align: center
------

.. centered:: |Documentation Status| |Build Status| |Coverage Status| |Release Version| |License Badge|

**CLgen** is an open source application for generating runnable programs using
deep learning. CLgen *learns* to program using neural networks which model the
semantics and usage from large volumes of program fragments, generating
many-core OpenCL programs that are representative of, but *distinct* from, the
programs it learns from.

Requirements
------------

For **Ubuntu 16.04** or **OS X**, the following command will install the
CLgen requirements automatically:

::

    $ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.4.0.dev0/install-deps.sh | bash

Note that on Ubuntu, sudo privileges are required to install these requirements
- you may be prompted for your password.

For **other Linux distributions** and older versions of Ubuntu, please check
each for each of the following requirements:

-  `Python <https://www.python.org/>`_ >= 3.6.
-  `Mono <http://www.mono-project.com/>`_.
-  `clang <http://llvm.org/releases/download.html>`_ >= 3.1.
-  `GNU Make <http://savannah.gnu.org/projects/make>`_ > 3.79.
-  `glibc <https://www.gnu.org/software/libc/>`_ >= 2.16.
-  `libhdf5 <https://support.hdfgroup.org/HDF5/release/obtainsrc.html>`_ >=
   1.8.11.
-  `libffi <https://sourceware.org/libffi/>`_ >= 3.0.13.
-  `zlib <http://zlib.net/>`_ >= 1.2.3.4.
-  `GNU Readline <https://cnswww.cns.cwru.edu/php/chet/readline/rltop.html>`_.
-  `ncurses <https://www.gnu.org/software/ncurses/>`_.
-  `git <https://git-scm.com/>`_.
-  `curl <https://curl.haxx.se/>`_.
-  `wget <https://www.gnu.org/software/wget/>`_.
-  `patch <https://linux.die.net/man/1/patch>`_.

Optional, but recommended:

- NVIDIA GPU with
  `CUDA 8.0 and cuDNN <https://www.tensorflow.org/get_started/os_setup#optional_install_cuda_gpus_on_linux>`_.


Installation - Virtualenv
-------------------------

We recommend using a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_
environment to install CLgen. This installs CLgen in its own directory, without
impacting any existing programs on the machine. Installing in a virtualenv
environment does not require sudo privileges.

Create a virtualenv environment in the directory `~/clgen`:

::

    $ virtualenv -p python3 ~/clgen

Activate this environment:

::

    $ source ~/clgen/bin/activate

Install the latest release of CLgen:

::

    # if you have an NVIDIA GPU with CUDA 8.0 and cuDNN:
    (clgen)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.4.0.dev0/install-cuda.sh | bash
    # CPU only:
    (clgen)$ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.4.0.dev0/install-cpu.sh | bash

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

Install the latest release of CLgen:

::

    # if you have an NVIDIA GPU with CUDA 8.0 and cuDNN:
    $ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.4.0.dev0/install-cuda.sh | bash
    # CPU only:
    $ curl -s https://raw.githubusercontent.com/ChrisCummins/clgen/0.4.0.dev0/install-cpu.sh | bash


Contents
--------

.. toctree::

   installation
   api/index
   bin/index
   versions
   license


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Documentation Status| image:: https://img.shields.io/badge/docs-0.4.0.dev0-brightgreen.svg?style=flat
   :target: http://chriscummins.cc/clgen/

.. |Build Status| image:: https://img.shields.io/travis/ChrisCummins/clgen/master.svg?style=flat
   :target: https://travis-ci.org/ChrisCummins/clgen

.. |Coverage Status| image:: https://img.shields.io/coveralls/ChrisCummins/clgen/master.svg?style=flat
   :target: https://coveralls.io/github/ChrisCummins/clgen?branch=master

.. |Release Version| image:: https://img.shields.io/badge/release-0.4.0.dev0-blue.svg?style=flat
   :target: https://github.com/ChrisCummins/clgen/releases

.. |License Badge| image:: https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
