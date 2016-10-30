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
