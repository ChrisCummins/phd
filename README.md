<div align="center">
  <img src="https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/logo.png" width="420">
</div>
-------

<div align="center">
  <a href="https://travis-ci.org/ChrisCummins/clgen" target="_blank">
    <img src="https://img.shields.io/travis/ChrisCummins/clgen/master.svg?style=flat">
  </a>
  <a href="https://coveralls.io/github/ChrisCummins/clgen?branch=master">
    <img src="https://img.shields.io/coveralls/ChrisCummins/clgen/master.svg?style=flat">
  </a>
  <a href="http://chriscummins.cc/clgen/" target="_blank">
    <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat">
  </a>
   <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-2%20%26%203-blue.svg?style=flat">
  </a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
    <img src="https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat">
  </a>
</div>

**CLgen** is an open source application for generating runnable programs using
deep learning. CLgen *learns* to program using neural networks which model the
semantics and usage from large volumes of program fragments, generating many-
core OpenCL programs that are representative of, but *distinct* from, the
programs it learns from.

<img src="https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/pipeline.png" width="500">

## Requirements

*  Linux (x64) or OS X.
*  [GCC](https://gcc.gnu.org/) > 4.7 or
   [clang](http://llvm.org/releases/download.html) >= 3.1.
*  [GNU Make](http://savannah.gnu.org/projects/make) > 3.79.
*  [Python](https://www.python.org/) 2.7 or >= 3.4.
*  [zlib](http://zlib.net/) >= 1.2.3.4.
*  [libhdf5](https://support.hdfgroup.org/HDF5/release/obtainsrc.html) >= 1.8.11.

Optional, but highly recommended:

*  [OpenCL](https://www.khronos.org/opencl/) == 1.2.
*  An NVIDIA GPU and
   [CUDA](http://www.nvidia.com/object/cuda_home_new.html) >= 6.5.

## Building from Source

Checkout this repository and all submodules:

```sh
$ git clone --recursive https://github.com/ChrisCummins/clgen.git
```

Configure and build CLgen:

```sh
$ cd clgen
$ ./configure
$ make
```

Install into your system path:

```sh
$ sudo make install
```

(Optional) Run the test suite:

```sh
$ make test
```

## Getting Started

Train and sample a very small clgen model using the included training
set:

```sh
$ clgen model.json sampler.json
```

See the [online documentation](http://chriscummins.cc/clgen/) for more
information.

## License

Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
