# Building CLgen

If your system has the [required
components](https://github.com/ChrisCummins/clgen#requirements), then building
CLgen should be a relatively straightforward. First configure your build:

```
$ ./configure
```
Then, compile and install the clgen software stack in one shot:

```sh
$ make
$ sudo make install
```

This may take some time (over two hours). If you encounter any problems,
please consider opening a [bug
report](https://github.com/ChrisCummins/clgen/issues). The remainder of this
document provides a brief outline of the build process, which may be useful for
diagnosing a problem, if one arises.

## Host Toolchain

CLgen requires the LLVM and Clang libraries and binaries. These are built in
three steps:

```sh
$ make ninja
```

The ninja build tool is downloaded and compiled in `native/ninja`.

```sh
$ make cmake
```

A precompiled version of CMake is downloaded to `native/cmake`.

```
$ make llvm
```

The clang compiler is downloaded and compiled in in `native/llvm`.

## Native File Compilation

```sh
$ make all
```

Once the toolchain has been built, it is used to compile the native components
of CLgen. Roughly, those are:

```sh
$ make libclc
```

Libclc provides headers for a generic OpenCL implementation.

```
$ make torch
```

The machine learning is implemented using torch.

```
$ make torch-rnn
```

Torch-rnn provides the recursive neural network.

## Python Package Installation

One the native components are compiled, the remainder of the build process
involves installing the python package:

```sh
$ sudo make install
$ make test
```

This is roughly equivalent to:

```sh
$ pip install numpy>=1.10.4
$ pip install -r requirements.txt
$ sudo python ./setup.py install
$ python ./setup.py test
```
