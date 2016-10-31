# Detailed Build Instructions

## Configuration

The configuration stage is used to enable features of CLgen:

```sh
./configure
```

Answer the yes/no prompts to reflect your system. OpenCL is required to
dynamically check and execute generated OpenCL programs. CUDA is required to
build CLgen models on the GPU.


## Native File Compilation

CLgen is written primarily in Python, but requires numerous components to be
compiled to native code. To compile these components, run:

```sh
$ make all
```

The program rejection and transformation pipeline requires the LLVM and Clang
libraries and binaries. These are built in three steps:

The ninja build tool is downloaded and compiled in `native/ninja`:

```sh
$ make ninja
```

A precompiled version of CMake is downloaded to `native/cmake`:

```sh
$ make cmake
```

The clang compiler is downloaded and compiled in in `native/llvm`:

```
$ make llvm
```

Once the LLVM toolchain has been compiled, the remainder of the native files
are built roughly as follows:

Libclc provides headers for a generic OpenCL implementation:

```sh
$ make libclc
```

The machine learning is implemented using torch:

```
$ make torch
```

Torch-rnn provides the recursive neural network:

```
$ make torch-rnn
```

(note there are other native files which are only compiled when running `make all`).

## Python Package Installation

Once the native components are compiled, the remainder of the build process
involves installing the python package:

```sh
$ sudo make install
```

This is roughly equivalent to:

```sh
$ pip install numpy>=1.10.4
$ pip install -r requirements.txt
$ sudo python ./setup.py install
$ python ./setup.py test
```
