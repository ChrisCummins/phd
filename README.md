# Plausible Test Case Generation for Compiler and Runtime Bug Discovery


## Requirements

* Ubuntu 16.04
* Python 3.6
* OpenCL
* ninja
* CMake >= 3.4.3

Optional, but recommended:

* Nvidia GPU and CUDA

## Installation

**Step 1:** Checkout the source.

```sh
$ git submodule update --init --recursive
```

**Step 2:** Build the code.

If CUDA is available:

```sh
$ make all
```

If CUDA is *not* available:

```sh
$ NO_CUDA=1 make all
```

Note our software stack is pretty large, requiring around 7 GB of dependencies to be built, including:
* TensorFlow
* CSmith (two different versions)
* oclgrind (two different versions)
* LLVM (two different versions)
* CLreduce
* Various python and perl packages

A clean build may take upwards of an hour.

## Running the code

Launch the Jupyter server:

```
$ make run
```
