# Plausible Test Case Generation for Compiler and Runtime Bug Discovery


## Requirements

* Ubuntu 16.04
* Python 3.6
* OpenCL
* ninja
* CMake >= 2.8.12

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


## Running the code

Launch the Jupyter server:

```
$ make run
```
