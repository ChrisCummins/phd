# DeepSmith: Compiler Fuzzing through Deep Learning


## Requirements

* GNU / Linux (we recommend Ubuntu 16.04).
* Python 3.6
* OpenCL
* GNU Make

Optional, but recommended:

* Nvidia GPU with [CUDA Toolkit 8.0 and cuDNN v6](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A).

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


## License

Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
