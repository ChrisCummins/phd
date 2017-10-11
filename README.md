# DeepSmith: Compiler Fuzzing through Deep Learning

DeepSmith is a novel approach to automate and accelerate compiler validation which takes advantage of state-of-the-art deep learning techniques. It constructs a learned model of the structure of real world code based on a large corpus of example programs. Then, it uses the model to automatically generate tens of thousands of realistic programs. Finally, it applies established differential testing methodologies on them to expose bugs in compilers.

## Requirements

* GNU / Linux (we recommend Ubuntu 16.04).
* Clang
* OpenCL
* GNU Make

Optional, but recommended:

* Nvidia GPU with [CUDA Toolkit 8.0 and cuDNN v6](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A).

## Installation

**Step 1:** Configure the build.

```sh
$ ./configure
```

You will be asked a series of questions to customize the build.

**Step 2:** Build the code.

```sh
$ make
```

Please note our software stack is pretty large, requiring around 7 GB of dependencies to be built. A clean build may take upwards of an hour.

**Step 3 (optional):** Run the test suite.

```sh
$ make test
```

Please report any problems using the [issue tracker](https://github.com/ChrisCummins/dsmith/issues).


## License

Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
