# DeepSmith: Compiler Fuzzing through Deep Learning


## Requirements

* GNU / Linux (we recommend Ubuntu 16.04).
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
