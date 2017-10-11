# DeepSmith: Compiler Fuzzing through Deep Learning

DeepSmith is a novel approach to automate and accelerate compiler validation which takes advantage of state-of-the-art deep learning techniques. It constructs a learned model of the structure of real world code based on a large corpus of example programs. Then, it uses the model to automatically generate tens of thousands of realistic programs. Finally, it applies established differential testing methodologies on them to expose bugs in compilers.

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


## Usage

DeepSmith is installed in a virtual environment. Activate it using:

```sh
$ source build/dsmith/bin/activate
```

When you are done using DeepSmith, you can deactivate the virtual environment using:

```sh
(dsmith) $ deactivate
```

Testing compilers using DeepSmith is a three step process:
1. **Generate testcases** using a random program generator
1. **Evaluate testcases** by executing them on multiple devices
1. **Differential test** results by comparing the results across devices

#### 1. Generate testcases

Generate testcases using CLSmith:

```sh
(dsmith) $ ./clsmith_mkprogram.py -n 1000
```

Generate testcases using DeepSmith:

```sh
# train and sample model
(dsmith) $ clgen sample model.json sampler.json
# export samples
(dsmith) $ clgen db dump $(clgen --sampler-dir model.json sampler.json)/kernels.json -d /tmp/export
# import samples
(dsmith) $ ./clgen_fetch.py /tmp/export --delete
```

#### 2. Evaluate testcases

Collect results for an OpenCL device:

```sh
(dsmith) $ ./runner.py [--verbose] [--only <testbed_ids>] [--exclude <testbed_ids>] [--batch-size <int>]
```

#### 3. Differential testing results

Prepare results for analysis:

```sh
(dsmith) $ ./set_metas.py
```

Analyze results:

```sh
(dsmith) $ ./analyze.py [--prune]
```


##### 3.1 Reduce interesting testcases

Run automated reductions:

```sh
(dsmith) $ ./run_reductions 0 0 [--clgen|--clsmith]
```

##### 3.2 Prepare interesting testcases for reports

Generate bug reports:

```sh
(dsmith) $ ./report.py
```

Submit bug reports by hand.

## License

Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
