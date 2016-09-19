# Deep Learning OpenCL Program Generator
[![Build Status](https://travis-ci.com/ChrisCummins/clgen.svg?token=RpzWC2nNxou66YeqVQYw&branch=master)](https://travis-ci.com/ChrisCummins/clgen)

CLgen is the world's first entirely probabilistic program generator. It
*learns* to program using neural networks which model the semantics and usage
from large volumes of program fragments, generating many-core
OpenCL programs that are representative of, but *distinct* from, the programs
it learns from.


## Getting started

```
TODO: build and install command
clgen model.json arguments.json
```

The first command builds and installs clgen. The second command will invoke the
full clgen preprocessing, training, and sampling pipeline on a small included
training set.


## Requirements

* A powerful machine (you *can* run this on your laptop, but I recommend
  something with a bit more horsepower and a
  [modern NVIDIA GPU](http://www.geforce.com/hardware/10series/titan-x-pascal)).
* [OpenCL](https://www.khronos.org/opencl/) `== 1.2`.
* [CUDA](http://www.nvidia.com/object/cuda_home_new.html) `>= 7.5`.
* [clang](http://llvm.org/releases/download.html) `>= 3.6`.
* TODO: build system


## License

Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
