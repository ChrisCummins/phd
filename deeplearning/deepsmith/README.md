# DeepSmith: Compiler Fuzzing through Deep Learning

DeepSmith is a novel approach to automate and accelerate compiler validation
which takes advantage of state-of-the-art deep learning techniques. It
constructs a learned model of the structure of real world code based on a large
corpus of example programs. Then, it uses the model to automatically generate
tens of thousands of realistic programs. Finally, it applies established
differential testing methodologies on them to expose bugs in compilers.

## Usage

First checkout this repo and install the build requirements:

```sh
$ git clone https://github.com/ChrisCummins/phd.git
$ cd phd
$ ./tools/bootstrap.sh
```

Then run the unit tests using:

```sh
$ bazel test //deeplearning/deepsmith/...
```

## License

Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE](/LICENSE) for
details.
