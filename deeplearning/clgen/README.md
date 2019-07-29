<div align="center">
  <a href="https://github.com/ChrisCummins/phd/tree/master/deeplearning/clgen">
    <img src="https://raw.githubusercontent.com/ChrisCummins/phd/master/deeplearning/clgen/docs/assets/logo.png" width="420">
  </a>
</div>

-------

<div align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
    <img src="https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat">
  </a>
</div>

**CLgen** is an open source application for generating runnable programs using
deep learning. CLgen *learns* to program using neural networks which model the
semantics and usage from large volumes of program fragments, generating
many-core OpenCL programs that are representative of, but *distinct* from, the
programs it learns from.

<img src="https://raw.githubusercontent.com/ChrisCummins/phd/master/deeplearning/clgen/docs/assets/pipeline.png" width="500">


## Getting Started

Configure the build and answer the yes/no questions. The default answers should
be fine:

```sh
$ ./configure
```

Note that CUDA support requires CUDA to have been installed separately,
see the [TensorFlow build docs](https://www.tensorflow.org/install/) for
instructions. CUDA support has only been tested for Linux builds, not macOS or
Docker containers.

```sh
$ bazel build //deeplearning/clgen
```

The configure process generates a `bootstrap.sh` script which will install the
required dependent packages. Since installing these packages will affect the
global state of your system, and may requires root access, inspect this script
carefully. Once you're happy to proceed, run it using:

```sh
$ bash ./bootstrap.sh
```

Finally, we must set up the shell environment for running bazel. The file `.env`
is created by the configure process and must be sourced for every shell we want
to use bazel with:

```sh
$ source $PWD/.env
```

Use our tiny example dataset to train and sample your first CLgen model:

```sh
$ bazel run //deeplearning/clgen -- \
    --config $PWD/deeplearning/clgen/tests/data/tiny/config.pbtxt
```

<img src="https://raw.githubusercontent.com/ChrisCummins/phd/master/deeplearning/clgen/docs/assets/clgen.gif" width="500">


#### What next?

CLgen is a tool for generating source code. How you use it will depend entirely
on what you want to do with the generated code. As a first port of call I'd 
recommend checking out how CLgen is configured. CLgen is configured through a 
handful of 
[protocol buffers](https://developers.google.com/protocol-buffers/) defined in
[//deeplearning/clgen/proto](/deeplearning/clgen/proto). 
The [clgen.Instance](/deeplearning/clgen/proto/clgen.proto) message type
combines a [clgen.Model](/deeplearning/clgen/proto/model.proto) and 
[clgen.Sampler](/deeplearning/clgen/proto/sampler.proto) which define the
way in which models are trained, and how new programs are generated, 
respectively. You will probably want to assemble a large corpus of source code 
to train a new model on - I have [tools](/datasets/github/scrape_repos) which 
may help with that. You may also want a means to execute arbitrary generated 
code - as it happens I have [tools](/gpu/cldrive) for that too. :-) Thought of a 
new use case? I'd love to hear about it!


## Resources

Presentation slides:

<a href="https://speakerdeck.com/chriscummins/synthesizing-benchmarks-for-predictive-modelling-cgo-17">
  <img src="https://raw.githubusercontent.com/ChrisCummins/phd/master/deeplearning/clgen/docs/assets/slides.png" width="500">
</a>

Publication
["Synthesizing Benchmarks for Predictive Modeling"](https://github.com/ChrisCummins/paper-synthesizing-benchmarks)
(CGO'17).

[Jupyter notebook](https://github.com/ChrisCummins/paper-synthesizing-benchmarks/blob/master/code/Paper.ipynb)
containing experimental evaluation of an early version of CLgen.

My documentation sucks. Don't be afraid to get stuck in and start 
[reading the code!](deeplearning/clgen/clgen.py)

## License

Copyright 2016, 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See
[LICENSE](/deeplearning/clgen/LICENSE) for details.
