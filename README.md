<div align="center">
  <a href="https://github.com/ChrisCummins/clgen">
    <img src="https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/logo.png" width="420">
  </a>
</div>

-------

<div align="center">
  <a href="http://chriscummins.cc/clgen/" target="_blank">
    <img src="https://img.shields.io/badge/docs-0.4.0.dev0-brightgreen.svg?style=flat">
  </a>
  <a href="https://travis-ci.org/ChrisCummins/clgen" target="_blank">
    <img src="https://img.shields.io/travis/ChrisCummins/clgen/master.svg?style=flat">
  </a>
  <a href="https://coveralls.io/github/ChrisCummins/clgen?branch=master">
    <img src="https://img.shields.io/coveralls/ChrisCummins/clgen/master.svg?style=flat">
  </a>
   <a href="https://github.com/ChrisCummins/clgen/releases" target="_blank">
    <img src="https://img.shields.io/badge/release-0.4.0.dev0-blue.svg?style=flat">
  </a>
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
    <img src="https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg?style=flat">
  </a>
</div>

**CLgen** is an open source application for generating runnable programs using
deep learning. CLgen *learns* to program using neural networks which model the
semantics and usage from large volumes of program fragments, generating
many-core OpenCL programs that are representative of, but *distinct* from, the
programs it learns from.

<img src="https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/pipeline.png" width="500">


## Getting Started

See the [online documentation](http://chriscummins.cc/clgen/) for instructions
on how to download and install CLgen.

Download a tiny example dataset to train and sample your first CLgen model:

```sh
$ wget https://github.com/ChrisCummins/clgen/raw/master/tests/data/tiny.tar.bz2
$ tar xf tiny.tar.bz2
$ clgen sample model.json sampler.json
```

<img src="https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/clgen.gif" width="500">


## Resources

Presentation slides:

<a href="https://speakerdeck.com/chriscummins/synthesizing-benchmarks-for-predictive-modelling-cgo-17">
  <img src="https://raw.githubusercontent.com/ChrisCummins/clgen/master/docs/assets/slides.png" width="500">
</a>

Publication
["Synthesizing Benchmarks for Predictive Modeling"](https://github.com/ChrisCummins/paper-synthesizing-benchmarks)
(CGO'17).

[Jupyter notebook](https://github.com/ChrisCummins/paper-synthesizing-benchmarks/blob/master/code/Paper.ipynb) containing experimental evaluation of
CLgen.

Documentation for the [Python API](http://chriscummins.cc/clgen/api/) and
[command line interface](http://chriscummins.cc/clgen/bin/).


## License

Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE.txt](/LICENSE.txt)
for details.
