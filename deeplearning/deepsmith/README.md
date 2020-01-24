# DeepSmith: Compiler Fuzzing through Deep Learning

DeepSmith is a novel approach to automate and accelerate compiler validation
which takes advantage of state-of-the-art deep learning techniques. It
constructs a learned model of the structure of real world code based on a large
corpus of example programs. Then, it uses the model to automatically generate
tens of thousands of realistic programs. Finally, it applies established
differential testing methodologies on them to expose bugs in compilers.

<img src="../../docs/2018_07_issta/img/deepsmith.png" height="500">

## Getting Started

Pull a docker image containing a pre-trained neural network and OpenCL
environment for fuzz testing Oclgrind:

```sh
$ docker run chriscummins/opencl_fuzz
```

See
[//docs/2018_07_issta/artifact_evaluation](/docs/2018_07_issta/artifact_evaluation)
for the supporting artifact of the original DeepSmith publication.

## Resources

Publication
"[Compiler Fuzzing through Deep Learning](https://chriscummins.cc/pub/2018-issta.pdf)"
(ISSTA'18).

## License

Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the GPLv3 license. See [LICENSE](/LICENSE) for
details.
