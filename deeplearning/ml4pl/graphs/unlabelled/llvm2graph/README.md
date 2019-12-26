# Program Graphs from LLVM IR

This package contains a tool for constructing program graphs from LLVM IR
files.


## Using docker

A docker image is available:

```
$ docker pull chriscummins/llvm2graph:latest
```

Run the image by passing in an IR file to stdin:

```
$ docker run -i chriscummins/llvm2graph < /tmp/foo.ll
```

The tool will print a program graph protocol buffer to stdout.


## Using bazel

The bazel equivalent to the docker invocation above is:

```
$ bazel run //deeplearning/ml4pl/graphs/unlabelled/llvm2graph -- < /tmp/foo.ll
```
