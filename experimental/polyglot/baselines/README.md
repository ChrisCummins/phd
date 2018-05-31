# CLgen Baseline models

## Usage

Build:

```sh
$ bazel build //experimental/polyglot/baselines:run
```

Run:

```sh
$ bazel-out/*-py3-opt/bin/experimental/polyglot/baselines/run \
  --corpus experimental/polyglot/baselines/corpuses/opencl-char.pbtxt \
  --model experimental/polyglot/baselines/models/512x2x50-adam.pbtxt \
  --sampler experimental/polyglot/baselines/samplers/opencl-1.0.pbtxt
```
