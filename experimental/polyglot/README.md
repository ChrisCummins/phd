# Experiments in poly-lingual PL modeling

Extending CLgen and DeepSmith to support multiple programming languages.

## Fetch repos

Index repositories on GitHub:

```sh
$ git rev-parse HEAD
1969647a89e1c56060da20065946f17f1e412855
$ bazel run //datasets/github/scrape_repos:scraper -- \
    --clone_list=$PHD/experimental/polyglot/clone_list.pbtxt

```

Clone repositories:

```sh
$ git rev-parse HEAD
1969647a89e1c56060da20065946f17f1e412855
$ bazel run //datasets/github/scrape_repos:cloner -- \
    --clone_list=$PHD/experimental/polyglot/clone_list.pbtxt
```

Import into a contentfiles database:

```sh
$ git rev-parse HEAD
1969647a89e1c56060da20065946f17f1e412855
$ bazel run //datasets/github/scrape_repos:importer -- \
    --clone_list=$PHD/experimental/polyglot/clone_list.pbtxt
```


## Build generators

Launch a CLgen DeepSmith generator service:

```sh
$ git rev-parse HEAD
b32364f44b8f349df24e524486bc089eb81ebc7d
$ bazel-phd/bazel-out/*-py3-opt/bin/deeplearning/deepsmith/services/clgen \
    --generator_config=$PHD/experimental/polyglot/generators/opencl/256x1x50-a.pbtxt
```
