# Experiments in poly-lingual PL modeling

Extending CLgen and DeepSmith to support multiple programming languages.

## Usage

Select a relevant language:

```sh
$ export POLYGLOT_LANGUAGE="opencl"
```

Prepare a corpus:

```sh
$ bazel build //datasets/github/scrape_repos/...
  
$ $PHD/bazel-phd/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/scraper \
    --clone_list=$PHD/experimental/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt
$ $PHD/bazel-phd/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/cloner \
    --clone_list=$PHD/experimental/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt
$ $PHD/bazel-phd/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/importer \
    --clone_list=$PHD/experimental/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt
$ $PHD/bazel-phd/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/export_corpus \
    --clone_list=$PHD/experimental/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt \
    --export_path=/var/phd/datasets/github/corpuses/
```

Train a CLgen model:

```sh
$ clgen --config=$PHD/experimental/polyglot/$POLYGLOT_LANGUAGE/clgen.pbtxt
```
