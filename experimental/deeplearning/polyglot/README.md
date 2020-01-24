# Experiments in poly-lingual PL modeling

Extending CLgen and DeepSmith to support multiple programming languages.

### Create a Corpus

Select a relevant language:

```sh
$ export POLYGLOT_LANGUAGE="opencl"
```

Prepare a corpus:

```sh
$ bazel build //datasets/github/scrape_repos/...

$ bazel-bin/datasets/github/scrape_repos/scraper \
    --clone_list experimental/deeplearning/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt
$ bazel-bin/datasets/github/scrape_repos/cloner \
    --clone_list experimental/deeplearning/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt
$ bazel-bin/datasets/github/scrape_repos/importer \
    --clone_list experimental/deeplearning/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt
$ bazel-bin/datasets/github/scrape_repos/export_corpus \
    --clone_list experimental/deeplearning/polyglot/$POLYGLOT_LANGUAGE/clone_list.pbtxt \
    --export_path ~/data/datasets/github/corpuses/
```

### Enumerate and test models

```sh
$ bazel build //experimental/deeplearning/polyglot/...
$ gpu $GPU_ID
$ bazel-bin/experimental/deeplearning/polyglot/run
```

Run a notebook server:

```sh
$ bazel build //experimental/deeplearning/polyglot/notebooks && \
  bazel-bin/experimental/deeplearning/polyglot/notebooks/notebooks
```
