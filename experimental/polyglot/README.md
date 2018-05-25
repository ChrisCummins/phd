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
    --contentfiles_path=/var/phd/datasets/github/contentfiles.db
```
