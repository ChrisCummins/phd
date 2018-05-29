# Java

Prepare corpus:

```sh
$ bazel run //datasets/github/scrape_repos:scraper -- \
    --clone_list=$PHD/experimental/polyglot/java/clone_list.pbtxt
$ bazel run //datasets/github/scrape_repos:cloner -- \
    --clone_list=$PHD/experimental/polyglot/java/clone_list.pbtxt
$ bazel run //datasets/github/scrape_repos:importer -- \
    --clone_list=$PHD/experimental/polyglot/java/clone_list.pbtxt
$ bazel run //datasets/github/scrape_repos:export_corpus -- \
    --clone_list=$PHD/experimental/polyglot/java/clone_list.pbtxt \
    --export_path=/var/phd/datasets/github/corpuses/
```

Train CLgen model:

```sh
$ clgen --config=$PHD/experimental/polyglot/java/clgen.pbtxt
```
