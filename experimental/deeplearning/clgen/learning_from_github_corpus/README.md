# Learning Directly from GitHub

**Question**: Given the corpus of OpenCL programs mined from GitHub in [1], how many of
   them could be used to automatically derive training data?

**Answer**:
```sh
$ bazel run //experimental/deeplearning/clgen/learning_from_github_corpus:run_github_corpus
```

To be run...
