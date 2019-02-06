py_binary(
    name = "run_github_corpus",
    srcs = ["run_github_corpus.py"],
    deps = [
        "//deeplearning/clgen/corpuses",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//labm8:labtypes",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/pandas",
    ],
)

sh_test(
    name = "run_github_corpus_smoke_test",
    srcs = ["run_github_corpus_smoke_test.sh"],
    data = [":run_github_corpus"],
)
