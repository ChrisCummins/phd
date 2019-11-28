py_binary(
    name = "run_github_corpus",
    srcs = ["run_github_corpus.py"],
    deps = [
        "//deeplearning/clgen/corpuses",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//deeplearning/deepsmith/proto:service_pb_py",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:labtypes",
        "//third_party/py/pandas",
    ],
)

sh_test(
    name = "run_github_corpus_smoke_test",
    srcs = ["run_github_corpus_smoke_test.sh"],
    data = [":run_github_corpus"],
)
