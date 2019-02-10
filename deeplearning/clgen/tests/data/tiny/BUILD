# This package contains the data for a tiny CLgen instance.

filegroup(
    name = "tiny",
    srcs = [
        "config.pbtxt",
        "corpus.tar.bz2",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    data = [":tiny"],
    deps = [
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)
