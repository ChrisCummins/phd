py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        ":dbutil",
        ":errors",
        ":languages",
        ":native",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//lib/labm8:fs",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_test(
    name = "preprocess_test",
    srcs = ["preprocess_test.py"],
    data = [
        "//deeplearning/clgen/tests/data:cl_kernels",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":preprocess",
        "//deeplearning/clgen/tests:testlib",
    ],
)
