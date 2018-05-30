# This package implements corpuses for training CLgen.

py_library(
    name = "atomizers",
    srcs = ["atomizers.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//lib/labm8:labdate",
    ],
)

py_test(
    name = "atomizers_test",
    srcs = ["atomizers_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":atomizers",
    ],
)

py_library(
    name = "corpuses",
    srcs = ["corpuses.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":atomizers",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:hashcache",
        "//lib/labm8:lockfile",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "corpuses_test",
    srcs = ["corpuses_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":corpuses",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors:common",
        "//deeplearning/clgen/preprocessors:cxx",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//lib/labm8:fs",
    ],
)

py_library(
    name = "encoded",
    srcs = ["encoded.py"],
    deps = [
        ":preprocessed",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//lib/labm8:sqlutil",
    ],
)

py_test(
    name = "encoded_test",
    srcs = ["encoded_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":atomizers",
        ":encoded",
        ":preprocessed",
    ],
)

py_library(
    name = "features",
    srcs = ["features.py"],
    data = [
        "//deeplearning/clgen/data/include:opencl-shim",
        "//deeplearning/clgen/native:clgen-features",
    ],
    deps = [
        "//config:getconfig",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:package_util",
        "//lib/labm8:bazelutil",
        "//lib/labm8:labmath",
    ],
)

py_library(
    name = "preprocessed",
    srcs = ["preprocessed.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
        "//lib/labm8:sqlutil",
    ],
)
