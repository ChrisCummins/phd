# This package implements corpuses for training CLgen.

load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "atomizers",
    srcs = ["atomizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen:errors",
        "//lib/labm8:labdate",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "atomizers_test",
    srcs = ["atomizers_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":atomizers",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
        "//lib/labm8:bazelutil",
        "//lib/labm8:crypto",
        "//lib/labm8:hashcache",
        "//lib/labm8:lockfile",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
        requirement("checksumdir"),
        requirement("humanize"),
        "//third_party/py/numpy",
        requirement("sqlalchemy"),
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
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
        requirement("humanize"),
        "//third_party/py/numpy",
        requirement("progressbar2"),
        requirement("python-utils"),
        "//third_party/py/absl",
        requirement("sqlalchemy"),
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
        "//deeplearning/clgen:errors",
        "//third_party/py/absl",
        "//third_party/py/pytest",
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
        "//deeplearning/clgen:errors",
        "//lib/labm8:bazelutil",
        "//lib/labm8:labmath",
        "//third_party/py/numpy",
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
        "//third_party/py/absl",
        requirement("progressbar2"),
        requirement("python-utils"),
        requirement("sqlalchemy"),
    ],
)
