# This package implements corpuses for training CLgen.

load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "atomizers",
    srcs = ["atomizers.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//deeplearning/clgen:errors",
        "//lib/labm8:labdate",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "atomizers_test",
    srcs = ["atomizers_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":atomizers",
        requirement("absl-py"),
        requirement("pytest"),
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
        requirement("absl-py"),
        requirement("checksumdir"),
        requirement("humanize"),
        requirement("numpy"),
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
        requirement("absl-py"),
        requirement("pytest"),
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
        requirement("numpy"),
        requirement("progressbar2"),
        requirement("python-utils"),
        requirement("absl-py"),
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
        requirement("absl-py"),
        requirement("pytest"),
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
        requirement("numpy"),
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
        requirement("absl-py"),
        requirement("progressbar2"),
        requirement("sqlalchemy"),
    ],
)
