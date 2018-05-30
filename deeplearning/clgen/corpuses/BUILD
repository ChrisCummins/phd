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
        ":features",
        ":fetch",
        ":preprocessed",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:dbutil",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
        "//lib/labm8:hashcache",
        "//lib/labm8:lockfile",
        "//lib/labm8:pbutil",
        "//lib/labm8:prof",
        "//lib/labm8:tar",
        "//lib/labm8:text",
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
        "//deeplearning/clgen/tests:testlib",
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
    name = "explore",
    srcs = ["explore.py"],
    deps = ["//deeplearning/clgen:dbutil"],
)

py_test(
    name = "explore_test",
    srcs = ["explore_test.py"],
    data = [
        "//deeplearning/clgen/tests/data:tiny",
        "//deeplearning/clgen/tests/data/tiny",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":explore",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/corpuses",
        "//deeplearning/clgen/tests:testlib",
        "//lib/labm8:fs",
    ],
)

py_library(
    name = "features",
    srcs = ["features.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:native",
        "//deeplearning/clgen:package_util",
        "//lib/labm8:labmath",
    ],
)

py_library(
    name = "fetch",
    srcs = ["fetch.py"],
    deps = [
        "//deeplearning/clgen:dbutil",
        "//deeplearning/clgen:errors",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
    ],
)

py_test(
    name = "fetch_test",
    srcs = ["fetch_test.py"],
    data = ["//deeplearning/clgen/tests/data:cl_kernels"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fetch",
        "//deeplearning/clgen/tests:testlib",
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
