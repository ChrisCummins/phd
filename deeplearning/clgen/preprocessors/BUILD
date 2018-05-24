# This package defines the preprocessor pipeline for CLgen.

py_library(
    name = "clang",
    srcs = ["clang.py"],
    deps = [
        "//config:getconfig",
        "//deeplearning/clgen:dbutil",
        "//deeplearning/clgen:errors",
    ],
)

py_test(
    name = "clang_test",
    srcs = ["clang_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clang",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "//config:getconfig",
        "//deeplearning/clgen:errors",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clang",
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    deps = [
        ":clang",
        "//config:getconfig",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:native",
    ],
)

py_test(
    name = "opencl_test",
    srcs = ["opencl_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clang",
        ":opencl",
        "//deeplearning/clgen:native",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/tests:testlib",
    ],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":clang",
        "//deeplearning/clgen:dbutil",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:languages",
        "//deeplearning/clgen:native",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//lib/labm8:fs",
    ],
)

py_test(
    name = "preprocessors_test",
    srcs = ["preprocessors_test.py"],
    data = [
        "//deeplearning/clgen/tests/data:cl_kernels",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":opencl",
        ":preprocessors",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/tests:testlib",
    ],
)
