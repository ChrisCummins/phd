# This package contains CLgen, a deep learning program generator.
#
# CLgen is an open source application for generating runnable programs using
# deep learning. CLgen learns to program using neural networks which model the
# semantics and usage from large volumes of program fragments, generating
# executable programs that are representative of, but distinct from,
# the programs it learns from.

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "cache",
    srcs = ["cache.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//lib/labm8:cache",
        "//lib/labm8:fs",
    ],
)

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":samplers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//lib/labm8:pbutil",
        "//lib/labm8:prof",
    ],
)

py_test(
    name = "clgen_test",
    size = "enormous",
    srcs = ["clgen_test.py"],
    data = [
        "//deeplearning/clgen/tests/data:pico",
        "//deeplearning/clgen/tests/data:tiny",
        "//deeplearning/clgen/tests/data/tiny",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clgen",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/tests:testlib",
        "//lib/labm8:fs",
        "//lib/labm8:tar",
    ],
)

py_library(
    name = "clutil",
    srcs = ["clutil.py"],
    deps = [
        "//lib/labm8:text",
    ],
)

py_library(
    name = "conftest",
    testonly = True,
    srcs = ["conftest.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":dbutil",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
    ],
)

py_library(
    name = "dbutil",
    srcs = ["dbutil.py"],
    data = ["//deeplearning/clgen/data/sql:scripts"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":errors",
        ":package_util",
    ],
)

py_test(
    name = "dbutil_test",
    srcs = ["dbutil_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":dbutil",
        ":errors",
        "//deeplearning/clgen/tests:testlib",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_library(
    name = "native",
    srcs = ["native.py"],
    data = [
        "//deeplearning/clgen/native:clgen-features",
        "//deeplearning/clgen/native:clgen-rewriter",
        "//deeplearning/clgen/data/include:opencl-shim",
        "//deeplearning/clgen/data/include:opencl",
        "//third_party/libclc/generic/include:headers",
        "@libcxx//:headers",
    ] + select({
        "//:darwin": [
            "@llvm_mac_headers//:headers",
            "@llvm_mac//:clang",
            "@llvm_mac//:clang-format",
            "@llvm_mac//:opt",
        ],
        "//conditions:default": [
            "@llvm_linux_headers//:headers",
            "@llvm_linux//:clang-format",
            "@llvm_linux//:clang",
            "@llvm_linux//:opt",
        ],
    }),
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":package_util",
        "//config:getconfig",
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
    ],
)

py_test(
    name = "native_test",
    srcs = ["native_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":native",
        "//lib/labm8:fs",
    ],
)

py_library(
    name = "package_util",
    srcs = ["package_util.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        ":errors",
        "//lib/labm8:fs",
    ],
)

py_test(
    name = "package_util_test",
    srcs = ["package_util_test.py"],
    data = [
        "//deeplearning/clgen/data/include:opencl-shim",
        "//deeplearning/clgen/data/sql:scripts",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":package_util",
    ],
)

py_library(
    name = "samplers",
    srcs = ["samplers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cache",
        ":errors",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//lib/labm8:crypto",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "samplers_test",
    srcs = ["samplers_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":errors",
        ":samplers",
    ],
)

py_library(
    name = "telemetry",
    srcs = ["telemetry.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//lib/labm8:jsonutil",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":telemetry",
    ],
)
