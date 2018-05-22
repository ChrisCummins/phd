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
    name = "atomizers",
    srcs = ["atomizers.py"],
    deps = [
        ":errors",
        ":languages",
    ],
)

py_test(
    name = "atomizers_test",
    srcs = ["atomizers_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":atomizers",
        ":languages",
    ],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    deps = [
        "//lib/labm8:cache",
        "//lib/labm8:fs",
    ],
)

py_binary(
    name = "cli",
    srcs = ["cli.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "atomizers",
        ":cache",
        ":dbutil",
        ":errors",
        ":explore",
        ":features",
        ":fetch",
        ":languages",
        ":model",
        ":preprocess",
        ":sampler",
        "//lib/labm8:fs",
        "//lib/labm8:jsonutil",
        "//lib/labm8:labtypes",
        "//lib/labm8:prof",
    ],
)

py_test(
    name = "cli_test",
    srcs = ["cli_test.py"],
    data = [
        "//deeplearning/clgen/tests/data:pico",
        "//deeplearning/clgen/tests/data:tiny",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli",
        ":conftest",
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
    name = "clgen_config",
    srcs = ["clgen_config.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_library(
    name = "conftest",
    testonly = True,
    srcs = ["conftest.py"],
)

py_library(
    name = "corpus",
    srcs = ["corpus.py"],
    deps = [
        ":atomizers",
        ":cache",
        ":dbutil",
        ":errors",
        ":features",
        ":fetch",
        ":languages",
        ":preprocess",
        "//lib/labm8:crypto",
        "//lib/labm8:dirhashcache",
        "//lib/labm8:fs",
        "//lib/labm8:jsonutil",
        "//lib/labm8:labtypes",
        "//lib/labm8:lockfile",
        "//lib/labm8:prof",
        "//lib/labm8:tar",
        "//lib/labm8:text",
    ],
)

py_test(
    name = "corpus_test",
    srcs = ["corpus_test.py"],
    data = ["//deeplearning/clgen/tests/data:tiny"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":corpus",
        ":errors",
        "//deeplearning/clgen/tests:testlib",
        "//lib/labm8:fs",
    ],
)

py_library(
    name = "dbutil",
    srcs = ["dbutil.py"],
    data = ["//deeplearning/clgen/data/sql:scripts"],
    deps = [
        ":errors",
        ":languages",
        ":package_util",
    ],
)

py_test(
    name = "dbutil_test",
    srcs = ["dbutil_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dbutil",
        ":errors",
        "//deeplearning/clgen/tests:testlib",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
)

py_library(
    name = "explore",
    srcs = ["explore.py"],
    deps = [":dbutil"],
)

py_test(
    name = "explore_test",
    srcs = ["explore_test.py"],
    data = ["//deeplearning/clgen/tests/data:tiny"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":corpus",
        ":explore",
        "//deeplearning/clgen/tests:testlib",
        "//lib/labm8:fs",
    ],
)

py_library(
    name = "features",
    srcs = ["features.py"],
    deps = [
        ":errors",
        ":native",
        ":package_util",
        "//lib/labm8:labmath",
    ],
)

py_library(
    name = "fetch",
    srcs = ["fetch.py"],
    deps = [
        ":dbutil",
        ":errors",
        ":languages",
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
    name = "languages",
    srcs = ["languages.py"],
    deps = [":errors"],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":cache",
        ":corpus",
        ":errors",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
        "//lib/labm8:jsonutil",
        "//lib/labm8:labtypes",
        "//lib/labm8:lockfile",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = ["//deeplearning/clgen/tests/data:tiny"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":model",
        "//deeplearning/clgen/tests:testlib",
    ],
)

py_library(
    name = "native",
    srcs = ["native.py"],
    data = [
        "//deeplearning/clgen/native:clgen-features",
        "//deeplearning/clgen/native:clgen-rewriter",
        "//deeplearning/clgen/data/include:opencl-shim",
        "//third_party/libclc/generic/include:headers",
    ] + select({
        "//:darwin": [
            "@llvm_mac//:clang",
            "@llvm_mac//:clang-format",
            "@llvm_mac//:opt",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang-format",
            "@llvm_linux//:clang",
            "@llvm_linux//:opt",
        ],
    }),
    deps = [
        ":package_util",
        "//config:getconfig",
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
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        ":dbutil",
        ":errors",
        ":languages",
        ":native",
        "//lib/labm8:fs",
    ],
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

py_library(
    name = "sampler",
    srcs = ["sampler.py"],
    deps = [
        ":cache",
        ":dbutil",
        ":errors",
        ":explore",
        ":languages",
        ":model",
        "//lib/labm8:crypto",
        "//lib/labm8:jsonutil",
        "//lib/labm8:labtypes",
        "//lib/labm8:lockfile",
    ],
)

py_test(
    name = "sampler_test",
    srcs = ["sampler_test.py"],
    data = ["//deeplearning/clgen/tests/data:tiny"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":dbutil",
        ":model",
        ":sampler",
        "//deeplearning/clgen/tests:testlib",
    ],
)
