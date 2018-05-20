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
    name = "errors",
    srcs = ["errors.py"],
)

py_library(
    name = "languages",
    srcs = ["languages.py"],
    deps = [":errors"],
)

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
    name = "native",
    srcs = ["native.py"],
    deps = [
        ":package_util",
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
