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
    name = "test_config",
    srcs = ["test_config.py"],
    data = [
        ":tiny",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//deeplearning/clgen",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
    ],
)
