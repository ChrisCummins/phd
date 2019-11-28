# A tool for creating structured packages of data files.

load("@subpar//:subpar.bzl", "par_binary")

par_binary(
    name = "dpack",
    srcs = ["dpack.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:labdate",
        "//labm8/py:pbutil",
        "//system/dpack/proto:dpack_pb_py",
    ],
)

py_test(
    name = "dpack_test",
    size = "small",
    srcs = ["dpack_test.py"],
    deps = [
        ":dpack",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [":dpack.par"],
)
