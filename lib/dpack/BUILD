py_binary(
    name = "dpack",
    srcs = ["dpack.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        "//lib/dpack/proto:dpack_pb2_py",
        "//lib/labm8:crypto",
        "//lib/labm8:fs",
        "//lib/labm8:labdate",
        "//lib/labm8:pbutil",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "dpack_test",
    size = "small",
    srcs = ["dpack_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dpack",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
