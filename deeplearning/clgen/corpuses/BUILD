# This package implements corpuses for training CLgen.

py_library(
    name = "atomizers",
    srcs = ["atomizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen:errors",
        "//labm8:labdate",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "atomizers_test",
    srcs = ["atomizers_test.py"],
    deps = [
        ":atomizers",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "corpuses",
    srcs = ["corpuses.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":atomizers",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//labm8:bazelutil",
        "//labm8:crypto",
        "//labm8:hashcache",
        "//labm8:lockfile",
        "//labm8:pbutil",
        "//third_party/py/absl",
        "//third_party/py/checksumdir",
        "//third_party/py/humanize",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "corpuses_test",
    srcs = ["corpuses_test.py"],
    deps = [
        ":corpuses",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors:common",
        "//deeplearning/clgen/preprocessors:cxx",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//labm8:fs",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "encoded",
    srcs = ["encoded.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":preprocessed",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//labm8:sqlutil",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/numpy",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "encoded_test",
    srcs = ["encoded_test.py"],
    deps = [
        ":atomizers",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen:errors",
        "//labm8:test",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "preprocessed",
    srcs = ["preprocessed.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//labm8:crypto",
        "//labm8:fs",
        "//labm8:sqlutil",
        "//third_party/py/absl",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)
