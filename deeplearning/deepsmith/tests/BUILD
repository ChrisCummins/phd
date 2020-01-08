# Testing utilities for //deeplearning/deepsmith.

py_library(
    name = "fixtures",
    testonly = 1,
    srcs = ["fixtures.py"],
    data = glob(["data/datastores/*"]),
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//labm8/py:test",
    ],
)
