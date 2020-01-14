# Testing utilities for //deeplearning/clgen.

py_library(
    name = "fixtures",
    testonly = True,
    srcs = ["fixtures.py"],
    visibility = [
        "//deeplearning/clgen:__subpackages__",
        "//docs/2018_07_issta:__subpackages__",
    ],
    deps = [
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:pbutil",
        "//labm8/py:test",
    ],
)
