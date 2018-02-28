py_library(
    name = "testing",
    srcs = ["testing.py"],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith/protos:deepsmith",
    ],
)

py_test(
    name = "tests/testing_test",
    srcs = ["tests/testing_test.py"],
    deps = [
        ":testing",
    ],
)
