load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "testing",
    srcs = ["testing.py"],
    deps = [
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith/protos:deepsmith",
        requirement("grpcio"),
        requirement("absl-py"),
    ],
    visibility = ["//deeplearning/deepsmith:__subpackages__"],
)

py_test(
    name = "tests/testing_test",
    srcs = ["tests/testing_test.py"],
    deps = [
        ":testing",
        requirement("pytest"),
        requirement("pytest-benchmark"),
        requirement("absl-py"),
    ]
)
