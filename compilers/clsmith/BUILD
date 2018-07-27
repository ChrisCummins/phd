# A python wrapper around CLSmith, a random generator of OpenCL C programs.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "clsmith",
    srcs = ["clsmith.py"],
    data = ["@CLSmith"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//lib/labm8:bazelutil",
        requirement("absl-py"),
    ],
)

py_test(
    name = "clsmith_test",
    srcs = ["clsmith_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clsmith",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
