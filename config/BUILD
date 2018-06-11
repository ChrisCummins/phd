# Library for accessing the global configuration.

load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "getconfig",
    srcs = ["getconfig.py"],
    data = ["//:config"],
    visibility = ["//visibility:public"],
    deps = [
        "//config/proto:config",
        "//lib/labm8:pbutil",
    ],
)

py_test(
    name = "getconfig_test",
    srcs = ["getconfig_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":getconfig",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
