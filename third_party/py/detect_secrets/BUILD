# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "detect_secrets",
    srcs = ["detect_secrets.py"],
    deps = [
        requirement("detect-secrets"),
        requirement("PyYAML"),
    ],
)
