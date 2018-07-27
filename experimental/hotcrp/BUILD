# Scripts for working with HotCRP.

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "json_to_acm_proceedings",
    srcs = ["json_to_acm_proceedings.py"],
    deps = [
        requirement("absl-py"),
    ],
)

py_test(
    name = "json_to_acm_proceedings_test",
    srcs = ["json_to_acm_proceedings_test.py"],
    deps = [
        ":json_to_acm_proceedings",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
