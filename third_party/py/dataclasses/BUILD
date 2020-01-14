# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "dataclasses",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        # Data classes have been implemented in python >= 3.7, and this package
        # breaks on those python versions. The workaround would be to add
        # a select on this requirement which checks that py3.7 is available,
        # use include it. I'm not yet sure what the best way to add this dynamic
        # behaviour is so for now I've hardcoded the assumption that py >= 3.7.
        # requirement("dataclasses"),
    ],
)
