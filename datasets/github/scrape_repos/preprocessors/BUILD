# This package defines contentfile preprocessors.

load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
    deps = [
        requirement("pytest"),
    ],
)

py_library(
    name = "inliners",
    srcs = ["inliners.py"],
    data = ["@libcxx//:headers"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":public",
        "//lib/labm8:bazelutil",
        requirement("absl-py"),
        requirement("fuzzywuzzy"),
    ],
)

py_test(
    name = "inliners_test",
    size = "small",
    srcs = ["inliners_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":inliners",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":inliners",
        ":public",
        requirement("absl-py"),
    ],
)

py_test(
    name = "preprocessors_test",
    size = "small",
    srcs = ["preprocessors_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":preprocessors",
        ":public",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        "//lib/labm8:fs",
        requirement("absl-py"),
    ],
)

py_test(
    name = "public_test",
    size = "small",
    srcs = ["public_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conftest",
        ":public",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)
