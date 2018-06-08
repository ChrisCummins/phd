# This package defines contentfile preprocessors.

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
)

py_library(
    name = "inliners",
    srcs = ["inliners.py"],
    data = ["@libcxx//:headers"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":public",
        "//lib/labm8:bazelutil",
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
    ],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":inliners",
        ":public",
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
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        "//lib/labm8:fs",
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
    ],
)
