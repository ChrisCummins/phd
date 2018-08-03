# This package defines contentfile preprocessors.

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
    deps = [
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "inliners",
    srcs = ["inliners.py"],
    data = select({
        "//:darwin": [
            "@libcxx_mac//:headers",
        ],
        "//conditions:default": [
            "@libcxx_linux//:headers",
        ],
    }),
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":public",
        "//lib/labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/fuzzywuzzy",
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
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        ":inliners",
        ":public",
        "//third_party/py/absl",
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
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
    visibility = ["//datasets/github/scrape_repos:__subpackages__"],
    deps = [
        "//lib/labm8:fs",
        "//third_party/py/absl",
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
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
