# This package defines contentfile preprocessors.

py_library(
    name = "inliners",
    srcs = ["inliners.py"],
    data = ["@libcxx//:headers"],
    deps = [
        ":public",
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
    ],
)

py_test(
    name = "inliners_test",
    size = "small",
    srcs = ["inliners_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [":inliners"],
)

py_library(
    name = "preprocessors",
    srcs = ["preprocessors.py"],
)

py_test(
    name = "preprocessors_test",
    size = "small",
    srcs = ["preprocessors_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":preprocessors",
        ":public",
    ],
)

py_library(
    name = "public",
    srcs = ["public.py"],
)
