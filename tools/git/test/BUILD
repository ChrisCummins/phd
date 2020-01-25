py_library(
    name = "fixtures",
    testonly = 1,
    srcs = ["fixtures.py"],
    visibility = ["//tools/git:__subpackages__"],
    deps = [
        "//labm8/py:fs",
        "//labm8/py:test",
        "//third_party/py/git",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "fixtures_test",
    srcs = ["fixtures_test.py"],
    deps = [
        ":fixtures",
        "//labm8/py:test",
        "//third_party/py/git",
    ],
)
