# Integration tests that verify the values imported by me.db after processing
# the //util/me/tests/test_inbox dataset.

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
    data = ["//util/me/tests:test_inbox_files"],
    testonly = 1,
)

py_test(
    name = "life_cycle_integration_test",
    srcs = ["life_cycle_integration_test.py"],
    deps = [
        ":conftest",
        "//util/me",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ]
)
