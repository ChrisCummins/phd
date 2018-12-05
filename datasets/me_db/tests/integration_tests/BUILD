# Integration tests that verify the values imported by me.db after processing
# the //datasets/me_db/tests/test_inbox dataset.

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    data = ["//datasets/me_db/tests:test_inbox_files"],
)

py_test(
    name = "life_cycle_integration_test",
    srcs = ["life_cycle_integration_test.py"],
    deps = [
        ":conftest",
        "//datasets/me_db",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
