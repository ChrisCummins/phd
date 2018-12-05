# Acceptance tests that perform dataset-agnostic, high level tests of database
# after importing a given dataset. By default, the //util/me/tests/test_inbox
# inbox is used, but this can be overriden using flag
# --medb_acceptance_tests_inbox=<path>.
#
# Example:
#
#     bazel test //util/me/tests/acceptance_tests --test_output=streamed \
#         --test_flag=--medb_acceptance_tests_inbox=/path/to/inbox

py_test(
    name = "acceptance_tests",
    srcs = ["acceptance_tests.py"],
    data = ["//util/me/tests:test_inbox_files"],
    deps = [
        ":flags",
        "//lib/labm8:labdate",
        "//third_party/py/absl",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
        "//third_party/py/sqlalchemy",
        "//util/me",
    ],
)

py_library(
    name = "flags",
    srcs = ["flags.py"],
    deps = [
        "//third_party/py/absl",
        "//third_party/py/pytest",
        "//util/me",
    ],
)
