# Acceptance tests that perform dataset-agnostic, high level tests of database
# after importing a given dataset. By default, the
# //datasets/me_db/tests/test_inbox inbox is used, but this can be overriden
# using flag --me_db_acceptance_tests_inbox=<path>.
#
# Example:
#
#     bazel test //datasets/me_db/tests/acceptance_tests --test_output=streamed \
#         --test_flag=--me_db_acceptance_tests_inbox=/path/to/inbox

py_test(
    name = "acceptance_test",
    srcs = ["acceptance_test.py"],
    data = ["//datasets/me_db/tests:test_inbox_files"],
    deps = [
        ":flags",
        "//datasets/me_db",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:test",
        "//third_party/py/pandas",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "flags",
    testonly = 1,
    srcs = ["flags.py"],
    deps = [
        "//datasets/me_db",
        "//labm8:app",
        "//labm8:test",
    ],
)
