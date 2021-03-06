# Acceptance tests that perform dataset-agnostic, high level tests of database
# after importing a given dataset. By default, the
# //datasets/me_db/tests/test_inbox inbox is used, but this can be overriden
# using flag --me_db_acceptance_tests_inbox=<path>.
#
# Example:
#
#     bazel test //datasets/me_db/tests/acceptance_tests --test_output=streamed \
#         --test_flag=--me_db_acceptance_tests_inbox=/path/to/inbox
#
# Copyright 2018-2020 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

py_test(
    name = "acceptance_test",
    srcs = ["acceptance_test.py"],
    data = ["//datasets/me_db/tests:test_inbox_files"],
    deps = [
        ":flags",
        "//datasets/me_db",
        "//labm8/py:app",
        "//labm8/py:labdate",
        "//labm8/py:test",
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
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
