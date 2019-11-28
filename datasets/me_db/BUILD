# me.db - Aggregate personal data from several sources.
#
# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
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

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

licenses(["restricted"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "importers",
    srcs = ["importers.py"],
    visibility = ["//datasets/me_db:__subpackages__"],
    deps = [
        ":me_pb_py",
        "//labm8/py:app",
        "//labm8/py:ppar",
    ],
)

py_binary(
    name = "me_db",
    srcs = ["me_db.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/me_db/providers/health_kit",
        "//datasets/me_db/providers/life_cycle",
        "//datasets/me_db/providers/timing",
        "//datasets/me_db/providers/ynab",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

proto_library(
    name = "me_pb",
    srcs = ["me.proto"],
)

python_proto_library(
    name = "me_pb_py",
    visibility = ["//datasets/me_db:__subpackages__"],
    deps = [":me_pb"],
)

cc_proto_library(
    name = "me_pb_cc",
    visibility = ["//datasets/me_db:__subpackages__"],
    deps = [":me_pb"],
)

py_library(
    name = "omnifocus",
    srcs = ["omnifocus.py"],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "spreadsheet",
    srcs = ["spreadsheet.py"],
    deps = [
        "//third_party/py/gspread",
        "//third_party/py/oauth2client",
    ],
)

py_library(
    name = "toggl_import",
    srcs = ["toggl_import.py"],
    deps = [
        ":utils",
        "//third_party/py/toggl",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)
