# Package to handle data from YNAB (You Need A Budget).
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

cc_binary(
    name = "json_budget_worker",
    srcs = ["json_budget_worker.cc"],
    deps = [
        "//datasets/me_db:me_proto_cc",
        "//phd:logging",
        "//phd:pbutil",
        "//phd:string",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

py_binary(
    name = "make_dataset",
    testonly = 1,
    srcs = ["make_dataset.py"],
    deps = [
        "//labm8:app",
        "//labm8:jsonutil",
    ],
)

py_binary(
    name = "ynab",
    srcs = ["ynab.py"],
    data = [":json_budget_worker"],
    visibility = ["//datasets/me_db:__pkg__"],
    deps = [
        "//datasets/me_db:importers",
        "//datasets/me_db:me_proto_pb2",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "ynab_test",
    srcs = ["ynab_test.py"],
    deps = [
        ":make_dataset",
        ":ynab",
        "//datasets/me_db:me_proto_pb2",
        "//labm8:app",
        "//labm8:test",
    ],
)
