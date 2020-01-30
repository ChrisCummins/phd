# Package to handle data from HealthKit.
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

py_binary(
    name = "health_kit",
    srcs = ["health_kit.py"],
    data = [":xml_export_worker"],
    visibility = ["//datasets/me_db:__pkg__"],
    deps = [
        "//datasets/me_db:importers",
        "//datasets/me_db:me_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
    ],
)

cc_library(
    name = "health_kit_lib",
    srcs = ["health_kit_lib.cc"],
    hdrs = ["health_kit_lib.h"],
    deps = [
        "//datasets/me_db:me_pb_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "@boost//:filesystem",
        "@boost//:property_tree",
        "@boost//:tokenizer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "health_kit_lib_test",
    srcs = ["health_kit_lib_test.cc"],
    deps = [
        ":health_kit_lib",
        "//datasets/me_db:me_pb_cc",
        "//labm8/cpp:test",
    ],
)

cc_binary(
    name = "xml_export_worker",
    srcs = ["xml_export_worker.cc"],
    deps = [
        ":health_kit_lib",
        "//labm8/cpp:pbutil",
    ],
)
