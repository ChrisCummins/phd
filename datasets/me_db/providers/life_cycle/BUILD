# Package to handle data from Life Cycle.
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

# lc_export_csv_worker rule split into two, since cc_test can only depend on
# cc_binary.
cc_binary(
    name = "lc_export_csv_worker",
    srcs = ["lc_export_csv_worker.cc"],
    deps = [
        ":life_cycle_lib",
        "//labm8/cpp:pbutil",
    ],
)

cc_library(
    name = "life_cycle_lib",
    srcs = ["life_cycle_lib.cc"],
    hdrs = ["life_cycle_lib.h"],
    deps = [
        "//datasets/me_db:me_pb_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "life_cycle_lib_test",
    srcs = ["life_cycle_lib_test.cc"],
    deps = [
        ":life_cycle_lib",
        "//datasets/me_db:me_pb_cc",
        "//labm8/cpp:test",
    ],
)

py_binary(
    name = "life_cycle",
    srcs = ["life_cycle.py"],
    data = [":lc_export_csv_worker"],
    visibility = ["//datasets/me_db:__pkg__"],
    deps = [
        "//datasets/me_db:importers",
        "//datasets/me_db:me_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "life_cycle_test",
    srcs = ["life_cycle_test.py"],
    deps = [
        ":life_cycle",
        ":make_dataset",
        "//datasets/me_db:me_pb_py",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_binary(
    name = "make_dataset",
    testonly = 1,
    srcs = ["make_dataset.py"],
    deps = [
        "//labm8:app",
    ],
)

py_test(
    name = "make_dataset_test",
    srcs = ["make_dataset_test.py"],
    deps = [
        ":make_dataset",
        "//labm8:app",
        "//labm8:test",
    ],
)
