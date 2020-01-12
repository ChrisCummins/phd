# This package contains the 'format' program for automated code formatting.
#
# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@subpar//:subpar.bzl", "par_binary")

py_library(
    name = "formatter_executor",
    srcs = ["formatter_executor.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:shell",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
        "//tools/format/formatters:suffix_mapping",
    ],
)

par_binary(
    name = "format",
    srcs = ["format.py"],
    deps = [
        ":formatter_executor",
        ":path_generator",
        "//:build_info",
        "//labm8/py:app",
        "//third_party/py/appdirs",
        "//third_party/py/fasteners",
        "//tools/format/formatters:suffix_mapping",
    ],
)

py_library(
    name = "path_generator",
    srcs = ["path_generator.py"],
    deps = [
        "//labm8/py:app",
    ],
)

py_test(
    name = "path_generator_test",
    srcs = ["path_generator_test.py"],
    deps = [
        ":path_generator",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)
