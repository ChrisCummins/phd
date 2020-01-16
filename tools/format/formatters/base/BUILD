# This package defines the base classes for implementing formatters.
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

py_library(
    name = "base_formatter",
    srcs = ["base_formatter.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:fs",
        "//third_party/py/fasteners",
        "//tools/format:app_paths",
    ],
)

py_library(
    name = "batched_file_formatter",
    srcs = ["batched_file_formatter.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        ":base_formatter",
        ":file_formatter",
        "//labm8/py:app",
    ],
)

py_library(
    name = "file_formatter",
    srcs = ["file_formatter.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        ":base_formatter",
        "//labm8/py:app",
    ],
)
