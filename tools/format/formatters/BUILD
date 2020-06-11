# This package defines the per-syntax formatters.
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
    name = "bazel",
    srcs = ["bazel.py"],
    data = [
        "@com_github_bazelbuild_buildtools//buildifier",
    ],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format/formatters/base:batched_file_formatter",
    ],
)

py_library(
    name = "cxx",
    srcs = ["cxx.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang-format",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang-format",
        ],
    }),
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        ":text",
        "//labm8/py:bazelutil",
        "//tools/format/formatters/base:batched_file_formatter",
    ],
)

py_library(
    name = "go",
    srcs = ["go.py"],
    data = select({
        "//:darwin": [
            "@go_mac//:gofmt",
        ],
        "//conditions:default": [
            "@go_linux//:gofmt",
        ],
    }),
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format/formatters/base:file_formatter",
    ],
)

py_library(
    name = "java",
    srcs = ["java.py"],
    data = ["//third_party/java:google_java_format_jar"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format/formatters/base:batched_file_formatter",
    ],
)

py_library(
    name = "javascript",
    srcs = ["javascript.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//labm8/py:fs",
        "//tools/format/formatters/base:batched_file_formatter",
    ],
)

py_library(
    name = "json",
    srcs = ["json.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//tools/format/formatters/base:file_formatter",
    ],
)

py_library(
    name = "protobuf",
    srcs = ["protobuf.py"],
    data = select({
        "//:darwin": [
            "@prototool_darwin//file",
        ],
        "//conditions:default": [
            "@prototool_linux//file",
        ],
    }),
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format/formatters/base:file_formatter",
    ],
)

py_library(
    name = "python",
    srcs = ["python.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//third_party/py/reorder_python_imports",
        "//tools/format/formatters/base:batched_file_formatter",
        "@black2-spaces//:black",
    ],
)

py_library(
    name = "shell",
    srcs = ["shell.py"],
    data = select({
        "//:darwin": [
            "@shfmt_mac//file",
        ],
        "//conditions:default": [
            "@shfmt_linux//file",
        ],
    }),
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format/formatters/base:batched_file_formatter",
    ],
)

py_library(
    name = "sql",
    srcs = ["sql.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//third_party/py/sqlparse",
        "//tools/format/formatters/base:file_formatter",
    ],
)

py_library(
    name = "text",
    srcs = ["text.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//tools/format/formatters/base:file_formatter",
    ],
)
