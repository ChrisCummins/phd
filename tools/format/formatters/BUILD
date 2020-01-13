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
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format:formatter",
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
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:system",
        "//tools/format:formatter",
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
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format:formatter",
    ],
)

py_library(
    name = "java",
    srcs = ["java.py"],
    data = ["//third_party/java:google_java_format_jar"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format:formatter",
    ],
)

py_library(
    name = "javascript",
    srcs = ["javascript.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//labm8/py:fs",
        "//tools/format:formatter",
    ],
)

py_library(
    name = "json",
    srcs = ["json.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//tools/format:formatter",
    ],
)

py_library(
    name = "python",
    srcs = ["python.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//tools/format:formatter",
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
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//labm8/py:bazelutil",
        "//tools/format:formatter",
    ],
)

py_library(
    name = "sql",
    srcs = ["sql.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//third_party/py/sqlparse",
        "//tools/format:formatter",
    ],
)

py_library(
    name = "suffix_mapping",
    srcs = ["suffix_mapping.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        ":bazel",
        ":cxx",
        ":go",
        ":java",
        ":javascript",
        ":json",
        ":python",
        ":shell",
        ":sql",
        ":text",
        "//labm8/py:app",
    ],
)

py_library(
    name = "text",
    srcs = ["text.py"],
    visibility = ["//tools/format/formatters:__subpackages__"],
    deps = [
        "//tools/format:formatter",
    ],
)
