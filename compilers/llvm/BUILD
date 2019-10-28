# A python wrapper around LLVM binaries.
#
# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "build",
    srcs = ["build.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:fs",
        "//labm8:prof",
    ],
)

py_binary(
    name = "clang",
    srcs = ["clang.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "clang_test",
    srcs = ["clang_test.py"],
    data = ["//datasets/benchmarks:nqueens_src"],
    deps = [
        ":clang",
        ":llvm",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:test",
    ],
)

py_binary(
    name = "clang_format",
    srcs = ["clang_format.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang-format",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang-format",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "clang_format_test",
    srcs = ["clang_format_test.py"],
    deps = [
        ":clang_format",
        ":llvm",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_binary(
    name = "llvm_as",
    srcs = ["llvm_as.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-as",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-as",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_binary(
    name = "llvm_dis",
    srcs = ["llvm_dis.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-dis",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-dis",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "llvm_dis_test",
    srcs = ["llvm_dis_test.py"],
    deps = [
        ":llvm_dis",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_binary(
    name = "llvm_link",
    srcs = ["llvm_link.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-link",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-link",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "llvm_link_test",
    srcs = ["llvm_link_test.py"],
    deps = [
        ":llvm",
        ":llvm_link",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:test",
    ],
)

py_library(
    name = "llvm",
    srcs = ["llvm.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
    ],
)

py_binary(
    name = "opt",
    srcs = ["opt.py"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:opt",
        ],
        "//conditions:default": [
            "@llvm_linux//:opt",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:system",
    ],
)

py_test(
    name = "opt_test",
    srcs = ["opt_test.py"],
    deps = [
        ":llvm",
        ":opt",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "opt_util",
    srcs = ["opt_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":llvm",
        ":llvm_as",
        ":opt",
        "//labm8:app",
        "//labm8:humanize",
    ],
)

py_test(
    name = "opt_util_test",
    srcs = ["opt_util_test.py"],
    deps = [
        ":llvm",
        ":opt",
        ":opt_util",
        "//labm8:app",
        "//labm8:test",
    ],
)
