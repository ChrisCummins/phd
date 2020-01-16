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

load("//tools/bzl:exports.bzl", "exports_repo")
load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

exports_files([
    "install.sh",
    "LICENSE",
    "README.md",
    "CNAME",
])

exports_repo(
    name = "export",
    extra_files = [
        "tools/format/install.sh",
    ],
    github_repo = "format",
    move_file_mapping = {
        "tools/format/README.md": "README.md",
        "tools/format/LICENSE": "LICENSE",
        "tools/format/travis.yml": ".travis.yml",
    },
    targets = ["//tools/format/..."],
)

py_library(
    name = "app_paths",
    srcs = ["app_paths.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//:build_info",
        "//labm8/py:app",
        "//third_party/py/appdirs",
    ],
)

py_library(
    name = "default_suffix_mapping",
    srcs = ["default_suffix_mapping.py"],
    visibility = ["//tools/format:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//tools/format/formatters:bazel",
        "//tools/format/formatters:cxx",
        "//tools/format/formatters:go",
        "//tools/format/formatters:java",
        "//tools/format/formatters:javascript",
        "//tools/format/formatters:json",
        "//tools/format/formatters:protobuf",
        "//tools/format/formatters:python",
        "//tools/format/formatters:shell",
        "//tools/format/formatters:sql",
        "//tools/format/formatters:text",
    ],
)

py_binary(
    name = "format",
    srcs = ["format.py"],
    deps = [
        ":app_paths",
        ":default_suffix_mapping",
        ":format_paths",
        ":path_generator",
        ":pre_commit",
        "//labm8/py:app",
    ],
)

bats_test(
    name = "format_test",
    srcs = ["format_test.bats"],
    data = [
        ":format",
        "//labm8/sh:test",
    ],
)

py_library(
    name = "format_paths",
    srcs = ["format_paths.py"],
    deps = [
        ":app_paths",
        ":default_suffix_mapping",
        "//labm8/py:app",
        "//labm8/py:ppar",
        "//labm8/py:shell",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "git_util",
    srcs = ["git_util.py"],
    deps = [
        "//labm8/py:app",
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

py_binary(
    name = "pre_commit",
    srcs = ["pre_commit.py"],
    deps = [
        ":app_paths",
        ":format_paths",
        ":git_util",
        ":path_generator",
        "//:build_info",
        "//labm8/py:app",
        "//labm8/py:fs",
    ],
)
