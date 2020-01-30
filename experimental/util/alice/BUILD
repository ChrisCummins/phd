# A little command executor.
#
# Copyright (c) 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# alice is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alice.  If not, see <https://www.gnu.org/licenses/>.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")
load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

proto_library(
    name = "alice_pb",
    srcs = [":alice.proto"],
)

python_grpc_library(
    name = "alice_pb_py_grpc",
    visibility = ["//visibility:public"],
    deps = [":alice_pb"],
)

py_library(
    name = "bazel",
    srcs = ["bazel.py"],
    visibility = ["//experimental/util/alice:__subpackages__"],
    deps = [
        ":alice_pb_py_grpc",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:system",
    ],
)

py_test(
    name = "bazel_test",
    srcs = ["bazel_test.py"],
    data = ["//experimental/util/alice/test:dummy_target"],
    deps = [
        ":alice_pb_py_grpc",
        ":bazel",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "git_repo",
    srcs = ["git_repo.py"],
    visibility = ["//experimental/util/alice:__subpackages__"],
    deps = [
        ":alice_pb_py_grpc",
        "//labm8/py:app",
        "//third_party/py/git",
    ],
)

py_test(
    name = "git_repo_test",
    srcs = ["git_repo_test.py"],
    deps = [
        ":git_repo",
        "//labm8/py:test",
        "//third_party/py/git",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [
        ":alice_pb_py_grpc",
        ":git_repo",
        "//labm8/py:app",
        "//third_party/py/grpcio",
    ],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    deps = [
        ":run",
        "//labm8/py:test",
    ],
)

py_library(
    name = "xdata",
    srcs = ["xdata.py"],
    deps = [
        "//labm8/py:app",
    ],
)

py_test(
    name = "xdata_test",
    srcs = ["xdata_test.py"],
    deps = [
        ":xdata",
        "//labm8/py:test",
    ],
)
