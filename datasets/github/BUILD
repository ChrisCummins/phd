# Root package for code that works with GitHub.
#
# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

proto_library(
    name = "github_pb",
    srcs = ["github.proto"],
)

python_proto_library(
    name = "github_pb_py",
    deps = [
        ":github_pb",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":github_pb_py",
        "//labm8/py:app",
        "//third_party/py/github",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        ":api",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)
