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

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "github_py_pb2",
    protos = ["github.proto"],
    deps = [
        "//third_party/py/protobuf",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":github_py_pb2",
        "//labm8:app",
        "//third_party/py/github",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        ":api",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)
