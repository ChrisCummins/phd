# Protos for describing OpenCL devices.
#
# Copyright 2017-2020 Chris Cummins <chrisc.101@gmail.com>.
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

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "clinfo_pb",
    srcs = ["clinfo.proto"],
)

cc_proto_library(
    name = "clinfo_pb_cc",
    deps = [":clinfo_pb"],
)

python_proto_library(
    name = "clinfo_pb_py",
    deps = [":clinfo_pb"],
)
