# Data for testing labm8.
#
# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
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

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

package(
    default_testonly = 1,
    default_visibility = ["//labm8:__subpackages__"],
)

proto_library(
    name = "test_protos_pb",
    srcs = ["test_protos.proto"],
)

python_proto_library(
    name = "test_protos_pb_py",
    deps = [":test_protos_pb"],
)

filegroup(
    name = "data1_file",
    srcs = ["data1"],
)

filegroup(
    name = "testdir_files",
    srcs = glob(["testdir/**"]),
)

filegroup(
    name = "makeproj_files",
    srcs = glob(["makeproj/**"]),
)

filegroup(
    name = "empty_file_file",
    srcs = ["empty_file"],
)

filegroup(
    name = "hello_world_file",
    srcs = ["hello_world"],
)

filegroup(
    name = "jsoncache_file",
    srcs = ["jsoncache.json"],
)
