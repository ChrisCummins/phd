# Protocol buffers for DeepSmith.
#
# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.

load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")

proto_library(
    name = "datastore_pb",
    srcs = ["datastore.proto"],
    deps = [
        ":deepsmith_pb",
        ":service_pb",
    ],
)

python_grpc_library(
    name = "datastore_pb_py",
    visibility = ["//visibility:public"],
    deps = [":datastore_pb"],
)

proto_library(
    name = "deepsmith_pb",
    srcs = ["deepsmith.proto"],
)

python_grpc_library(
    name = "deepsmith_pb_py_grpc",
    visibility = ["//visibility:public"],
    deps = [":deepsmith_pb"],
)

proto_library(
    name = "generator_pb",
    srcs = ["generator.proto"],
    deps = [
        ":deepsmith_pb",
        ":service_pb",
        "//deeplearning/clgen/proto:clgen_pb",
    ],
)

python_grpc_library(
    name = "generator_pb_py_grpc",
    visibility = ["//visibility:public"],
    deps = [":generator_pb"],
)

proto_library(
    name = "harness_pb",
    srcs = ["harness.proto"],
    deps = [
        ":deepsmith_pb",
        ":service_pb",
    ],
)

python_grpc_library(
    name = "harness_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":harness_pb"],
)

proto_library(
    name = "service_pb",
    srcs = ["service.proto"],
)

python_grpc_library(
    name = "service_pb_py",
    visibility = ["//visibility:public"],
    deps = [":service_pb"],
)
