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

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "datastore_py_pb2",
    proto_deps = [
        ":deepsmith_py_pb2",
        ":service_py_pb2",
    ],
    protos = ["datastore.proto"],
    visibility = ["//visibility:public"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "deepsmith_py_pb2",
    protos = ["deepsmith.proto"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "generator_py_pb2",
    proto_deps = [
        ":deepsmith_py_pb2",
        ":service_py_pb2",
        "//deeplearning/clgen/proto:clgen_py_pb2",
    ],
    protos = ["generator.proto"],
    visibility = ["//visibility:public"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "harness_py_pb2",
    proto_deps = [
        ":deepsmith_py_pb2",
        ":service_py_pb2",
    ],
    protos = ["harness.proto"],
    visibility = ["//visibility:public"],
    with_grpc = True,
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "service_py_pb2",
    protos = ["service.proto"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/protobuf"],
)
