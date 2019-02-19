# Protocol buffers for Clgen.
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")
load("@org_pubref_rules_protobuf//java:rules.bzl", "java_proto_library")

# Default to public visibility so that X_proto_library targets in other packages
# may depend on these protos using proto_deps. Specifying individual targets as
# public does not work since X_proto_library macros uses intermediate targets
# which must also be public.
package(default_visibility = ["//visibility:public"])

py_proto_library(
    name = "clgen_py_pb2",
    proto_deps = [
        ":corpus_py_pb2",
        ":model_py_pb2",
        ":sampler_py_pb2",
    ],
    protos = ["clgen.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "corpus_py_pb2",
    protos = ["corpus.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "internal_py_pb2",
    proto_deps = [
        ":corpus_py_pb2",
        ":model_py_pb2",
        ":sampler_py_pb2",
    ],
    protos = ["internal.proto"],
    deps = ["//third_party/py/protobuf"],
)

java_proto_library(
    name = "internal_java",
    proto_deps = [
        ":corpus_py_pb2",
        ":model_py_pb2",
        ":sampler_py_pb2",
    ],
    protos = ["internal.proto"],
)

py_proto_library(
    name = "model_py_pb2",
    proto_deps = [":corpus_py_pb2"],
    protos = ["model.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "sampler_py_pb2",
    protos = ["sampler.proto"],
    deps = ["//third_party/py/protobuf"],
)

py_proto_library(
    name = "telemetry_py_pb2",
    protos = ["telemetry.proto"],
    deps = ["//third_party/py/protobuf"],
)
