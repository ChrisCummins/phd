# Protocol buffers for Clgen.
#
# Copyright (c) 2016-2020 Chris Cummins.
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

load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")
load("@build_stack_rules_proto//java:java_proto_compile.bzl", "java_proto_compile")

proto_library(
    name = "clgen_pb",
    srcs = [
        "clgen.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":model_pb",
        ":sampler_pb",
    ],
)

proto_library(
    name = "corpus_pb",
    srcs = [
        "corpus.proto",
    ],
)

proto_library(
    name = "internal_pb",
    srcs = [
        "internal.proto",
    ],
    deps = [
        ":corpus_pb",
        ":model_pb",
        ":sampler_pb",
    ],
)

proto_library(
    name = "model_pb",
    srcs = [
        "model.proto",
    ],
    deps = [
        ":corpus_pb",
    ],
)

proto_library(
    name = "sampler_pb",
    srcs = [
        "sampler.proto",
    ],
)

proto_library(
    name = "telemetry_pb",
    srcs = [
        "telemetry.proto",
    ],
)

python_grpc_library(
    name = "clgen_pb_py",
    visibility = ["//visibility:public"],
    deps = [
        ":clgen_pb",
        ":corpus_pb",
        ":internal_pb",
        ":model_pb",
        ":sampler_pb",
        ":telemetry_pb",
    ],
)

java_proto_library(
    name = "clgen_pb_java",
    visibility = ["//visibility:public"],
    deps = [
        ":clgen_pb",
        ":corpus_pb",
        ":internal_pb",
        ":model_pb",
        ":sampler_pb",
        ":telemetry_pb",
    ],
)

cc_proto_library(
    name = "internal_pb_cc",
    visibility = ["//visibility:public"],
    deps = [
        ":internal_pb",
    ],
)
