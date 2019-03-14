# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of libcecl.
#
# libcecl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libcecl.  If not, see <https://www.gnu.org/licenses/>.

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "libcecl_cc_pb2",
    proto_deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
    ],
    protos = ["libcecl.proto"],
)

py_proto_library(
    name = "libcecl_py_pb2",
    proto_deps = [
        "//gpu/clinfo/proto:clinfo_py_pb2",
    ],
    protos = ["libcecl.proto"],
    deps = ["//third_party/py/protobuf"],
)
