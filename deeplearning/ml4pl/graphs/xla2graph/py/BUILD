# This package contains python bindings for xla2graph.
#
# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
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

load(
    "//third_party/pybind11_bazel:build_defs.bzl",
    "pybind_extension",
)
load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

pybind_extension(
    "xla2graph_pybind",
    srcs = ["xla2graph_pybind.cc"],
    deps = [
        "//deeplearning/ml4pl/graphs/xla2graph:hlo_module_graph_builder",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

py_library(
    name = "xla2graph",
    srcs = ["xla2graph.py"],
    data = [":xla2graph_pybind.so"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_lib_py",
    ],
)

py_test(
    name = "xla2graph_test",
    srcs = ["xla2graph_test.py"],
    data = ["//deeplearning/ml4pl/testing/data/hlo:hlo_protos"],
    deps = [
        ":xla2graph",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:test",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_lib_py",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_lib_py",
    ],
)
