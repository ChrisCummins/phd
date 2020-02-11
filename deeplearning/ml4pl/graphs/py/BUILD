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
    "graph_builder_pybind",
    srcs = ["graph_builder_pybind.cc"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_builder",
    ],
)

py_library(
    name = "graph_builder",
    srcs = ["graph_builder.py"],
    data = [":graph_builder_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "graph_builder_test",
    srcs = ["graph_builder_test.py"],
    deps = [
        ":graph_builder",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:test",
    ],
)

pybind_extension(
    "random_graph_builder",
    testonly = 1,
    srcs = ["random_graph_builder.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:random_graph_builder",
    ],
)

py_test(
    name = "random_graph_builder_test",
    srcs = ["random_graph_builder_test.py"],
    data = ["random_graph_builder.so"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:test",
    ],
)
