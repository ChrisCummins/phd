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

pybind_extension(
    "node_link_graph_pybind",
    srcs = ["node_link_graph_pybind.cc"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:string",
        "//programl/graph/format:node_link_graph",
        "//programl/proto:program_graph_cc",
        "@nlohmann_json//:json",
        "@pybind11_json",
    ],
)

py_library(
    name = "nx_format",
    srcs = ["nx_format.py"],
    data = [":node_link_graph_pybind.so"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//programl/proto:program_graph_py",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "nx_format_test",
    srcs = ["nx_format_test.py"],
    deps = [
        ":nx_format",
        "//labm8/py:test",
        "//programl/proto:program_graph_py",
        "//third_party/py/networkx",
    ],
)
