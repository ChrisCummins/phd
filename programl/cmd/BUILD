# This package contains the commandline binaries for ProGraML.
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

cc_binary(
    name = "analyze",
    srcs = ["analyze.cc"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:status",
        "//programl/graph/analysis",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "graph2dot",
    srcs = ["graph2dot.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph/format:graphviz_converter",
        "//programl/proto:program_graph_cc",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "graph2json",
    srcs = ["graph2json.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//programl/graph/format:node_link_graph",
        "//programl/proto:program_graph_cc",
        "@com_google_protobuf//:protobuf",
    ],
)

py_binary(
    name = "graph2nx",
    srcs = ["graph2nx.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:pbutil",
        "//programl/graph/format/py:nx_format",
        "//programl/proto:program_graph_py",
    ],
)

cc_binary(
    name = "graph2pb",
    srcs = ["graph2pb.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:program_graph_cc",
        "@com_google_protobuf//:protobuf",
    ],
)

py_binary(
    name = "inst2vec",
    srcs = ["inst2vec.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:pbutil",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto:program_graph_py",
    ],
)

cc_binary(
    name = "llvm2graph",
    srcs = ["llvm2graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:statusor",
        "//programl/ir/llvm",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_options_cc",
        "//third_party/llvm",
    ],
)

cc_binary(
    name = "pb2pbtxt",
    srcs = ["pb2pbtxt.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:ir_cc",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "//programl/proto:src_cc",
        "@com_google_protobuf//:protobuf",
    ],
)
