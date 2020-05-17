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

py_binary(
    name = "create",
    srcs = ["create.py"],
    data = [
        ":create_cdfg",
        ":create_labels",
    ],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//programl/ir/llvm/py:llvm",
        "//programl/proto:ir_py",
        "//third_party/py/mysql",
    ],
)

cc_binary(
    name = "create_cdfg",
    srcs = ["create_cdfg.cc"],
    copts = ["-fopenmp"],
    deps = [
        ":graph_map",
        "//labm8/cpp:app",
        "//labm8/cpp:fsutil",
        "//labm8/cpp:logging",
        "//programl/graph/format:cdfg",
        "//programl/proto:node_cc",
        "//programl/proto:program_graph_cc",
        "//third_party/openmp",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "create_labels",
    srcs = ["create_labels.cc"],
    deps = [
        ":graph_map",
        "//labm8/cpp:app",
        "//labm8/cpp:fsutil",
        "//labm8/cpp:logging",
        "//programl/graph/analysis:domtree",
        "//programl/graph/analysis:reachability",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "graph_map",
    srcs = ["graph_map.cc"],
    hdrs = ["graph_map.h"],
    copts = ["-fopenmp"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "//programl/graph/analysis:domtree",
        "//programl/graph/analysis:reachability",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "//third_party/tbb",
        "@boost//:filesystem",
        "@com_google_absl//absl/strings:str_format",
    ],
)
