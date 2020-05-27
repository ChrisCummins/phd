# This package contains the code for learning data flow analyses.
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

py_library(
    name = "ggnn",
    srcs = ["ggnn.py"],
    deps = [
        ":dataflow",
        ":ggnn_batch_builder",
        ":graph_loader",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//labm8/py:ppar",
        "//programl/ml/batch:async_batch_builder",
        "//programl/ml/model/ggnn",
        "//programl/proto:checkpoint_py",
        "//programl/proto:epoch_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ggnn_batch_builder",
    srcs = ["ggnn_batch_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//programl/graph/format/py:graph_tuple_builder",
        "//programl/ml/batch:base_batch_builder",
        "//programl/ml/batch:base_graph_loader",
        "//programl/ml/batch:batch_data",
        "//programl/ml/model/ggnn:ggnn_batch",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dataflow",
    srcs = ["dataflow.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_loader",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//labm8/py:ppar",
        "//programl/ml/batch:async_batch_builder",
        "//programl/ml/model/ggnn",
        "//programl/proto:checkpoint_py",
        "//programl/proto:epoch_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "export_logs",
    srcs = ["export_logs.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:google_sheets",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//labm8/py:pdutil",
        "//labm8/py:prof",
        "//programl/proto:epoch_py",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "graph_loader",
    srcs = ["graph_loader.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//programl/graph/format/py:cdfg",
        "//programl/ml/batch:base_graph_loader",
        "//programl/proto:epoch_py",
        "//programl/proto:program_graph_features_py",
        "//programl/proto:program_graph_py",
    ],
)

py_library(
    name = "lstm_batch_builder",
    srcs = ["lstm_batch_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//programl/graph/format/py:graph_serializer",
        "//programl/ml/batch:base_batch_builder",
        "//programl/ml/batch:base_graph_loader",
        "//programl/ml/batch:batch_data",
        "//programl/ml/model/lstm:lstm_batch",
        "//programl/proto:program_graph_features_py",
        "//programl/proto:program_graph_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "train_ncc",
    srcs = ["train_ncc.py"],
    deps = [
        ":dataflow",
        ":graph_loader",
        ":lstm_batch_builder",
        "//deeplearning/ncc:vocabulary",
        "//labm8/py:app",
        "//labm8/py:gpu_scheduler",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//labm8/py:ppar",
        "//programl/ml/batch:async_batch_builder",
        "//programl/ml/model/lstm",
        "//programl/proto:epoch_py",
        "//third_party/py/numpy",
    ],
)

sh_binary(
    name = "overview",
    srcs = ["overview.sh"],
)

py_binary(
    name = "test_vocab",
    srcs = ["test_vocab.py"],
    deps = [
        ":vocabulary",
        "//deeplearning/ncc:vocabulary",
        "//labm8/py:app",
        "//labm8/py:jsonutil",
        "//labm8/py:pbutil",
        "//labm8/py:progress",
        "//programl/proto:node_py",
        "//programl/proto:program_graph_py",
        "//programl/task/dataflow/dataset:pathflag",
    ],
)

py_binary(
    name = "train_ggnn",
    srcs = ["train_ggnn.py"],
    deps = [
        ":dataflow",
        ":ggnn",
        ":vocabulary",
        "//labm8/py:app",
    ],
)

py_library(
    name = "vocabulary",
    srcs = ["vocabulary.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:humanize",
    ],
)
