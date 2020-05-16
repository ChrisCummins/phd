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
    name = "graph_loader",
    srcs = ["graph_loader.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//programl/ml/batch:base_graph_loader",
        "//programl/proto:program_graph_features_py",
        "//programl/proto:program_graph_py",
    ],
)

py_binary(
    name = "train_ggnn",
    srcs = ["train_ggnn.py"],
    deps = [
        ":graph_loader",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//labm8/py:ppar",
        "//programl/ml/batch:rolling_results",
        "//programl/ml/model/ggnn",
        "//programl/ml/model/ggnn:ggnn_batch_builder",
        "//programl/proto:epoch_py",
    ],
)
