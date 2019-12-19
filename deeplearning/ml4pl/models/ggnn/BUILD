# Gated graph neural network.
#
# Copyright 2019 the ProGraML authors.
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
    name = "ggnn",
    srcs = ["ggnn.py"],
    visibility = ["//deeplearning/ml4pl/experiments:__subpackages__"],
    deps = [
        ":ggnn_config",
        ":ggnn_modules",
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:node_encoder",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:run",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/torch",
    ],
)

py_test(
    name = "ggnn_test",
    size = "enormous",
    timeout = "eternal",
    srcs = ["ggnn_test.py"],
    deps = [
        ":ggnn",
        ":ggnn_config",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:batch_iterator",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models:logger",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:test",
    ],
)

py_library(
    name = "ggnn_config",
    srcs = ["ggnn_config.py"],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        "//labm8/py:app",
    ],
)

py_library(
    name = "ggnn_modules",
    srcs = ["ggnn_modules.py"],
    deps = [
        ":ggnn_config",
        "//labm8/py:app",
        "//third_party/py/torch",
    ],
)
