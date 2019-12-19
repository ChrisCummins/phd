# LSTM models.
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

py_library(
    name = "graph_lstm",
    srcs = ["graph_lstm.py"],
    deps = [
        ":lstm_base",
        ":lstm_utils",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/seq:graph2seq",
        "//deeplearning/ml4pl/seq:ir2seq",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "graph_lstm_test",
    size = "enormous",
    srcs = ["graph_lstm_test.py"],
    deps = [
        ":graph_lstm",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/models:batch_iterator",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models:logger",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:random_unlabelled_graph_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:progress",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "lstm",
    srcs = ["lstm.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":graph_lstm",
        ":node_lstm",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:run",
        "//labm8/py:app",
    ],
)

py_library(
    name = "lstm_base",
    srcs = ["lstm_base.py"],
    deps = [
        ":lstm_utils",
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/seq:graph2seq",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:progress",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "lstm_utils",
    srcs = ["lstm_utils.py"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "node_lstm",
    srcs = ["node_lstm.py"],
    deps = [
        ":lstm_base",
        ":lstm_utils",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//deeplearning/ml4pl/models:batch",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/seq:graph2seq",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "node_lstm_test",
    size = "enormous",
    srcs = ["node_lstm_test.py"],
    deps = [
        ":node_lstm",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/models:batch_iterator",
        "//deeplearning/ml4pl/models:classifier_base",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models:logger",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:random_unlabelled_graph_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:progress",
        "//labm8/py:test",
    ],
)
