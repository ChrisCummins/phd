# This package contains utility code for writing tests.
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
    name = "generator_flags",
    testonly = 1,
    srcs = ["generator_flags.py"],
    deps = [
        "//labm8/py:app",
    ],
)

py_library(
    name = "random_graph_tuple_generator",
    testonly = 1,
    srcs = ["random_graph_tuple_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":random_networkx_generator",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//labm8/py:test",
    ],
)

py_test(
    name = "random_graph_tuple_generator_test",
    size = "enormous",
    srcs = ["random_graph_tuple_generator_test.py"],
    deps = [
        ":random_graph_tuple_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "random_graph_tuple_database_generator",
    testonly = 1,
    srcs = ["random_graph_tuple_database_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":generator_flags",
        ":random_graph_tuple_generator",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:app",
    ],
)

py_test(
    name = "random_graph_tuple_database_generator_test",
    size = "enormous",
    srcs = ["random_graph_tuple_database_generator_test.py"],
    deps = [
        ":random_graph_tuple_database_generator",
        ":testing_databases",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "random_log_database_generator",
    testonly = 1,
    srcs = ["random_log_database_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":random_graph_tuple_database_generator",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
    ],
)

py_test(
    name = "random_log_database_generator_test",
    size = "enormous",
    srcs = ["random_log_database_generator_test.py"],
    deps = [
        ":random_log_database_generator",
        ":testing_databases",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_library(
    name = "random_networkx_generator",
    testonly = 1,
    srcs = ["random_networkx_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":random_programl_generator",
        "//deeplearning/ml4pl/graphs:programl",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "random_networkx_generator_test",
    size = "enormous",
    srcs = ["random_networkx_generator_test.py"],
    deps = [
        ":random_graph_tuple_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "random_programl_generator",
    testonly = 1,
    srcs = ["random_programl_generator.py"],
    data = [
        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs",
    ],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
    ],
)

py_test(
    name = "random_programl_generator_test",
    srcs = ["random_programl_generator_test.py"],
    deps = [
        ":random_programl_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "random_unlabelled_graph_database_generator",
    testonly = 1,
    srcs = ["random_unlabelled_graph_database_generator.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":generator_flags",
        ":random_programl_generator",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//labm8/py:app",
    ],
)

py_test(
    name = "random_unlabelled_graph_database_generator_test",
    size = "enormous",
    srcs = ["random_unlabelled_graph_database_generator_test.py"],
    deps = [
        ":random_unlabelled_graph_database_generator",
        ":testing_databases",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_library(
    name = "test_annotators",
    srcs = ["test_annotators.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//labm8/py:app",
    ],
)

py_library(
    name = "testing_databases",
    testonly = 1,
    srcs = ["testing_databases.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl:filesystem_paths",
        "//deeplearning/ml4pl:run_id",
        "//labm8/py:app",
        "//labm8/py:sqlutil",
    ],
)

py_test(
    name = "testing_databases_test",
    srcs = ["testing_databases_test.py"],
    deps = [
        ":testing_databases",
        "//labm8/py:sqlutil",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)
