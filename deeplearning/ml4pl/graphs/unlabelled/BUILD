# Unlabelled program graphs.
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

# TODO(github.com/ChrisCummins/ProGraML/issues/28): Fix me.
#py_binary(
#    name = "make_unlabelled_graph_dataset",
#    srcs = ["make_unlabelled_graph_dataset.py"],
#    visibility = ["//visibility:public"],
#    deps = [
#        "//deeplearning/ml4pl/ir:ir_database",
#        "//deeplearning/ml4pl/graphs:database_exporters",
#        "//deeplearning/ml4pl/graphs:graph_database",
#        "//deeplearning/ml4pl/graphs/llvm2graph/legacy:graph_builder",
#        "//labm8/py:app",
#        "//labm8/py:prof",
#    ],
#)

py_binary(
    name = "split",
    srcs = ["split.py"],
    deps = [
        ":unlabelled_graph_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/ir:split",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "split_test",
    srcs = ["split_test.py"],
    deps = [
        ":unlabelled_graph_database",
        "//deeplearning/ml4pl/graphs/unlabelled:split",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/ir:split",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "unlabelled_graph_database",
    srcs = ["unlabelled_graph_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:jsonutil",
        "//labm8/py:labdate",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "unlabelled_graph_database_test",
    srcs = ["unlabelled_graph_database_test.py"],
    deps = [
        ":unlabelled_graph_database",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "unlabelled_graph_database_exporter",
    srcs = ["unlabelled_graph_database_exporter.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":unlabelled_graph_database",
        "//deeplearning/ml4pl/graphs:programl",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:labtypes",
        "//labm8/py:ppar",
        "//labm8/py:progress",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "unlabelled_graph_database_exporter_test",
    srcs = ["unlabelled_graph_database_exporter_test.py"],
    deps = [
        ":unlabelled_graph_database",
        ":unlabelled_graph_database_exporter",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/testing:random_unlabelled_graph_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//labm8/py:test",
    ],
)
