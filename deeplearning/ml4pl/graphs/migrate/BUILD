# Scripts for graph data migrations.
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
#    name = "delete_single_node_graphs",
#    srcs = ["delete_single_node_graphs.py"],
#    deps = [
#        "//deeplearning/ml4pl/graphs:graph_database",
#        "//labm8/py:app",
#    ],
#)
#
#py_binary(
#    name = "migrate_graph_database",
#    srcs = ["migrate_graph_database.py"],
#    deps = [
#        "//deeplearning/ml4pl/graphs:graph_database",
#        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
#        "//labm8/py:app",
#    ],
#)
#
#py_test(
#    name = "migrate_graph_database_test",
#    srcs = ["migrate_graph_database_test.py"],
#    deps = [
#        ":migrate_graph_database",
#        "//labm8/py:test",
#    ],
#)

# TODO(github.com/ChrisCummins/ProGraML/issues/2): Redundant after migration.
py_binary(
    name = "networkx_to_protos",
    srcs = ["networkx_to_protos.py"],
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/testing:__subpackages__",
    ],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

# TODO(github.com/ChrisCummins/ProGraML/issues/2): Redundant after migration.
#py_test(
#    name = "networkx_to_protos_test",
#    srcs = ["networkx_to_protos_test.py"],
#    data = [
#        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs",
#    ],
#    deps = [
#        ":networkx_to_protos",
#        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
#        "//deeplearning/ml4pl/testing:testing_databases",
#        "//labm8/py:decorators",
#        "//labm8/py:test",
#        "//third_party/py/networkx",
#    ],
#)

# TODO(github.com/ChrisCummins/ProGraML/issues/28): Fix me.
#py_binary(
#    name = "one_hot_x_features_to_embeddings",
#    srcs = ["one_hot_x_features_to_embeddings.py"],
#    deps = [
#        "//deeplearning/ml4pl/graphs:graph_database",
#        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
#        "//labm8/py:app",
#        "//labm8/py:humanize",
#        "//labm8/py:prof",
#        "//third_party/py/numpy",
#        "//third_party/py/sqlalchemy",
#    ],
#)
#
#py_binary(
#    name = "reapply_split",
#    srcs = ["reapply_split.py"],
#    deps = [
#        "//deeplearning/ml4pl/graphs:graph_database",
#        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
#        "//labm8/py:app",
#        "//labm8/py:humanize",
#        "//third_party/py/numpy",
#        "//third_party/py/scikit_learn",
#        "//third_party/py/sqlalchemy",
#    ],
#)
#
#py_binary(
#    name = "propagate_split",
#    srcs = ["propagate_split.py"],
#    deps = [
#        "//deeplearning/ml4pl/graphs:graph_database",
#        "//labm8/py:app",
#        "//third_party/py/sqlalchemy",
#    ],
#)
