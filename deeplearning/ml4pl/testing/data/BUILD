# Test data for //deeplearning/ml4pl.
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

# A random 100 unlabelled graphs, as a compressed tarball of pickled networkx
# graphs.
#
# Generated using:
#     bazel run //deeplearning/ml4pl/graphs:copy_database -- \
#        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus' \
#        --output_dir=/tmp/graphs \
#        --max_rows=100 \
#        --max_node_count=8000
filegroup(
    name = "100_unlabelled_networkx_graphs",
    testonly = 1,
    srcs = ["100_unlabelled_networkx_graphs.tar.bz2"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
)

# A random 100 unlabelled networkx graphs, as a compressed SQLite graph_database.
#
# Generated using:
#     bazel run //deeplearning/ml4pl/graphs:copy_database -- \
#        --input_db='file:///var/phd/db/cc1.mysql?ml4pl_unlabelled_corpus' \
#        --output_db='sqlite:////tmp/graphs.db' \
#        --max_rows=100 \
#        --max_node_count=8000
filegroup(
    name = "100_unlabelled_networkx_graphs_db",
    testonly = 1,
    srcs = ["100_unlabelled_networkx_graphs.db.tar.bz2"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
)
