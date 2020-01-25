# This package contains tests for the GGNN model.
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
    name = "fixtures",
    testonly = 1,
    srcs = ["fixtures.py"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models:logger",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
    ],
)

py_test(
    name = "graph_classification_test",
    size = "enormous",
    timeout = "long",
    srcs = ["graph_classification_test.py"],
    shard_count = 16,  # 1440 tests ~= 90 tests / shard
    # Testing models is CPU-intensive, so parallel execution of "nproc" tests
    # can overload the system. Attempt to mitigate this by setting the "cpu"
    # tag to minimize the number of jobs which get executed, though it is
    # unclear to me whether this tag has any effect. See:
    # https://github.com/bazelbuild/bazel/issues/3688
    # https://docs.bazel.build/versions/master/test-encyclopedia.html#other-resources
    tags = ["cpu:4"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:batch_iterator",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models/ggnn",
        "//deeplearning/ml4pl/models/ggnn/test:fixtures",
        "//labm8/py:test",
    ],
)

py_test(
    name = "load_restore_test",
    size = "enormous",
    srcs = ["load_restore_test.py"],
    shard_count = 8,  # 120 tests ~= 15 tests / shard
    tags = ["cpu:4"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:batch_iterator",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models/ggnn",
        "//deeplearning/ml4pl/models/ggnn/test:fixtures",
        "//labm8/py:test",
    ],
)

py_test(
    name = "node_classification_test",
    size = "enormous",
    timeout = "long",
    srcs = ["node_classification_test.py"],
    shard_count = 16,  # 1440 tests ~= 90 tests / shard
    tags = ["cpu:4"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/models:batch_iterator",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models/ggnn",
        "//deeplearning/ml4pl/models/ggnn/test:fixtures",
        "//labm8/py:test",
    ],
)
