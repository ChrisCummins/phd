# Machine learning models.
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
    name = "model",
    srcs = ["model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:progress",
        "//programl/ml/batch:batch_data",
        "//programl/ml/batch:batch_results",
        "//programl/proto:checkpoint_py",
        "//programl/proto:epoch_py",
    ],
)

#####################################

py_library(
    name = "classifier_base",
    srcs = ["classifier_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        #        ":base_utils",
        #        ":logger",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        #        "//labm8/py:app",
        #        "//labm8/py:decorators",
        #        "//labm8/py:gpu_scheduler",
        #        "//labm8/py:humanize",
        #        "//labm8/py:progress",
        #        "//programl/ml/epoch",
        #        "//programl/ml/epoch:checkpoints",
        #        "//third_party/py/numpy",
        #        "//third_party/py/pandas",
    ],
)

py_test(
    name = "classifier_base_test",
    size = "enormous",
    srcs = ["classifier_base_test.py"],
    shard_count = 8,
    deps = [
        #        ":classifier_base",
        #        ":log_database",
        #        ":logger",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        #        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        #        "//deeplearning/ml4pl/testing:testing_databases",
        #        "//labm8/py:progress",
        #        "//labm8/py:test",
        #        "//programl/ml/batch:batch_data",
        #        "//programl/ml/epoch",
    ],
)

py_binary(
    name = "copy_database",
    srcs = ["copy_database.py"],
    deps = [
        ":log_database",
        "//labm8/py:app",
        "//labm8/py:prof",
        "//labm8/py:sqlutil",
    ],
)

py_binary(
    name = "export_logs",
    srcs = ["export_logs.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        #        ":log_database",
        #        ":logger",
        #        ":schedules",
        #        "//deeplearning/ml4pl/models/ggnn:ggnn_config",
        #        "//deeplearning/ml4pl/models/lstm",
        #        "//labm8/py:app",
        #        "//labm8/py:google_sheets",
        #        "//labm8/py:pdutil",
        #        "//third_party/py/pandas",
    ],
)

py_library(
    name = "log_analysis",
    srcs = ["log_analysis.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        #        ":export_logs",
        #        ":log_database",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        #        "//labm8/py:app",
        #        "//labm8/py:decorators",
        #        "//labm8/py:progress",
        #        "//third_party/py/matplotlib",
        #        "//third_party/py/numpy",
        #        "//third_party/py/pandas",
        #        "//third_party/py/scikit_learn",
    ],
)

py_test(
    name = "log_analysis_test",
    size = "enormous",
    srcs = ["log_analysis_test.py"],
    shard_count = 8,
    deps = [
        #        ":log_analysis",
        #        ":log_database",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        #        "//deeplearning/ml4pl/testing:random_log_database_generator",
        #        "//deeplearning/ml4pl/testing:testing_databases",
        #        "//labm8/py:decorators",
        #        "//labm8/py:test",
        #        "//third_party/py/numpy",
        #        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "log_database",
    srcs = ["log_database.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        #        "//labm8/py:app",
        #        "//labm8/py:crypto",
        #        "//labm8/py:decorators",
        #        "//labm8/py:google_sheets",
        #        "//labm8/py:humanize",
        #        "//labm8/py:jsonutil",
        #        "//labm8/py:labdate",
        #        "//labm8/py:pdutil",
        #        "//labm8/py:sqlutil",
        #        "//programl/ml/batch:batch_data",
        #        "//programl/ml/epoch",
        #        "//programl/ml/epoch:checkpoints",
        #        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "log_database_test",
    size = "enormous",
    srcs = ["log_database_test.py"],
    shard_count = 8,
    deps = [
        #        ":log_database",
        #        "//deeplearning/ml4pl/testing:random_log_database_generator",
        #        "//deeplearning/ml4pl/testing:testing_databases",
        #        "//labm8/py:app",
        #        "//labm8/py:test",
        #        "//third_party/py/numpy",
        #        "//third_party/py/pandas",
        #        "//third_party/py/pytest",
    ],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
    visibility = ["//deeplearning/ml4pl/models:__subpackages__"],
    deps = [
        #        ":log_database",
        #        ":schedules",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        #        "//labm8/py:app",
        #        "//labm8/py:prof",
        #        "//labm8/py:progress",
        #        "//labm8/py:sqlutil",
        #        "//programl/ml/batch:batch_data",
        #        "//programl/ml/epoch",
        #        "//programl/ml/epoch:checkpoints",
        #        "//third_party/py/pandas",
        #        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "run",
    srcs = ["run.py"],
    visibility = [
        "//deeplearning/ml4pl/experiments:__subpackages__",
        "//deeplearning/ml4pl/models:__subpackages__",
    ],
    deps = [
        #        ":batch_iterator",
        #        ":classifier_base",
        #        ":logger",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        #        "//labm8/py:app",
        #        "//labm8/py:pdutil",
        #        "//labm8/py:prof",
        #        "//labm8/py:progress",
        #        "//programl/ml/epoch",
        #        "//third_party/py/pyfiglet",
    ],
)

py_test(
    name = "run_test",
    size = "enormous",
    srcs = ["run_test.py"],
    shard_count = 8,
    deps = [
        #        ":classifier_base",
        #        ":log_analysis",
        #        ":log_database",
        #        ":run",
        #        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        #        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        #        "//deeplearning/ml4pl/testing:testing_databases",
        #        "//labm8/py:progress",
        #        "//labm8/py:test",
        #        "//programl/ml/batch:batch_data",
        #        "//programl/ml/epoch",
    ],
)

py_library(
    name = "schedules",
    srcs = ["schedules.py"],
    visibility = [
        "//deeplearning/ml4pl/experiments:__subpackages__",
        "//deeplearning/ml4pl/models:__subpackages__",
    ],
    deps = [
        "//labm8/py:app",
    ],
)
