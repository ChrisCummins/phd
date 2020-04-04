# This package defines mini-batches for models.
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
    name = "batch_data",
    srcs = ["batch_data.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "batch_results",
    srcs = ["batch_results.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
    ],
)

py_library(
    name = "rolling_results",
    srcs = ["rolling_results.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":batch_data",
        ":batch_results",
        "//programl/proto:epoch_py",
    ],
)

py_test(
    name = "rolling_results_test",
    srcs = ["rolling_results_test.py"],
    deps = [
        ":batch_data",
        ":batch_results",
        ":rolling_results",
        "//labm8/py:test",
    ],
)
