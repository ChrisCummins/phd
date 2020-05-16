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
        "//programl/ml/batch:rolling_results",
        "//programl/proto:checkpoint_py",
        "//programl/proto:epoch_py",
    ],
)
