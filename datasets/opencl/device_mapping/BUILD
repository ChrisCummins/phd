# A dataset of OpenCL Heterogeneous Device Mappings.
#
# Copyright 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["restricted"])

filegroup(
    name = "csv_files",
    srcs = [
        "amd.csv",
        "nvidia.csv",
    ],
)

py_library(
    name = "opencl_device_mapping_dataset",
    srcs = ["opencl_device_mapping_dataset.py"],
    data = [":csv_files"],
    visibility = ["//visibility:public"],
    deps = [
        # "//deeplearning/deeptune/opencl/adversary:opencl_deadcode_inserter",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "opencl_device_mapping_dataset_test",
    srcs = ["opencl_device_mapping_dataset_test.py"],
    deps = [
        ":opencl_device_mapping_dataset",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)
