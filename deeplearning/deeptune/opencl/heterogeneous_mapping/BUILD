# Port of "Case Study A" from "//docs/2017_09_pact/code:Case Study A.ipynb".
#
# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    visibility = ["//deeplearning/deeptune/opencl/heterogeneous_mapping:__subpackages__"],
    deps = [
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/corpuses:atomizers",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "heterogeneous_mapping",
    srcs = ["heterogeneous_mapping.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//labm8/py:app",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "heterogeneous_mapping_test",
    size = "enormous",
    srcs = ["heterogeneous_mapping_test.py"],
    deps = [
        ":conftest",
        ":heterogeneous_mapping",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//deeplearning/deeptune/opencl/heterogeneous_mapping:__subpackages__"],
    deps = [
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8/py:app",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scikit_learn",
        "//third_party/py/scipy",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":conftest",
        ":utils",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
